"""Generate Huikang-format reasoning CoTs for every puzzle newly solved by
probe_v2.  Output goes to a fresh directory that can be merged with
Huikang's reasoning/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))

from extended_solver import OPS_BY_NAME, num_to_digits  # type: ignore


PROBE_RESULTS = PROJECT_ROOT / "training" / "opd" / "solvers" / "runs" / "probe_v2_results.jsonl"
PROBLEMS_DIR = PROJECT_ROOT / "resource" / "winning-solution" / "problems"
OUT_DIR = PROJECT_ROOT / "training" / "opd" / "solvers" / "runs" / "new_reasoning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


OP_TEXT = {
    "add": "a + b", "sub_ab": "a - b", "sub_ba": "b - a", "abs_diff": "|a - b|",
    "mul": "a * b", "concat": "a || b", "rev_concat": "b || a",
    "div_ab": "a / b", "div_ba": "b / a", "mod_ab": "a mod b", "mod_ba": "b mod a",
    "digit_add": "digit-wise (a_i + b_i) mod base",
    "digit_sub": "digit-wise (a_i - b_i) mod base",
    "digit_mul": "digit-wise (a_i * b_i) mod base",
    "digit_xor": "digit-wise (a_i XOR b_i)",
    "digit_max": "digit-wise max(a_i, b_i)",
    "digit_min": "digit-wise min(a_i, b_i)",
    "add_plus_1": "a + b + 1", "add_minus_1": "a + b - 1",
    "mul_plus_1": "a * b + 1", "mul_minus_1": "a * b - 1",
    "concat_no_op": "(query operator unseen → keep both chunks)",
}


def load_problem(pid: str):
    fp = PROBLEMS_DIR / f"{pid}.jsonl"
    with open(fp) as f:
        return json.loads(f.readline())


def fmt_op_compute(op_name: str, a: int, b: int, base: int) -> tuple[int, str]:
    """Return (result_value, human-readable computation string)."""
    fn = OPS_BY_NAME.get(op_name)
    if fn is None:
        return 0, ""
    result = fn(a, b, base)
    if op_name == "add":
        return result, f"{a} + {b} = {result}"
    if op_name == "sub_ab":
        return result, f"{a} - {b} = {result}"
    if op_name == "sub_ba":
        return result, f"{b} - {a} = {result}"
    if op_name == "abs_diff":
        return result, f"|{a} - {b}| = {result}"
    if op_name == "mul":
        return result, f"{a} * {b} = {result}"
    if op_name == "concat":
        return result, f"{a} || {b} = {result}"
    if op_name == "rev_concat":
        return result, f"{b} || {a} = {result}"
    return result, f"{op_name}({a},{b}) = {result}"


def make_reasoning(pid: str, result: dict) -> str:
    """Render a CoT for one solved puzzle."""
    data = load_problem(pid)
    examples = data["examples"]
    q = data["question"]
    answer = data["answer"]
    base = result["config"][0]
    post_rev = result["config"][1]
    mapping = result["mapping"]
    op_assign = result["op_assign"]

    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # ----- Measurement step (process-supervision opening) -----
    alphabet = sorted(set(
        c for e in examples for c in e["input_value"] + e["output_value"] if c not in {"+", "-", "*", "/"}
    ) | set(c for c in q if c not in {"+", "-", "*", "/"}))
    rhs_lens = [len(e["output_value"]) for e in examples]
    lines.append("[Measure the puzzle structure]")
    lines.append(f"  Distinct non-operator symbols across examples and query: {alphabet}")
    lines.append(f"  Output lengths per example: {rhs_lens}")
    lines.append(f"  Alphabet size = {len(alphabet)}")
    if len(alphabet) <= 12 and max(rhs_lens) - min(rhs_lens) > 0:
        lines.append("  Lengths vary → outputs are computed; alphabet ≤ 10-12 → likely a digit cipher.")
    elif len(alphabet) > 12:
        lines.append("  Alphabet larger than digit count → not a pure base-10 cipher.")
    elif max(rhs_lens) == min(rhs_lens):
        lines.append("  Constant output length → likely permutation/concat pattern.")
    lines.append("")

    # ----- Differential probe (Δinput → Δoutput) -----
    def _hamming(a: str, b: str) -> int:
        if len(a) != len(b):
            n = min(len(a), len(b))
            return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
        return sum(1 for x, y in zip(a, b) if x != y)

    def _nonop(s: str) -> str:
        return s[:2] + s[3:] if len(s) == 5 else s

    amp = col = pres = 0
    pair_lines: list[str] = []
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            dp = _hamming(_nonop(examples[i]["input_value"]), _nonop(examples[j]["input_value"]))
            dc = _hamming(examples[i]["output_value"], examples[j]["output_value"])
            if dc > dp: amp += 1
            elif dc < dp: col += 1
            elif dp > 0: pres += 1
            if len(pair_lines) < 3:
                pair_lines.append(f"    ex{i} vs ex{j}: |ΔP|={dp}, |ΔC|={dc}")
    lines.append("[Differential probe: how do input changes propagate to output?]")
    for pl in pair_lines:
        lines.append(pl)
    lines.append(f"  Aggregate: amplify={amp}, preserve={pres}, collapse={col}")
    if amp >= pres and amp >= col:
        lines.append("  Δ-amplification dominates → output is computed (carry/arithmetic-like), not copied.")
    elif pres > amp and pres > col:
        lines.append("  Δ is preserved 1-to-1 → positional / digit-wise rule (no carry).")
    elif col > 0:
        lines.append("  Δ-collapse observed → output projects/combines bits of input.")
    lines.append("")

    # ----- Linear / positional correlation probe -----
    rhs_uniform = len(set(rhs_lens)) == 1
    lock_perm: list = []
    if rhs_uniform and all(len(e["input_value"]) == 5 for e in examples):
        L = 4
        R = rhs_lens[0]
        for j in range(R):
            cand = None
            for ii in range(L):
                if all(examples[k]["output_value"][j] == _nonop(examples[k]["input_value"])[ii] for k in range(len(examples))):
                    cand = ii
                    break
            lock_perm.append(cand)
    lines.append("[Linear/positional probe: does output[j] track input[i] across all examples?]")
    if lock_perm and all(x is not None for x in lock_perm):
        lines.append(f"  Position lock found: output positions map to input positions {lock_perm} for ALL examples.")
        lines.append("  This is a near-certain copy/permutation signature.")
    elif lock_perm:
        partial = [x for x in lock_perm if x is not None]
        if partial:
            lines.append(f"  Partial lock: some output positions track input positions ({len(partial)}/{len(lock_perm)}).")
        else:
            lines.append("  No fixed positional copy holds across all examples → output is not a permutation.")
    else:
        lines.append("  Output length varies across examples → no global positional lock to test.")
    lines.append("")

    # ----- Hypothesis & solution -----
    lines.append(f"[Hypothesis: base-{base} digit cipher" + (" with reverse-result post-processing" if post_rev else "") + "]")
    if mapping:
        lines.append("Inferred character → digit mapping:")
        for c, d in sorted(mapping.items(), key=lambda kv: kv[1]):
            lines.append(f"  '{c}' = {d}")
    lines.append("")
    lines.append("Inferred operator-symbol → operation mapping:")
    for sym, op in sorted(op_assign.items()):
        lines.append(f"  '{sym}' = {OP_TEXT.get(op, op)}")
    lines.append("")

    # ----- Verification on each example -----
    if mapping:
        inv = {d: c for c, d in mapping.items()}
        lines.append("Verify on each example:")
        for e in examples:
            inp = e["input_value"]
            out = e["output_value"]
            if len(inp) != 5:
                continue
            op_sym = inp[2]
            op_name = op_assign.get(op_sym)
            if op_name and all(c in mapping for c in inp[:2] + inp[3:]):
                a = mapping[inp[0]] * base + mapping[inp[1]]
                b = mapping[inp[3]] * base + mapping[inp[4]]
                result_val, comp_str = fmt_op_compute(op_name, a, b, base)
                # Encode result back to chars
                from extended_solver import num_to_digits
                if op_name in ("digit_add", "digit_sub", "digit_mul",
                               "digit_xor", "digit_max", "digit_min"):
                    rd = (result_val // base, result_val % base)
                elif op_name in ("concat", "rev_concat"):
                    rd = (result_val // (base ** 3),
                          (result_val // (base ** 2)) % base,
                          (result_val // base) % base,
                          result_val % base)
                else:
                    rd = num_to_digits(result_val, base)
                encoded = "".join(inv.get(d, "?") for d in rd)
                mark = "✓ matches" if encoded == out else "(mismatch)"
                lines.append(f"  {inp} = {out}:  {a}, {b} → {comp_str} → encode → {encoded}  {mark}")
            else:
                lines.append(f"  {inp} = {out}:  (cipher-free op '{op_name}')")
        lines.append("")

    # ----- Apply to query -----
    lines.append(f"Apply the rule to the query: {q}")
    op_sym = q[2]
    op_name = op_assign.get(op_sym, "concat_no_op")
    if mapping and all(c in mapping for c in q[:2] + q[3:]):
        a = mapping[q[0]] * base + mapping[q[1]]
        b = mapping[q[3]] * base + mapping[q[4]]
        _, comp_str = fmt_op_compute(op_name, a, b, base)
        lines.append(f"  Operands under mapping: {a}, {b}")
        lines.append(f"  Apply '{op_name}': {comp_str}")
        lines.append(f"  Encode result digits back to chars → {answer}")
    else:
        lines.append(f"  Operator '{op_sym}' has rule '{op_name}'.")
        lines.append(f"  Result: {answer}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}.")
    lines.append(f"The answer is \\boxed{{{answer}}}.")
    return "\n".join(lines)


def main():
    if not PROBE_RESULTS.exists():
        print(f"No probe results yet at {PROBE_RESULTS}")
        sys.exit(1)
    n = 0
    with open(PROBE_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("solved"):
                continue
            pid = r["id"]
            text = make_reasoning(pid, r)
            out_path = OUT_DIR / f"{pid}.txt"
            out_path.write_text(text)
            n += 1
    print(f"Wrote {n} new reasoning files to {OUT_DIR}")


if __name__ == "__main__":
    main()
