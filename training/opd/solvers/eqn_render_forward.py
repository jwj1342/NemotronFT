"""Operator-fingerprinting CoT renderer for equation_numeric (plain digits, no cipher).
Stateless: the CoT TESTS candidate operations against the examples (local arithmetic the
model already does), finds the one consistent op, and applies it — explicitly refusing
to assume the glyph's conventional meaning. This is the unified 'fingerprint the operator'
method (same as cryptarithm concat, here for the no-cipher case)."""
from __future__ import annotations
import sys, re
from typing import Optional, Tuple
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crd_solver import _apply_op, ALL_OPS  # type: ignore

EX = re.compile(r"^\s*(\d+)(\D)(\d+)\s*=\s*(\S+)\s*$")
QY = re.compile(r"determine the result for:\s*(\d+)(\D)(\d+)\s*$")
_digits = lambda s: "".join(c for c in str(s) if c.isdigit())
_OP_HUMAN = {"add": "a+b", "add+1": "a+b+1", "add-1": "a+b-1", "sub": "a-b", "rsub": "b-a",
             "abs_sub": "|a-b|", "neg_abs": "-|a-b|", "maxmod": "max%min", "mul": "a*b",
             "mul+1": "a*b+1", "mul-1": "a*b-1", "concat": "a then b (joined)",
             "rconcat": "b then a (joined)"}


def render_eqn(prompt: str, answer: Optional[str] = None) -> Optional[Tuple[str, str]]:
    exs = []
    for ln in prompt.splitlines():
        m = EX.match(ln)
        if m:
            exs.append((int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)))
    qm = QY.search(prompt)
    if not qm:
        return None
    qa, qg, qb = int(qm.group(1)), qm.group(2), int(qm.group(3))
    gex = [(a, b, res) for (a, g, b, res) in exs if g == qg]
    if not gex:
        return None  # guess (op absent) — not fingerprintable
    # find ops magnitude-consistent across all examples of this glyph
    cands = [op for op in ALL_OPS
             if all(_apply_op(op, a, b) is not None and _digits(_apply_op(op, a, b)) == _digits(res)
                    for (a, b, res) in gex)]
    if not cands:
        return None
    # need a unique predicted magnitude
    qvals = {_digits(_apply_op(op, qa, qb)) for op in cands}
    if len(qvals) != 1:
        return None
    op = cands[0]
    raw = _apply_op(op, qa, qb)
    # infer sign/encoding from an example whose op-result is negative (if any)
    pred = str(raw)
    for (a, b, res) in gex:
        v = _apply_op(op, a, b)
        if v is not None and v < 0:
            # how is the negative written? capture the non-digit prefix
            pref = res[:len(res) - len(_digits(res))]
            if raw < 0:
                pred = pref + _digits(raw)
            break
    lines = [f"The operators are disguised — I read '{qg}' from the examples, not its usual meaning.",
             f"Testing operations against the examples that use '{qg}':"]
    for (a, b, res) in gex[:3]:
        lines.append(f"  {a}{qg}{b} = {res}  → {_OP_HUMAN[op]} gives {_apply_op(op, a, b)}. consistent.")
    lines.append(f"So '{qg}' = {_OP_HUMAN[op]}.")
    lines.append(f"Apply to {qa}{qg}{qb}: {_OP_HUMAN[op]} = {pred}.")
    return "\n".join(lines), pred


if __name__ == "__main__":
    import json, glob, csv
    PROJECT = Path("/scratch/jwj/NemotronFT")
    csv.field_size_limit(10**7)
    prompts = {r["id"]: r["prompt"] for r in csv.DictReader(open(PROJECT / "resource/winning-solution/train.csv"))}
    shown = 0
    for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
        for line in open(shard):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["category"] != "equation_numeric_deduce" or r["id"] not in prompts:
                continue
            out = render_eqn(prompts[r["id"]], str(r["answer"]))
            if out and not r["correct"]:
                cot, pred = out
                print("=" * 80)
                print(f"id={r['id']}  true={r['answer']}  pred={pred}  match={pred==str(r['answer'])}  baseline=WRONG")
                print("PROMPT:\n" + prompts[r["id"]][:300])
                print("--- FINGERPRINT CoT ---\n<think>\n" + cot + "\n</think>\n\\boxed{" + pred + "}")
                shown += 1
                if shown >= 4:
                    break
        if shown >= 4:
            break
