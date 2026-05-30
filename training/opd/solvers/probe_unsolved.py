"""Probe unsolved puzzles in the largest 'looks-like-arithmetic' bucket.

For each unsolved puzzle in this bucket, try a richer set of operations than
Huikang's 5-op DSL: digit-wise variants, base-N variants, modular variants,
reverse-input/reverse-output composition.

Goal: discover which extension recovers the most coverage.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))

from category import detect_category  # noqa: E402
from measurements import parse_prompt, measure  # noqa: E402


# ---------------------------------------------------------------------------
# Extended ops on (a, b) → int (or string), at digit-cipher level
# ---------------------------------------------------------------------------


def num_digits(n: int, base: int) -> tuple[int, ...]:
    if n < 0:
        return ()
    if n == 0:
        return (0,)
    out = []
    while n > 0:
        out.append(n % base)
        n //= base
    return tuple(reversed(out))


def try_op(op_name: str, a_digits: tuple[int, int], b_digits: tuple[int, int], base: int = 10) -> tuple[int, ...] | None:
    """Apply the named op to the two-digit operands; return result digit tuple or None."""
    a = a_digits[0] * base + a_digits[1]
    b = b_digits[0] * base + b_digits[1]
    if op_name == "add":
        return num_digits(a + b, base)
    if op_name == "sub_ab":
        if a < b: return None
        return num_digits(a - b, base)
    if op_name == "sub_ba":
        if b < a: return None
        return num_digits(b - a, base)
    if op_name == "abs_diff":
        return num_digits(abs(a - b), base)
    if op_name == "mul":
        return num_digits(a * b, base)
    if op_name == "concat":
        return a_digits + b_digits
    if op_name == "rev_concat":
        return b_digits + a_digits
    if op_name == "div_ab":
        if b == 0 or a % b != 0: return None
        return num_digits(a // b, base)
    if op_name == "div_ba":
        if a == 0 or b % a != 0: return None
        return num_digits(b // a, base)
    if op_name == "mod_ab":
        if b == 0: return None
        return num_digits(a % b, base)
    if op_name == "mod_ba":
        if a == 0: return None
        return num_digits(b % a, base)
    if op_name == "digit_add_mod":
        return ((a_digits[0] + b_digits[0]) % base, (a_digits[1] + b_digits[1]) % base)
    if op_name == "digit_sub_mod":
        return ((a_digits[0] - b_digits[0]) % base, (a_digits[1] - b_digits[1]) % base)
    if op_name == "digit_xor":
        if base != 10: return None
        return (a_digits[0] ^ b_digits[0], a_digits[1] ^ b_digits[1])
    if op_name == "digit_mul_mod":
        return ((a_digits[0] * b_digits[0]) % base, (a_digits[1] * b_digits[1]) % base)
    if op_name == "max_concat":
        ma = max(a_digits); mb = max(b_digits)
        return (ma, mb)
    if op_name == "min_concat":
        ma = min(a_digits); mb = min(b_digits)
        return (ma, mb)
    if op_name == "digit_max":
        return (max(a_digits[0], b_digits[0]), max(a_digits[1], b_digits[1]))
    if op_name == "digit_min":
        return (min(a_digits[0], b_digits[0]), min(a_digits[1], b_digits[1]))
    return None


ALL_OPS = [
    "add", "sub_ab", "sub_ba", "abs_diff", "mul",
    "concat", "rev_concat",
    "div_ab", "div_ba", "mod_ab", "mod_ba",
    "digit_add_mod", "digit_sub_mod", "digit_mul_mod", "digit_xor",
    "max_concat", "min_concat", "digit_max", "digit_min",
]


def solve_with_op(puzzle, op_name: str, base: int, post_reverse: bool = False) -> dict | None:
    """Try to solve the puzzle assuming all example operators share this op.

    Searches over char→digit bijection. Returns the mapping + query answer if found.
    """
    if puzzle.op_position is None:
        return None
    # Collect digit chars (lhs non-op + rhs + query)
    alphabet: set[str] = set()
    for e in puzzle.parsed.examples:
        for i, c in enumerate(e.lhs):
            if i != puzzle.op_position:
                alphabet.add(c)
        alphabet |= set(e.rhs)
    qa = puzzle.parsed.query[:puzzle.op_position] + puzzle.parsed.query[puzzle.op_position + 1:]
    alphabet |= set(qa)
    chars = sorted(alphabet)
    if len(chars) > base:
        return None

    # Use ALL examples (treat each example op independently with the same hypothesis op
    # only when matching; for cryptarithm_guess where q_op is unique, use *all* examples
    # under the assumption the rule is uniform — most ambiguity lives in operator mapping)
    examples_parsed = []
    for e in puzzle.parsed.examples:
        nol = e.lhs[:puzzle.op_position] + e.lhs[puzzle.op_position + 1:]
        if len(nol) != 4:
            continue
        examples_parsed.append((nol, e.rhs))
    if not examples_parsed:
        return None

    # DFS over assignments
    used = [False] * base
    assign: dict[str, int] = {}
    found: list[dict] = []

    def check_all() -> bool:
        for nol, rhs in examples_parsed:
            a_digs = (assign[nol[0]], assign[nol[1]])
            b_digs = (assign[nol[2]], assign[nol[3]])
            result = try_op(op_name, a_digs, b_digs, base)
            if result is None:
                return False
            if post_reverse:
                result = tuple(reversed(result))
            if len(result) != len(rhs):
                return False
            for c, d in zip(rhs, result):
                if c in assign:
                    if assign[c] != d:
                        return False
                else:
                    if used[d]:
                        return False
        return True

    # Hard cap on branches explored — bail out early if a configuration is hopeless.
    counter = [0]
    BRANCH_CAP = 50_000

    def backtrack(i: int) -> None:
        if len(found) >= 2:
            return
        counter[0] += 1
        if counter[0] > BRANCH_CAP:
            return
        if i == len(chars):
            if check_all():
                found.append(dict(assign))
            return
        c = chars[i]
        for d in range(base):
            if used[d]:
                continue
            assign[c] = d
            used[d] = True
            ok = True
            for nol, rhs in examples_parsed:
                if not all(x in assign for x in nol + rhs):
                    continue
                a_digs = (assign[nol[0]], assign[nol[1]])
                b_digs = (assign[nol[2]], assign[nol[3]])
                result = try_op(op_name, a_digs, b_digs, base)
                if result is None:
                    ok = False; break
                if post_reverse:
                    result = tuple(reversed(result))
                if len(result) != len(rhs):
                    ok = False; break
                if any(assign[c2] != d2 for c2, d2 in zip(rhs, result)):
                    ok = False; break
            if ok:
                backtrack(i + 1)
            used[d] = False
            del assign[c]

    backtrack(0)

    if len(found) != 1:
        return None

    mapping = found[0]
    # apply to query
    a_digs = (mapping.get(qa[0]), mapping.get(qa[1]))
    b_digs = (mapping.get(qa[2]), mapping.get(qa[3]))
    if None in a_digs or None in b_digs:
        return None
    result = try_op(op_name, a_digs, b_digs, base)  # type: ignore
    if result is None:
        return None
    if post_reverse:
        result = tuple(reversed(result))
    inv = {d: c for c, d in mapping.items()}
    out_chars = []
    for d in result:
        if d not in inv:
            return None
        out_chars.append(inv[d])
    return {
        "op_name": op_name,
        "base": base,
        "post_reverse": post_reverse,
        "mapping": mapping,
        "predicted_answer": "".join(out_chars),
    }


class PuzzleCtx:
    def __init__(self, parsed):
        self.parsed = parsed
        # determine op_position and q_op
        m = measure(parsed)
        self.op_position = m.op_position
        if m.op_position is not None and len(parsed.query) >= m.op_position + 1:
            self.q_op = parsed.query[m.op_position]
        else:
            self.q_op = None


def load_problems():
    out = []
    with open(PROJECT_ROOT / "resource" / "winning-solution" / "problems.jsonl") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def load_prompt(pid: str) -> str | None:
    fp = PROJECT_ROOT / "resource" / "winning-solution" / "problems" / f"{pid}.jsonl"
    if not fp.exists():
        return None
    with open(fp) as f:
        rec = json.loads(f.readline())
    return rec.get("prompt")


def load_problem_with_answer(pid: str):
    fp = PROJECT_ROOT / "resource" / "winning-solution" / "problems" / f"{pid}.jsonl"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.loads(f.readline())


import time
import multiprocessing as mp


def solve_one_target(t: dict) -> dict:
    """Worker: try all (op, base, post_rev) configs on one puzzle."""
    ctx = PuzzleCtx(t["parsed"])
    if ctx.q_op is None:
        return {"id": t["id"], "solved": False, "config": None}
    gt = str(t["answer"])
    for op in ALL_OPS:
        for base in (10, 9, 8, 11):
            for post_rev in (False, True):
                res = solve_with_op(ctx, op, base, post_rev)
                if res and res["predicted_answer"] == gt:
                    return {"id": t["id"], "solved": True, "config": (op, base, post_rev)}
    return {"id": t["id"], "solved": False, "config": None}


def main() -> None:
    problems = load_problems()
    # Focus on unsolved cryptarithm with the most common arithmetic-looking signature
    targets = []
    for p in problems:
        if p.get("category") not in ("cryptarithm_deduce", "cryptarithm_guess"):
            continue
        if p.get("status") != "rule_unknown":
            continue
        rec = load_problem_with_answer(p["id"])
        if rec is None or "prompt" not in rec or "answer" not in rec:
            continue
        parsed = parse_prompt(rec["prompt"], answer=rec.get("answer"))
        if parsed is None:
            continue
        m = measure(parsed)
        op_is_arith = m.op_position is not None and all(c in "+-*/" for c in m.op_chars_set)
        if not (m.hyp_digit_cipher_arith and op_is_arith and 9 <= m.alphabet_size <= 10):
            continue
        targets.append({"id": p["id"], "parsed": parsed, "answer": str(rec["answer"]), "m": m})

    print(f"Targeting {len(targets)} unsolved cryptarithm puzzles in arithmetic bucket")

    # Parallelize across CPU cores. SLURM_CPUS_PER_TASK is set on compute nodes.
    import os as _os
    n_workers = int(_os.environ.get("SLURM_CPUS_PER_TASK", str(min(8, mp.cpu_count()))))
    print(f"Using {n_workers} workers (mp.cpu_count={mp.cpu_count()})", flush=True)
    t0 = time.time()
    op_success: Counter = Counter()
    config_success: Counter = Counter()
    solved = 0
    solved_ids: list[str] = []
    completed = 0
    with mp.Pool(n_workers) as pool:
        for res in pool.imap_unordered(solve_one_target, targets, chunksize=2):
            completed += 1
            if res["solved"]:
                solved += 1
                op, base, post_rev = res["config"]
                op_success[op] += 1
                config_success[(op, base, post_rev)] += 1
                solved_ids.append(res["id"])
            if completed % 20 == 0 or completed == len(targets):
                dt = time.time() - t0
                print(f"[{completed}/{len(targets)}] solved={solved} elapsed={dt:.1f}s rate={completed/dt:.2f}/s", flush=True)

    print(f"\nNewly solved: {solved} / {len(targets)} = {solved/len(targets)*100:.1f}%")
    print(f"\n=== Per-op success counts ===")
    for op, n in op_success.most_common():
        print(f"  {op:20s} {n}")
    print(f"\n=== Per (op, base, post_reverse) ===")
    for (op, base, rev), n in config_success.most_common(15):
        print(f"  {op:20s} base={base} rev={rev}  {n}")


if __name__ == "__main__":
    main()
