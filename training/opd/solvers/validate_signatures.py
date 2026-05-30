"""Validate that measurement signatures cluster by ground-truth rule family.

Ground truth source: resource/winning-solution/investigations/*.txt for the
puzzles Huikang's investigator successfully solved (with the operator name
explicitly written in the investigation file).

We then run our measurement DSL on each of those puzzles and check whether
puzzles labelled 'add' have similar signatures (close in measurement space),
puzzles labelled 'mul' have similar signatures, etc.  If yes → the signature
DSL actually carries useful information.  If no → it's noise.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))

from category import detect_category  # noqa: E402
from measurements import parse_prompt, measure  # noqa: E402


def load_huikang_investigations() -> dict[str, dict]:
    """Return {pid: {op_name: ..., category: ..., rule_line: ...}}."""
    inv_dir = PROJECT_ROOT / "resource" / "winning-solution" / "investigations"
    out: dict[str, dict] = {}
    for f in sorted(inv_dir.glob("*.txt")):
        text = f.read_text()
        info = {}
        m = re.search(r"category:\s*(\S+)", text)
        if m:
            info["category"] = m.group(1).strip()
        # operator-to-operation mapping line(s) in cryptarithm_deduce investigations:
        # the LAST one applies to the query
        op_matches = re.findall(r"=\s*(add|abs_diff|mul|concat|rev_concat)", text)
        if op_matches:
            info["op_name"] = op_matches[-1]
        # bit_manipulation: "rule: ..." or "inferred rule:"
        m = re.search(r"rule:\s*([^\n]+)", text)
        if m:
            info["bm_rule"] = m.group(1).strip()
        out[f.stem] = info
    return out


def load_problems_metadata() -> dict[str, dict]:
    """Index of all 9500 problems from problems.jsonl (id -> {category, answer, ...})."""
    out: dict[str, dict] = {}
    path = PROJECT_ROOT / "resource" / "winning-solution" / "problems.jsonl"
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                out[rec["id"]] = rec
            except Exception:
                continue
    return out


def load_problem_prompt(pid: str) -> str | None:
    """Each problem has its own jsonl under problems/."""
    fp = PROJECT_ROOT / "resource" / "winning-solution" / "problems" / f"{pid}.jsonl"
    if not fp.exists():
        return None
    with open(fp) as f:
        rec = json.loads(f.readline())
    return rec.get("prompt")


def main() -> None:
    print("Loading Huikang investigations + problem metadata...")
    inv = load_huikang_investigations()
    meta = load_problems_metadata()
    print(f"  investigations: {len(inv)}")
    print(f"  problems:       {len(meta)}")

    # Only consider cryptarithm_deduce / cryptarithm_guess with a known op_name
    selected = []
    for pid, info in inv.items():
        if info.get("category") not in ("cryptarithm_deduce", "cryptarithm_guess"):
            continue
        if "op_name" not in info:
            continue
        prompt = load_problem_prompt(pid)
        if prompt is None:
            continue
        selected.append((pid, info, prompt))

    print(f"\n=== Selected: {len(selected)} solved cryptarithm puzzles with known op ===")

    # Group by op_name and compute signature stats
    by_op: dict[str, list[dict]] = defaultdict(list)
    rows = []
    for pid, info, prompt in selected:
        p = parse_prompt(prompt, answer=meta[pid].get("answer"))
        if p is None:
            continue
        m = measure(p)
        rec = {
            "pid": pid,
            "category": info["category"],
            "op_name": info["op_name"],
            "alphabet_size": m.alphabet_size,
            "rhs_in_lhs_rate": round(m.rhs_in_lhs_rate, 2),
            "rhs_len_var": round(m.rhs_len_var, 2),
            "rhs_lens": str(m.rhs_lens),
            "is_pure_concat": m.is_pure_concat,
            "is_pure_revconcat": m.is_pure_revconcat,
            "hyp_digit_cipher_arith": m.hyp_digit_cipher_arith,
            "hyp_concat_family": m.hyp_concat_family,
            "hyp_string_dsl": m.hyp_string_dsl,
            "op_uniform": m.op_uniform,
        }
        by_op[info["op_name"]].append(rec)
        rows.append(rec)

    print(f"\n=== Aggregate by op_name ===")
    print(f"{'op_name':10s} {'n':>4s} {'avg_alpha':>10s} {'avg_in_lhs':>11s} {'concat%':>9s} {'cipher%':>9s} {'string%':>9s}")
    for op in sorted(by_op):
        recs = by_op[op]
        n = len(recs)
        avg_alpha = sum(r["alphabet_size"] for r in recs) / n
        avg_rate = sum(r["rhs_in_lhs_rate"] for r in recs) / n
        concat_pct = sum(1 for r in recs if r["hyp_concat_family"]) / n * 100
        cipher_pct = sum(1 for r in recs if r["hyp_digit_cipher_arith"]) / n * 100
        string_pct = sum(1 for r in recs if r["hyp_string_dsl"]) / n * 100
        print(f"{op:10s} {n:4d} {avg_alpha:10.2f} {avg_rate:11.2f} {concat_pct:8.0f}% {cipher_pct:8.0f}% {string_pct:8.0f}%")

    print(f"\n=== Signature-rule alignment check ===")
    # For each op, what fraction of puzzles have the EXPECTED hypothesis vote?
    expected = {
        "concat": "hyp_concat_family",
        "rev_concat": "hyp_concat_family",
        "add": "hyp_digit_cipher_arith",
        "abs_diff": "hyp_digit_cipher_arith",
        "mul": "hyp_digit_cipher_arith",
    }
    print(f"{'op_name':10s} {'expected_hyp':25s} {'recall':>8s}")
    for op, hyp_key in expected.items():
        recs = by_op.get(op, [])
        if not recs:
            print(f"{op:10s} {hyp_key:25s} {'n/a':>8s}")
            continue
        hits = sum(1 for r in recs if r[hyp_key])
        print(f"{op:10s} {hyp_key:25s} {hits/len(recs)*100:7.1f}%")

    # Sample raw rows for inspection
    print(f"\n=== Sample rows (first 3 per op) ===")
    for op in sorted(by_op):
        print(f"\n-- op={op} --")
        for r in by_op[op][:3]:
            print(f"  {r['pid']}: alpha={r['alphabet_size']} preserve={r['rhs_in_lhs_rate']} "
                  f"rhs_lens={r['rhs_lens']} concat={r['is_pure_concat']}/{r['is_pure_revconcat']} "
                  f"cat={r['category']}")


if __name__ == "__main__":
    main()
