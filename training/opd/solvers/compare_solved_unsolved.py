"""Compare signature distributions: Huikang-solved vs Huikang-unsolved.

If the unsolved puzzles have DIFFERENT signatures, it means the signatures
identify the gap — they tell us "this puzzle needs a different solver family".
That's actionable: we know what to build next.

If the signatures are IDENTICAL between solved and unsolved, it means
Huikang's solver coverage gap is not about rule-family identification — it's
about something else (e.g., search complexity, ambiguity, multi-step rules
within the same family).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))

from category import detect_category  # noqa: E402
from measurements import parse_prompt, measure, Measurements  # noqa: E402


def load_problems() -> list[dict]:
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


def signature_bucket(m: Measurements) -> str:
    """Coarse signature class for grouping."""
    bits = []
    if m.is_pure_concat:
        bits.append("CONCAT")
    if m.is_pure_revconcat:
        bits.append("REVCONCAT")
    if m.position_lock:
        bits.append("POS-LOCK")
    if m.hyp_digit_cipher_arith:
        bits.append("DC-ARITH")
    if m.hyp_string_dsl:
        bits.append("STR-DSL")
    # Operator char family
    op_arith = m.op_chars_set and all(c in "+-*/" for c in m.op_chars_set)
    bits.append("op:arith" if op_arith else "op:sym" if m.op_chars_set else "op:none")
    # Alphabet bucket
    if m.alphabet_size <= 8:
        bits.append("a≤8")
    elif m.alphabet_size <= 10:
        bits.append("a=9-10")
    elif m.alphabet_size <= 12:
        bits.append("a=11-12")
    else:
        bits.append("a>12")
    # Preservation bucket
    r = m.rhs_in_lhs_rate
    if r < 0.2:
        bits.append("pres:low")
    elif r < 0.5:
        bits.append("pres:mid")
    elif r < 0.85:
        bits.append("pres:high")
    else:
        bits.append("pres:full")
    return "|".join(bits)


def main() -> None:
    problems = load_problems()
    print(f"Loaded {len(problems)} problems")

    # Focus on cryptarithm_deduce + cryptarithm_guess
    targets = [p for p in problems if p.get("category") in ("cryptarithm_deduce", "cryptarithm_guess")]
    print(f"Cryptarithm problems: {len(targets)}")

    # Build measurements for each
    enriched = []
    for p in targets:
        prompt = load_prompt(p["id"])
        if prompt is None:
            continue
        parsed = parse_prompt(prompt, answer=p.get("answer"))
        if parsed is None:
            continue
        m = measure(parsed)
        enriched.append({
            "id": p["id"],
            "category": p["category"],
            "status": p.get("status", "?"),  # rule_found / hypothesis_formed / rule_unknown
            "measurements": m,
            "bucket": signature_bucket(m),
        })

    # Group by status
    by_status: dict[str, list[dict]] = defaultdict(list)
    for r in enriched:
        by_status[r["status"]].append(r)

    print(f"\n=== Status distribution (cryptarithm only) ===")
    for s, rows in sorted(by_status.items()):
        print(f"  {s:25s} {len(rows)}")

    # Most common buckets in each status
    print(f"\n=== Top 8 signature buckets per status ===")
    for s in ("rule_found", "hypothesis_formed", "rule_unknown"):
        rows = by_status.get(s, [])
        if not rows:
            continue
        bk = Counter(r["bucket"] for r in rows)
        print(f"\n-- status={s} (n={len(rows)}) --")
        for bucket, n in bk.most_common(8):
            print(f"  {n:4d} ({n/len(rows)*100:5.1f}%)  {bucket}")

    # Cross-tab: which signatures predict solvability?
    print(f"\n=== Which signature buckets are 'mostly solved' vs 'mostly unsolved'? ===")
    bk_status: dict[str, Counter] = defaultdict(Counter)
    for r in enriched:
        bk_status[r["bucket"]][r["status"]] += 1
    rows_summary = []
    for bucket, counts in bk_status.items():
        total = sum(counts.values())
        if total < 5:
            continue  # ignore rare buckets
        rows_summary.append({
            "bucket": bucket,
            "total": total,
            "rule_found": counts.get("rule_found", 0),
            "hypothesis_formed": counts.get("hypothesis_formed", 0),
            "rule_unknown": counts.get("rule_unknown", 0),
            "solve_rate": counts.get("rule_found", 0) / total,
        })
    rows_summary.sort(key=lambda r: (-r["total"], -r["solve_rate"]))
    print(f"{'bucket':70s} {'total':>6s} {'solved':>7s} {'hypo':>5s} {'unk':>5s} {'%solv':>6s}")
    for r in rows_summary[:20]:
        print(f"{r['bucket']:70s} {r['total']:6d} {r['rule_found']:7d} "
              f"{r['hypothesis_formed']:5d} {r['rule_unknown']:5d} {r['solve_rate']*100:5.1f}%")


if __name__ == "__main__":
    main()
