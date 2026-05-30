"""Cluster the cryptarithm puzzles our v4 solver did NOT solve, by structural
signature.  Goal: see what the dominant rule families are among the ~643
unsolved cryptarithm puzzles so we can scope the next solver (string-DSL etc).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))

from measurements import measure, parse_prompt  # type: ignore

PROBLEMS_FILE = PROJECT_ROOT / "resource" / "winning-solution" / "problems.jsonl"
PROBE_RESULTS = PROJECT_ROOT / "training" / "opd" / "solvers" / "runs" / "probe_v2_results.jsonl"
PROBLEMS_DIR = PROJECT_ROOT / "resource" / "winning-solution" / "problems"


def main():
    solved_ids = set()
    with open(PROBE_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if r["solved"]:
                solved_ids.add(r["id"])
    print(f"Solved by our v4 solver: {len(solved_ids)}")

    unsolved = []
    huikang_solved = []
    with open(PROBLEMS_FILE) as f:
        for line in f:
            p = json.loads(line)
            if p.get("category") not in ("cryptarithm_deduce", "cryptarithm_guess"):
                continue
            if p.get("status") == "rule_found":
                huikang_solved.append(p)
                continue
            if p["id"] in solved_ids:
                continue
            unsolved.append(p)
    print(f"Huikang solved (skipped):    {len(huikang_solved)}")
    print(f"Still unsolved:              {len(unsolved)}")

    # Cluster signatures
    buckets: dict[str, list[dict]] = defaultdict(list)
    bucket_stats: dict[str, Counter] = defaultdict(Counter)
    for p in unsolved:
        # problem.jsonl entries don't have full prompt — load from per-id file
        pid = p["id"]
        try:
            with open(PROBLEMS_DIR / f"{pid}.jsonl") as f:
                d = json.loads(f.readline())
        except Exception:
            continue
        prompt = d.get("prompt", "")
        parsed = parse_prompt(prompt)
        if not parsed:
            continue
        m = measure(parsed)

        # Bucketing by qualitative signature
        if m.is_pure_concat:
            key = "A_pure_concat"
        elif m.is_pure_revconcat:
            key = "B_pure_revconcat"
        elif m.position_lock:
            key = "C_position_lock_permutation"
        elif m.alphabet_size > 12:
            key = "D_large_alphabet_(string-dsl)"
        elif m.rhs_in_lhs_rate > 0.85:
            key = "E_high_preservation_(copy-like)"
        elif m.alphabet_size <= 12 and not m.rhs_len_uniform:
            key = "F_digit_cipher_arith_unsolved_(harder_op)"
        elif m.alphabet_size <= 12 and m.rhs_len_uniform:
            key = "G_digit_cipher_const_rhs_len"
        elif m.op_in_rhs_count >= max(1, m.n_examples // 2):
            key = "H_operator_encoded"
        else:
            key = "Z_other"

        buckets[key].append({"id": pid, "summary": m.summary(), "category": p["category"]})
        bucket_stats[key]["alphabet_avg"] += m.alphabet_size
        bucket_stats[key]["rhs_len_var_avg"] += m.rhs_len_var
        bucket_stats[key]["preserve_avg"] += m.rhs_in_lhs_rate

    # Print summary
    print("\n=== UNSOLVED CRYPTARITHM CLUSTERS ===")
    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
        ex = buckets[key]
        n = len(ex)
        cat_counts = Counter(e["category"] for e in ex)
        print(f"\n{key}: {n} puzzles  (cat: {dict(cat_counts)})")
        print(f"  avg alphabet={bucket_stats[key]['alphabet_avg']/n:.1f}, "
              f"avg rhs_var={bucket_stats[key]['rhs_len_var_avg']/n:.2f}, "
              f"avg preserve={bucket_stats[key]['preserve_avg']/n:.2f}")
        print("  examples:")
        for e in ex[:3]:
            print(f"    {e['id']}: {e['summary']}")

    # Dump 3 sample puzzles per bucket to disk for inspection
    out_path = PROJECT_ROOT / "training" / "opd" / "solvers" / "runs" / "unsolved_clusters.txt"
    with open(out_path, "w") as f:
        for key in sorted(buckets, key=lambda k: -len(buckets[k])):
            f.write(f"\n========== {key} ({len(buckets[key])} puzzles) ==========\n")
            for e in buckets[key][:5]:
                pid = e["id"]
                with open(PROBLEMS_DIR / f"{pid}.jsonl") as pf:
                    d = json.loads(pf.readline())
                f.write(f"\n--- {pid} ({e['category']}) ---\n")
                f.write(d["prompt"][:600] + "\n")
                f.write(f"answer: {d.get('answer')}\n")
                f.write(f"sig: {e['summary']}\n")
    print(f"\nSample puzzles per bucket: {out_path}")


if __name__ == "__main__":
    main()
