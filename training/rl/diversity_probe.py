"""Analyze a bit rollout JSONL for GRPO viability (the gatekeeper).

Joins the rollout output back to the pool's greedy_correct flag and reports:
  - WRONG prompts (where GRPO must learn): fraction of groups with >=1 correct sample
    (= nonzero group advantage = GRPO gradient exists), mean reward (~pass@1), distinct
    sequences/group.
  - CORRECT prompts (regression risk): mean reward at T>0 (does sampling break them?).
  - Overall: fraction of NON-DEGENERATE groups (not all-0 / not all-1) — Codex's >30%
    retention gate for a usable training signal.

Verdict: GRPO can START on bit iff a meaningful fraction of WRONG groups are
recoverable (>=1 correct sample) AND groups aren't entropy-collapsed (distinct >~3).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load_pool_flags(pool_path):
    flags = {}
    for line in open(pool_path):
        if line.strip():
            r = json.loads(line)
            flags[r["id"]] = r.get("greedy_correct")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--pool", required=True)
    args = ap.parse_args()
    flags = load_pool_flags(args.pool)

    groups = []
    for line in open(args.rollout):
        if not line.strip():
            continue
        rec = json.loads(line)
        g = rec.get("group", [])
        if not g:
            continue
        rewards = [x["reward"] for x in g]
        n = len(rewards)
        n_correct = sum(1 for x in rewards if x > 0.5)
        heads = {x["text"][:120] for x in g}
        groups.append({
            "id": rec["prompt_id"], "n": n, "n_correct": n_correct,
            "mean_r": n_correct / n, "distinct": len(heads),
            "nondegenerate": 0 < n_correct < n,
            "greedy_correct": flags.get(rec["prompt_id"]),
        })

    def summ(rows, label):
        if not rows:
            print(f"  {label}: (none)"); return
        ng = len(rows)
        recov = sum(1 for r in rows if r["n_correct"] > 0)
        nondeg = sum(1 for r in rows if r["nondegenerate"])
        print(f"  {label}: {ng} groups | mean_reward={sum(r['mean_r'] for r in rows)/ng:.3f} "
              f"| groups w/ >=1 correct={recov}/{ng} ({recov/ng*100:.0f}%) "
              f"| non-degenerate(gradient)={nondeg}/{ng} ({nondeg/ng*100:.0f}%) "
              f"| mean distinct seqs/group={sum(r['distinct'] for r in rows)/ng:.1f}")

    wrong = [g for g in groups if g["greedy_correct"] is False]
    correct = [g for g in groups if g["greedy_correct"] is True]
    print(f"\n=== BIT DIVERSITY PROBE ({len(groups)} groups, G={groups[0]['n'] if groups else '?'}) ===")
    summ(groups, "ALL")
    summ(wrong, "greedy-WRONG (GRPO learns here)")
    summ(correct, "greedy-CORRECT (regression risk)")

    nondeg_all = sum(1 for g in groups if g["nondegenerate"]) / max(1, len(groups))
    recov_wrong = (sum(1 for g in wrong if g["n_correct"] > 0) / len(wrong)) if wrong else 0
    distinct_all = sum(g["distinct"] for g in groups) / max(1, len(groups))
    print("\n=== GATE ===")
    print(f"  non-degenerate groups overall: {nondeg_all*100:.0f}%  (Codex usable-signal gate >30%)")
    print(f"  recoverable WRONG groups (>=1 correct): {recov_wrong*100:.0f}%")
    print(f"  mean distinct seqs/group: {distinct_all:.1f}  (entropy-collapse if <~3 at G={groups[0]['n'] if groups else '?'})")
    verdict = "VIABLE — GRPO has gradient on bit" if (nondeg_all > 0.30 and distinct_all > 3) else \
              "DEAD — entropy-collapsed / no usable signal"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
