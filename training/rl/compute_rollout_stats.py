"""Recompute aggregate rollout stats from a (possibly merged multi-node) rollout
JSONL, for TensorBoard collapse-watch. Truncation is proxied by
len(response_ids) >= max_tokens (the merged JSONL doesn't carry finish_reason).

Usage: python compute_rollout_stats.py <rollout.jsonl> <stats_out.json> <max_tokens>
"""
from __future__ import annotations
import json, sys


def main() -> None:
    rollout, out, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
    recs = [json.loads(l) for l in open(rollout) if l.strip()]
    all_r, resp_lens, distinct = [], [], []
    n_trunc = n_boxed = n_mixed = n_all_right = n_all_wrong = 0
    for rec in recs:
        rs = [g["reward"] for g in rec["group"]]
        all_r += rs
        for g in rec["group"]:
            L = len(g["response_ids"]); resp_lens.append(L)
            if L >= max_tokens:
                n_trunc += 1
            if "\\boxed{" in g.get("text", ""):
                n_boxed += 1
        if all(x == rs[0] for x in rs):
            if rs[0] > 0:
                n_all_right += 1
            else:
                n_all_wrong += 1
        else:
            n_mixed += 1
        distinct.append(len(set(g.get("text", "") for g in rec["group"])) / max(1, len(rec["group"])))
    n = max(1, len(all_r))
    mean_r = sum(all_r) / n
    var_r = sum((x - mean_r) ** 2 for x in all_r) / n
    stats = {
        "reward_mean": mean_r, "reward_var": var_r,
        "frac_mixed_groups": n_mixed / max(1, len(recs)),
        "n_mixed": n_mixed, "n_all_right": n_all_right, "n_all_wrong": n_all_wrong,
        "n_prompts": len(recs),
        "resp_len_mean": sum(resp_lens) / max(1, len(resp_lens)),
        "frac_truncated": n_trunc / n, "boxed_rate": n_boxed / n,
        "distinct_ratio": sum(distinct) / max(1, len(distinct)),
    }
    with open(out, "w") as f:
        json.dump(stats, f)
    print(f"[stats] {len(recs)} prompts reward_mean={mean_r:.4f} reward_var={var_r:.4f} "
          f"frac_mixed={stats['frac_mixed_groups']:.3f} resp_len={stats['resp_len_mean']:.0f} "
          f"frac_trunc={stats['frac_truncated']:.3f} distinct={stats['distinct_ratio']:.3f} -> {out}")


if __name__ == "__main__":
    main()
