"""Reward wrapper + group-relative advantage normalization.

reward = verify(extract_final_answer(text), gt) ∈ {0, 1}
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from extract import extract_final_answer, verify  # type: ignore  # noqa: E402


def score(text: str, gt: str) -> float:
    pred = extract_final_answer(text)
    return 1.0 if verify(str(gt), pred) else 0.0


def group_advantages(rewards: Iterable[float], eps: float = 1e-8) -> list[float]:
    r = [float(x) for x in rewards]
    if not r:
        return []
    mu = sum(r) / len(r)
    var = sum((x - mu) ** 2 for x in r) / len(r)
    sigma = var ** 0.5
    denom = sigma + eps
    return [(x - mu) / denom for x in r]


def main() -> None:
    """Sanity-check a rollout JSONL: re-score and recompute advantages, compare."""
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    args = ap.parse_args()

    n_total = 0
    n_mismatch = 0
    cat_stats: dict[str, list[float]] = {}
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            group = rec.get("group", [])
            recomp_r = [score(g["text"], rec["gt"]) for g in group]
            stored_r = [float(g["reward"]) for g in group]
            for a, b in zip(recomp_r, stored_r):
                n_total += 1
                if abs(a - b) > 1e-9:
                    n_mismatch += 1
            recomp_adv = group_advantages(recomp_r)
            stored_adv = [float(g["advantage"]) for g in group]
            cat = rec.get("category", "?")
            cat_stats.setdefault(cat, []).extend(recomp_r)
            tag = "OK " if all(abs(a - b) < 1e-9 for a, b in zip(recomp_r, stored_r)) else "BAD"
            print(f"[{tag}] {rec.get('prompt_id', '?'):>10s} {cat:25s} "
                  f"r={recomp_r}  adv=[{','.join(f'{x:+.2f}' for x in recomp_adv)}]")

    print()
    print(f"[reward sanity] {n_total} rewards checked; {n_mismatch} mismatches.")
    for cat, rs in sorted(cat_stats.items()):
        if rs:
            mu = sum(rs) / len(rs)
            print(f"  {cat:30s} n={len(rs)}  pass@1={mu:.3f}")


if __name__ == "__main__":
    main()
