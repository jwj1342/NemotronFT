"""Build per-task prompt pools from train.csv. Supports per-node sharding.

Uses stdlib csv (no pandas dependency) so it can run in either venv/venv-vllm.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from category import detect_category  # type: ignore  # noqa: E402
from utils import write_jsonl  # noqa: E402


def build_pool(train_csv: Path, categories: set[str], cap_per_cat: int | None) -> list[dict]:
    cap = cap_per_cat if (cap_per_cat is not None and cap_per_cat > 0) else None
    per_cat_count: dict[str, int] = {}
    out: list[dict] = []
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = detect_category(row["prompt"])
            if category not in categories:
                continue
            n_so_far = per_cat_count.get(category, 0)
            if cap is not None and n_so_far >= cap:
                continue
            per_cat_count[category] = n_so_far + 1
            out.append({
                "id": str(row["id"]),
                "prompt": row["prompt"],
                "gt": str(row["answer"]),
                "category": category,
            })
    return out


def shard_pool(pool: list[dict], shard_index: int, shard_total: int) -> list[dict]:
    if shard_total <= 1:
        return pool
    return pool[shard_index::shard_total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv",
                    default=str(PROJECT_ROOT / "resource" / "winning-solution" / "train.csv"))
    ap.add_argument("--categories", required=True,
                    help="comma-separated category names")
    ap.add_argument("--cap_per_cat", type=int, default=None)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--shard_total", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    pool = build_pool(Path(args.train_csv), cats, args.cap_per_cat)
    pool = shard_pool(pool, args.shard_index, args.shard_total)
    n = write_jsonl(args.out, pool)
    by_cat: dict[str, int] = {}
    for p in pool:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    print(f"[pool] wrote {n} prompts to {args.out}")
    for cat in sorted(by_cat):
        print(f"  {cat:30s} {by_cat[cat]}")


if __name__ == "__main__":
    main()
