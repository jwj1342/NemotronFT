"""Build the OPD SFT CSV: diff_set (baseline-wrong ∩ Flash-COT-correct) + anchors.

Inputs:
  --scan_dir   directory with scan_node*.jsonl from scan_baseline.sbatch
  --flashcot   Flash-COT-Filtered root (each category in subfolders)
  --out_csv    output CSV path
  --anchors_per_cat  number of baseline-correct anchors per category (default 100)
  --seed       RNG seed for anchor sampling

Output CSV schema matches training/dataset.py:HuikangCoTDataset:
    id, prompt, answer, generated_cot, type
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from category import detect_category  # type: ignore  # noqa: E402


def load_flashcot(flashcot_dir: Path) -> dict[str, dict]:
    """Return {id: {prompt, answer, cot_answer, category}} (canonical category)."""
    out: dict[str, dict] = {}
    for jsonl in sorted(flashcot_dir.rglob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cat = detect_category(rec["prompt"])
                if cat == "unknown":
                    continue
                out[str(rec["id"])] = {
                    "prompt": rec["prompt"],
                    "answer": str(rec["answer"]),
                    "cot_answer": rec["cot_answer"],
                    "category": cat,
                }
    return out


def load_baseline_scan(scan_dir: Path) -> dict[str, float]:
    """Return {id: reward} merged across shards."""
    reward: dict[str, float] = {}
    for p in sorted(scan_dir.glob("scan_node*.jsonl")):
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                reward[str(rec["prompt_id"])] = float(rec["group"][0]["reward"])
    return reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", type=Path,
                    default=PROJECT_ROOT / "training/opd/runs/baseline_scan")
    ap.add_argument("--flashcot", type=Path,
                    default=PROJECT_ROOT / "data/Flash-COT-Filtered")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--anchors_per_cat", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    flashcot = load_flashcot(args.flashcot)
    rewards = load_baseline_scan(args.scan_dir)
    print(f"[prep] flashcot={len(flashcot)}  scan_rewards={len(rewards)}")

    # Partition by baseline outcome
    diff_set: list[dict] = []
    anchor_pool: dict[str, list[dict]] = defaultdict(list)
    n_missing_scan = 0
    for pid, fc in flashcot.items():
        r = rewards.get(pid)
        if r is None:
            n_missing_scan += 1
            continue
        row = {
            "id": pid,
            "prompt": fc["prompt"],
            "answer": fc["answer"],
            "generated_cot": fc["cot_answer"],
            "type": fc["category"],
        }
        if r < 0.5:
            diff_set.append(row)
        else:
            anchor_pool[fc["category"]].append(row)

    if n_missing_scan:
        print(f"[prep] WARN: {n_missing_scan} flashcot ids had no scan result")

    # Sample anchors
    anchors: list[dict] = []
    for cat, pool in anchor_pool.items():
        rng.shuffle(pool)
        anchors.extend(pool[: args.anchors_per_cat])

    # Combine
    rows = diff_set + anchors
    rng.shuffle(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot", "type"])
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    diff_by_cat = defaultdict(int)
    anc_by_cat = defaultdict(int)
    for r in diff_set:
        diff_by_cat[r["type"]] += 1
    for r in anchors:
        anc_by_cat[r["type"]] += 1

    print(f"\n[prep] wrote {len(rows)} rows to {args.out_csv}")
    print(f"  diff_set:  {len(diff_set)}")
    print(f"  anchors:   {len(anchors)}  (target {args.anchors_per_cat}/cat)")
    print("\n[prep] per-category breakdown:")
    cats = sorted(set(list(diff_by_cat) + list(anc_by_cat)))
    print(f"  {'category':32s} {'diff':>6s} {'anchor':>7s} {'total':>7s}")
    for c in cats:
        d, a = diff_by_cat[c], anc_by_cat[c]
        print(f"  {c:32s} {d:6d} {a:7d} {d+a:7d}")


if __name__ == "__main__":
    main()
