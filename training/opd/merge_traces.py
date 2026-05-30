"""Merge per-source OPD trace JSONLs into a single CSV ready for SFT.

Output CSV schema matches training/dataset.py:HuikangCoTDataset:
    id, prompt, answer, generated_cot, type

Each source can be specified with a multiplier ``W@path.jsonl`` (default 1.0)
so weak categories' RL adapter traces can be oversampled relative to anchors.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


def extract_cot(text: str) -> str:
    """Pull the reasoning portion out of a generated trace.

    Preferred: content inside the first <think>...</think>.
    Fallback: full text with the final \\boxed{...} stripped.
    """
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    return BOXED_RE.sub("", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help='List of "WEIGHT@path.jsonl" (WEIGHT defaults to 1.0).')
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only_correct", action="store_true",
                    help='Drop traces with correct=false.')
    ap.add_argument("--max_per_cat", type=int, default=0,
                    help='Cap rows per category (0 = no cap). Applied AFTER weighting.')
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows: list[dict] = []
    for src in args.sources:
        if "@" in src:
            w_str, path_s = src.split("@", 1)
            weight = float(w_str)
        else:
            weight, path_s = 1.0, src
        path = Path(path_s)
        if not path.exists():
            print(f"[merge] skipping missing source: {path}")
            continue
        n_in, n_kept = 0, 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_in += 1
                if args.only_correct and not rec.get("correct", True):
                    continue
                cot = extract_cot(rec["text"])
                if not cot:
                    continue
                base_row = {
                    "id": rec["id"],
                    "prompt": rec["prompt"],
                    "answer": rec["answer"],
                    "generated_cot": cot,
                    "type": rec["category"],
                }
                # Replicate by weight: integer floor + fractional Bernoulli
                n_copies = int(weight)
                frac = weight - n_copies
                if frac > 0 and rng.random() < frac:
                    n_copies += 1
                for _ in range(n_copies):
                    rows.append(base_row)
                    n_kept += 1
        print(f"[merge] {path.name}  weight={weight:.2f}  in={n_in}  kept={n_kept}")

    # Per-category cap
    if args.max_per_cat > 0:
        rng.shuffle(rows)
        capped: list[dict] = []
        cnt: dict[str, int] = {}
        for r in rows:
            n = cnt.get(r["type"], 0)
            if n >= args.max_per_cat:
                continue
            capped.append(r)
            cnt[r["type"]] = n + 1
        rows = capped

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot", "type"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["type"]] = by_cat.get(r["type"], 0) + 1
    print(f"[merge] wrote {len(rows)} rows to {args.out}")
    for cat in sorted(by_cat):
        print(f"  {cat:30s} {by_cat[cat]}")


if __name__ == "__main__":
    main()
