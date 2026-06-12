"""
Build augmented training CSV for SFT round 2.

Strategy: take the 556 new CoT examples (cipher + numeric) and mix with a
balanced sample of existing training data to prevent catastrophic forgetting.
The new cipher-digit examples are 6× what was in the original training set,
so they'll dominate learning on that category without overwhelming others.
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
random.seed(42)

# ---------------------------------------------------------------------------
# Load existing training data
# ---------------------------------------------------------------------------
existing: list[dict] = []
with open(PROJECT / "training" / "train_with_cot.csv") as f:
    for row in csv.DictReader(f):
        existing.append(row)

by_type: dict[str, list[dict]] = defaultdict(list)
for row in existing:
    by_type[row["type"]].append(row)

print("Existing training data:")
for t, rows in sorted(by_type.items()):
    print(f"  {t:35s}: {len(rows)}")

# ---------------------------------------------------------------------------
# Load new CoT data
# ---------------------------------------------------------------------------
new_records: list[dict] = []

with open(PROJECT / "training" / "opd" / "runs" / "cot_data" / "cipher_digit_cot.jsonl") as f:
    for line in f:
        r = json.loads(line)
        new_records.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "answer": r["answer"],
            "generated_cot": r["cot_trace"],
            "type": "cryptarithm_deduce",
        })

with open(PROJECT / "training" / "opd" / "runs" / "cot_data" / "numeric_cot.jsonl") as f:
    for line in f:
        r = json.loads(line)
        # Distinguish deduce vs guess based on whether we had 2+ same-op examples
        new_records.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "answer": r["answer"],
            "generated_cot": r["cot_trace"],
            "type": "equation_numeric_deduce",
        })

print(f"\nNew CoT records: {len(new_records)}")

# ---------------------------------------------------------------------------
# Build combined dataset
# ---------------------------------------------------------------------------
# Keep ALL new records (556)
# Add balanced sample from existing categories to prevent forgetting
# Target: ~2x new records from existing = ~1100 existing + 556 new = ~1650 total

new_ids = {r["id"] for r in new_records}

combined: list[dict] = list(new_records)

# From each existing category, sample proportionally to keep balance
# Skip cryptarithm_deduce from existing (our new data supersedes it)
# Keep other categories at full or sampled
KEEP_FULL = {"cipher", "numeral", "gravity", "unit_conversion"}
KEEP_SAMPLE_N = {
    "bit_manipulation": 300,
    "equation_numeric_deduce": 100,  # existing + new will cover this
    "equation_numeric_guess": 100,
    "cryptarithm_guess": 100,
    "cryptarithm_deduce": 0,  # superseded by new data
}

for cat, rows in by_type.items():
    # Skip rows that are now in new_records (avoid duplicates)
    rows_excl = [r for r in rows if r["id"] not in new_ids]

    if cat in KEEP_FULL:
        combined.extend(rows_excl)
    elif cat in KEEP_SAMPLE_N:
        n = KEEP_SAMPLE_N[cat]
        if n == 0:
            pass  # skip
        else:
            sampled = random.sample(rows_excl, min(n, len(rows_excl)))
            combined.extend(sampled)

random.shuffle(combined)

print(f"\nCombined dataset: {len(combined)} total")
from collections import Counter
type_dist = Counter(r["type"] for r in combined)
for t, n in sorted(type_dist.items()):
    print(f"  {t:35s}: {n}")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
out_path = PROJECT / "training" / "train_cot_v3.csv"
fieldnames = ["id", "prompt", "answer", "generated_cot", "type"]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(combined)

print(f"\nWritten to: {out_path}")
print(f"Total rows: {len(combined)}")
