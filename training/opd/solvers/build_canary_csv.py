"""Build canary CSV: 50 cryptarithm rows + 50 anchor (non-cryptarithm) rows
from opd_v2 train.csv.  Goal: minimum dataset to verify "can the model move
at all" under aggressive hyperparameters.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
IN_CSV = PROJECT / "training" / "opd" / "runs" / "opd_v3" / "train.csv"  # uses richer CoTs
OUT_CSV = PROJECT / "training" / "opd" / "runs" / "canary" / "train.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

random.seed(7)
csv.field_size_limit(sys.maxsize)

crypto = []
other = []
with open(IN_CSV) as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["type"] in ("cryptarithm_deduce", "cryptarithm_guess"):
            crypto.append(r)
        else:
            other.append(r)

random.shuffle(crypto)
random.shuffle(other)

# 50 unique cryptarithm rows (de-dup by id to avoid sampling oversampled triples)
seen = set()
crypto_uniq = []
for r in crypto:
    if r["id"] in seen:
        continue
    seen.add(r["id"])
    crypto_uniq.append(r)
    if len(crypto_uniq) == 50:
        break

other_sample = other[:50]
combined = crypto_uniq + other_sample
random.shuffle(combined)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot", "type"])
    w.writeheader()
    w.writerows(combined)

print(f"Canary CSV: {OUT_CSV}")
print(f"  cryptarithm: {len(crypto_uniq)}, other: {len(other_sample)}, total: {len(combined)}")
print(f"  ids first/last: {combined[0]['id']} / {combined[-1]['id']}")
