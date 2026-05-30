"""Build training CSV for opd_v2 = opd_v1 (anchors + flash-cot diff_set)
plus newly-solved cryptarithm puzzles with our solver-grounded CoTs.

Output schema matches training/dataset.py:HuikangCoTDataset:
    id, prompt, answer, generated_cot, type
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from category import detect_category  # type: ignore  # noqa: E402

PROBE_RESULTS = PROJECT_ROOT / "training" / "opd" / "solvers" / "runs" / "probe_v2_results.jsonl"
NEW_REASONING_DIR = PROJECT_ROOT / "training" / "opd" / "solvers" / "runs" / "new_reasoning"
OPD_V1_CSV = PROJECT_ROOT / "training" / "opd" / "runs" / "opd_v1" / "train.csv"
PROBLEMS_DIR = PROJECT_ROOT / "resource" / "winning-solution" / "problems"
OUT_CSV = PROJECT_ROOT / "training" / "opd" / "runs" / "opd_v3" / "train.csv"


def load_problem(pid: str):
    fp = PROBLEMS_DIR / f"{pid}.jsonl"
    with open(fp) as f:
        return json.loads(f.readline())


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    new_rows = []
    n_skipped = 0
    with open(PROBE_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if not r["solved"]:
                continue
            pid = r["id"]
            cot_file = NEW_REASONING_DIR / f"{pid}.txt"
            if not cot_file.exists():
                n_skipped += 1
                continue
            cot_text = cot_file.read_text().strip()
            data = load_problem(pid)
            new_rows.append({
                "id": pid,
                "prompt": data["prompt"],
                "answer": str(data["answer"]),
                "generated_cot": cot_text,
                "type": detect_category(data["prompt"]),
            })

    print(f"New cryptarithm rows: {len(new_rows)}  (skipped {n_skipped})")

    # Read opd_v1 rows
    csv.field_size_limit(sys.maxsize)
    base_rows = []
    with open(OPD_V1_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_rows.append(row)
    print(f"opd_v1 base rows: {len(base_rows)}")

    # Drop Flash-CoT cryptarithm rows from base — Flash hallucinates on these.
    # Keep all other-category Flash rows (bit_manip, cipher, etc. are reliable).
    base_filtered = [r for r in base_rows if r["type"] not in ("cryptarithm_deduce", "cryptarithm_guess")]
    print(f"Base rows after dropping Flash cryptarithm: {len(base_filtered)} (was {len(base_rows)})")

    # De-dup against new rows (new wins)
    new_ids = {r["id"] for r in new_rows}
    base_filtered = [r for r in base_filtered if r["id"] not in new_ids]

    # Oversample new verified rows 3x — Codex recommendation: more exposure for new behavior
    OVERSAMPLE = 3
    new_oversampled = new_rows * OVERSAMPLE
    print(f"New verified rows × {OVERSAMPLE} = {len(new_oversampled)}")

    combined = base_filtered + new_oversampled
    print(f"Final combined rows: {len(combined)}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot", "type"])
        writer.writeheader()
        writer.writerows(combined)

    print(f"Wrote {OUT_CSV}")

    # Category breakdown
    from collections import Counter
    cat_counts = Counter(r["type"] for r in combined)
    new_counts = Counter(r["type"] for r in new_rows)
    print("\nCategory breakdown (final | new):")
    for cat in sorted(cat_counts):
        print(f"  {cat:30s} {cat_counts[cat]:5d} | new {new_counts.get(cat, 0)}")


if __name__ == "__main__":
    main()
