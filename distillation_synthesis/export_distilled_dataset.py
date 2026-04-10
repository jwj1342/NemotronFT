"""
Export a distilled CSV into per-category ``distilled.jsonl`` files that
``distillation_training/dataset.py`` can load directly.

Input  (distillation.py output CSV):
    Columns: id, prompt, answer, category, teacher_cot, status, ...

Output (one file per category):
    data/distillation/<category>/distilled.jsonl
    Each line:  {"id": ..., "prompt": ..., "cot_answer": ..., "answer": ...}

``cot_answer`` is the teacher CoT with ``\\boxed{answer}`` appended, matching
the format used by cot_training / distillation_training dataset loaders.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def export_dataset(
    input_path: Path,
    output_dir: Path,
    success_only: bool = True,
) -> dict[str, dict[str, int]]:
    """Read distilled CSV → write per-category ``distilled.jsonl``.

    Returns a dict of ``{category: {total, written, skipped_failed, skipped_empty}}``.
    """
    set_csv_field_size_limit()

    with input_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = set(reader.fieldnames or [])
        required = {"id", "prompt", "teacher_cot", "answer", "category", "status"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(missing)}"
            )

        # Group rows by category
        by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            by_category[row["category"]].append(row)

    all_stats: dict[str, dict[str, int]] = {}

    for category in sorted(by_category):
        rows = by_category[category]
        stats = {"total": len(rows), "written": 0, "skipped_failed": 0, "skipped_empty": 0}

        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = cat_dir / "distilled.jsonl"

        with jsonl_path.open("w", encoding="utf-8") as dst:
            for row in rows:
                status = (row.get("status") or "").strip().lower()
                prompt = (row.get("prompt") or "").strip()
                cot = (row.get("teacher_cot") or "").strip()
                answer = (row.get("answer") or "").strip()
                row_id = (row.get("id") or "").strip()

                if success_only and status != "success":
                    stats["skipped_failed"] += 1
                    continue

                if not prompt or not cot or not answer:
                    stats["skipped_empty"] += 1
                    continue

                # Strip any trailing \boxed{} already present
                boxed_suffix = f"\\boxed{{{answer}}}"
                if cot.endswith(boxed_suffix):
                    cot = cot[: -len(boxed_suffix)].rstrip()

                cot_answer = f"{cot}\n\n\\boxed{{{answer}}}"

                obj = {
                    "id": row_id,
                    "prompt": prompt,
                    "cot_answer": cot_answer,
                    "answer": answer,
                }
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["written"] += 1

        all_stats[category] = stats

    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export distilled CSV to per-category JSONL for training."
    )
    parser.add_argument(
        "--input",
        default="data/sampled-train-distilled.csv",
        help="Input distilled CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/distillation",
        help="Output root directory (per-category subdirs will be created)",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Keep rows even if status is not 'success'",
    )
    args = parser.parse_args()

    all_stats = export_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        success_only=not args.include_failed,
    )

    print(f"Input:  {args.input}")
    print(f"Output: {args.output_dir}/")
    print()
    total_written = 0
    for category, stats in sorted(all_stats.items()):
        skip_parts = []
        if stats["skipped_failed"]:
            skip_parts.append(f"{stats['skipped_failed']} failed")
        if stats["skipped_empty"]:
            skip_parts.append(f"{stats['skipped_empty']} empty")
        skip_msg = f" (skipped: {', '.join(skip_parts)})" if skip_parts else ""
        print(f"  {category:22s}: {stats['written']:4d} written / {stats['total']:4d} total{skip_msg}")
        total_written += stats["written"]
    print(f"\nTotal written: {total_written}")


if __name__ == "__main__":
    main()
