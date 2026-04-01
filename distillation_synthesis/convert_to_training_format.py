"""
Convert distilled JSON datasets to JSONL training format compatible with cot_training/dataset.py.

Input  (data/distillation/<category>/<category>_distilled.json):
  [{"instruction": ..., "input": "", "output": <cot>, "answer": ..., "model_name": ...}, ...]

Output (data/distillation/<category>/level_1.jsonl):
  {"prompt": ..., "cot_answer": <cot>\n\n\\boxed{answer}, "answer": ..., "id": ...}  (one per line)

The output filename is always "level_1.jsonl" so that cot_training/dataset.py picks it up via
its `glob("level_*.jsonl")` scan without any code changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def make_id(text: str) -> str:
    """Stable 8-char hex ID derived from content."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def convert_file(src: Path, dst: Path) -> dict[str, int]:
    with src.open(encoding="utf-8") as f:
        records = json.load(f)

    stats = {"total": len(records), "written": 0, "skipped_empty": 0}

    with dst.open("w", encoding="utf-8") as out:
        for r in records:
            prompt = (r.get("instruction") or "").strip()
            cot = (r.get("output") or "").strip()
            answer = (r.get("answer") or "").strip()

            if not prompt or not cot or not answer:
                stats["skipped_empty"] += 1
                continue

            # Strip any accidental trailing \boxed already present (avoid duplicates)
            if cot.endswith(f"\\boxed{{{answer}}}"):
                cot_body = cot[: -len(f"\\boxed{{{answer}}}")].rstrip()
            else:
                cot_body = cot

            cot_answer = f"{cot_body}\n\n\\boxed{{{answer}}}"

            obj = {
                "id": make_id(prompt),
                "prompt": prompt,
                "cot_answer": cot_answer,
                "answer": answer,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert distilled JSON files to JSONL format for cot_training."
    )
    parser.add_argument(
        "--distillation-dir",
        default="data/distillation",
        help="Root directory containing per-category subdirectories (default: data/distillation)",
    )
    parser.add_argument(
        "--output-filename",
        default="distilled.jsonl",
        help="Output filename written into each category directory (default: distilled.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating files",
    )
    args = parser.parse_args()

    base = Path(args.distillation_dir)
    if not base.is_dir():
        raise SystemExit(f"Directory not found: {base}")

    total_written = 0
    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue

        # Find the *_distilled.json file in this category directory
        src_files = list(cat_dir.glob("*_distilled.json"))
        if not src_files:
            continue
        src = src_files[0]
        dst = cat_dir / args.output_filename

        if args.dry_run:
            print(f"[dry-run] {src} -> {dst}")
            continue

        stats = convert_file(src, dst)
        total_written += stats["written"]
        skipped = stats["skipped_empty"]
        skip_msg = f", skipped {skipped} empty" if skipped else ""
        print(f"  {cat_dir.name:22s}: {stats['written']:4d} records -> {dst}{skip_msg}")

    if not args.dry_run:
        print(f"\nDone. Total records written: {total_written}")
        print(f"Files are named '{args.output_filename}' and will be auto-discovered by cot_training/dataset.py (scans *.jsonl).")


if __name__ == "__main__":
    main()
