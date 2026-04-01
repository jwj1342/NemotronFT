import argparse
import csv
import json
import sys
from pathlib import Path


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def convert_dataset(
    input_path: Path,
    output_path: Path,
    model_name: str,
    success_only: bool = True,
) -> dict[str, int]:
    set_csv_field_size_limit()

    with input_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = set(reader.fieldnames or [])
        required = {"prompt", "teacher_cot", "answer", "status"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"input csv is missing required columns: {', '.join(missing)}")

        records = []
        stats = {
            "total_rows": 0,
            "kept_rows": 0,
            "skipped_failed_rows": 0,
            "skipped_empty_rows": 0,
        }

        for row in reader:
            stats["total_rows"] += 1

            status = (row.get("status") or "").strip().lower()
            instruction = row.get("prompt") or ""
            output = row.get("teacher_cot") or ""
            answer = row.get("answer") or ""

            if success_only and status != "success":
                stats["skipped_failed_rows"] += 1
                continue

            if not instruction.strip() or not output.strip() or not answer.strip():
                stats["skipped_empty_rows"] += 1
                continue

            records.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "answer": answer,
                    "model_name": model_name,
                }
            )
            stats["kept_rows"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        json.dump(records, dst, ensure_ascii=False, indent=2)
        dst.write("\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a distilled CSV file into a JSON training dataset."
    )
    parser.add_argument(
        "--input",
        default="distillation/data/sample1200-train-distilled.csv",
        help="Input distilled CSV path",
    )
    parser.add_argument(
        "--output",
        default="distillation/dataset/sample1200-train-distilled-step-3.5-flash-0326.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--model-name",
        default="step-3.5-flash-0326",
        help="model_name value written to each sample",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Keep rows even if status is not success",
    )
    args = parser.parse_args()

    stats = convert_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model_name,
        success_only=not args.include_failed,
    )

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model name: {args.model_name}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Kept rows: {stats['kept_rows']}")
    print(f"Skipped failed rows: {stats['skipped_failed_rows']}")
    print(f"Skipped empty rows: {stats['skipped_empty_rows']}")


if __name__ == "__main__":
    main()
