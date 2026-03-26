import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def sample_by_label(
    input_path: Path,
    output_path: Path,
    samples_per_label: int = 200,
    seed: int = 42,
) -> None:
    grouped_rows = defaultdict(list)

    with input_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        if "label" not in fieldnames:
            raise ValueError("input csv must contain a 'label' column")

        for row in reader:
            grouped_rows[row["label"]].append(row)

    rng = random.Random(seed)
    sampled_rows = []

    for label, rows in grouped_rows.items():
        if len(rows) < samples_per_label:
            raise ValueError(
                f"label '{label}' only has {len(rows)} rows, "
                f"cannot sample {samples_per_label}"
            )
        sampled_rows.extend(rng.sample(rows, samples_per_label))

    rng.shuffle(sampled_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the same number of rows from each label.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path")
    parser.add_argument(
        "--output",
        default="data/sample1200-train.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=200,
        help="Number of rows to sample for each label",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sample_by_label(
        input_path=Path(args.input),
        output_path=Path(args.output),
        samples_per_label=args.samples_per_label,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
