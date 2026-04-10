import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def sample_by_category(
    input_dir: Path,
    output_path: Path,
    samples_per_category: int = 200,
    seed: int = 42,
) -> None:
    """Read per-category CSVs from *input_dir*, sample uniformly, write a single CSV.

    Each CSV in *input_dir* must have columns ``id, prompt, answer``.
    The output CSV adds a ``category`` column derived from the filename stem.
    """
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for csv_file in sorted(input_dir.glob("*.csv")):
        category = csv_file.stem
        with csv_file.open("r", encoding="utf-8", newline="") as src:
            reader = csv.DictReader(src)
            fieldnames = reader.fieldnames or []
            for required in ("id", "prompt", "answer"):
                if required not in fieldnames:
                    raise ValueError(
                        f"{csv_file} is missing required column: {required}"
                    )
            for row in reader:
                row["category"] = category
                grouped_rows[category].append(row)

    if not grouped_rows:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    rng = random.Random(seed)
    sampled_rows: list[dict[str, str]] = []

    for category in sorted(grouped_rows):
        rows = grouped_rows[category]
        n = min(samples_per_category, len(rows))
        sampled_rows.extend(rng.sample(rows, n))
        print(f"  {category:22s}: {len(rows):5d} available, sampled {n}")

    rng.shuffle(sampled_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fields = ["id", "prompt", "answer", "category"]
    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"Wrote {len(sampled_rows)} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample training rows from per-category CSVs in data/by_category/."
    )
    parser.add_argument(
        "--input-dir",
        default="data/by_category",
        help="Directory containing per-category CSVs (default: data/by_category)",
    )
    parser.add_argument(
        "--output",
        default="data/sampled-train.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=200,
        help="Maximum rows to sample for each category",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sample_by_category(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
