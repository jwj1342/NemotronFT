"""
Generate cipher-digit CoT training data for SFT.

Usage:
    python gen_cipher_cot.py [output_jsonl]

Reads data/by_category/equation_symbolic.csv, runs the answer-guided CoT
builder on each row, and writes solved problems to a JSONL file with fields:
    id, prompt, answer, cot_trace, combo

Also generates equation_numeric CoT using symbol_digit_solver.
"""
import csv
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "training" / "opd" / "solvers"))

from cipher_cot_builder import build_cot
from symbol_digit_solver import parse_problem_prompt, solve_symbol_digit, generate_cot_trace


def generate_cipher_digit_data(output_path: Path):
    csv_path = PROJECT / "data" / "by_category" / "equation_symbolic.csv"
    total = 0; solved = 0; t0 = time.time()

    with open(csv_path) as f_in, open(output_path, "w") as f_out:
        for row in csv.DictReader(f_in):
            total += 1
            res = build_cot(row["prompt"], row["answer"])
            if not res.solved:
                continue
            solved += 1
            record = {
                "id": row["id"],
                "category": "equation_symbolic",
                "prompt": row["prompt"],
                "answer": row["answer"],
                "cot_trace": res.cot_trace,
                "combo": str(res.combo),
            }
            f_out.write(json.dumps(record) + "\n")
            if solved % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Solved {solved}/{total} in {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Cipher-digit: {solved}/{total} solved ({100*solved/total:.1f}%) in {elapsed:.0f}s")
    return solved


def generate_numeric_data(output_path: Path):
    csv_path = PROJECT / "data" / "by_category" / "equation_numeric.csv"
    total = 0; solved = 0; t0 = time.time()

    with open(csv_path) as f_in, open(output_path, "w") as f_out:
        for row in csv.DictReader(f_in):
            total += 1
            examples, question = parse_problem_prompt(row["prompt"])
            if examples is None:
                continue
            res = solve_symbol_digit(examples, question)
            if not res.solved or res.answer != row["answer"]:
                continue
            solved += 1
            cot = generate_cot_trace(examples, question, res)
            record = {
                "id": row["id"],
                "category": "equation_numeric",
                "prompt": row["prompt"],
                "answer": row["answer"],
                "cot_trace": cot,
                "combo": str(res.combo),
            }
            f_out.write(json.dumps(record) + "\n")

    elapsed = time.time() - t0
    print(f"Numeric:      {solved}/{total} solved ({100*solved/total:.1f}%) in {elapsed:.0f}s")
    return solved


if __name__ == "__main__":
    out_dir = PROJECT / "training" / "opd" / "runs" / "cot_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    cipher_out = out_dir / "cipher_digit_cot.jsonl"
    numeric_out = out_dir / "numeric_cot.jsonl"

    print("Generating cipher-digit CoT data...")
    n1 = generate_cipher_digit_data(cipher_out)
    print(f"  Written to: {cipher_out}")

    print("Generating equation-numeric CoT data...")
    n2 = generate_numeric_data(numeric_out)
    print(f"  Written to: {numeric_out}")

    print(f"\nTotal CoT examples generated: {n1 + n2}")
