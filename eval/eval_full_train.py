"""
eval/eval_full_train.py — Per-category evaluation on the full 9500-problem
train.csv using vLLM + the 0.85 strong baseline LoRA.

Categorization matches Huikang's 9-class scheme used in the validation
notebook (cipher, bit_manipulation, gravity, numeral, unit_conversion,
equation_numeric_deduce/guess, cryptarithm_deduce/guess).

Generation config mirrors the Kaggle grader exactly:
  temperature=0.0  top_p=1.0  max_tokens=7680  max_model_len=8192

Usage:
  python eval_full_train.py --adapter_dir <path> [--max_samples N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "model")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import extract_final_answer, verify, BOXED_INSTRUCTION


# ── Huikang's 9-category detector (ported from adapter-validation-notebook) ──
def detect_category(prompt: str) -> str:
    if "secret bit manipulation rule transforms 8-bit binary numbers" in prompt:
        return "bit_manipulation"
    if "secret encryption rules are used on text" in prompt:
        return "cipher"
    if "secret set of transformation rules is applied to equations" in prompt:
        after_header = prompt.split("Below are a few examples:\n", 1)[1]
        examples_text, rest = after_header.split("\nNow, determine the result for: ", 1)
        question_text = rest.strip()
        if any(c.isdigit() for c in examples_text):
            q_match = re.fullmatch(r"(\d+)(\D)(\d+)", question_text)
            if q_match and re.search(
                r"\d" + re.escape(q_match.group(2)) + r"\d", examples_text
            ):
                return "equation_numeric_deduce"
            return "equation_numeric_guess"
        if len(question_text) == 5:
            q_op = question_text[2]
            for ex_line in examples_text.strip().splitlines():
                inp = ex_line.split(" = ")[0].strip()
                if len(inp) == 5 and inp[2] == q_op:
                    return "cryptarithm_deduce"
        return "cryptarithm_guess"
    if "gravitational constant has been secretly changed" in prompt:
        return "gravity"
    if "converted into a different numeral system" in prompt:
        return "numeral"
    if "secret unit conversion is applied to measurements" in prompt:
        return "unit_conversion"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True, type=Path)
    parser.add_argument("--train_csv", default=str(PROJECT_ROOT / "resource" / "winning-solution" / "train.csv"))
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap eval size for smoke testing.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="0-based shard index for parallel eval.")
    parser.add_argument("--shard_total", type=int, default=1,
                        help="Total shards. shard_index splits df[i::shard_total].")
    parser.add_argument("--max_tokens", type=int, default=7680)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_lora_rank", type=int, default=32)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--output_jsonl", default=None,
                        help="Path to write per-sample predictions JSONL.")
    parser.add_argument("--output_summary", default=None,
                        help="Path to write per-category summary JSON.")
    args = parser.parse_args()

    shard_tag = f"_shard{args.shard_index}of{args.shard_total}" if args.shard_total > 1 else ""
    if args.output_jsonl is None:
        adapter_tag = args.adapter_dir.name
        args.output_jsonl = str(PROJECT_ROOT / "output" / f"eval_full_{adapter_tag}{shard_tag}.jsonl")
    if args.output_summary is None:
        adapter_tag = args.adapter_dir.name
        args.output_summary = str(PROJECT_ROOT / "output" / f"eval_full_{adapter_tag}{shard_tag}_summary.json")

    print(f"[config]")
    print(f"  adapter_dir      = {args.adapter_dir}")
    print(f"  train_csv        = {args.train_csv}")
    print(f"  max_samples      = {args.max_samples or 'all'}")
    print(f"  max_tokens       = {args.max_tokens}")
    print(f"  max_model_len    = {args.max_model_len}")
    print(f"  tensor_parallel  = {args.tensor_parallel_size}")
    print(f"  output_jsonl     = {args.output_jsonl}")
    print(f"  output_summary   = {args.output_summary}")
    print()

    # Load train.csv
    df = pd.read_csv(args.train_csv, dtype={"answer": str})
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()
    if args.shard_total > 1:
        df = df.iloc[args.shard_index::args.shard_total].reset_index(drop=True)
        print(f"[data] shard {args.shard_index+1}/{args.shard_total}: {len(df)} samples")
    print(f"[data] {len(df)} samples loaded")
    df["category"] = df["prompt"].apply(detect_category)
    cat_counts = df["category"].value_counts()
    print(f"[data] category distribution:")
    for cat, n in cat_counts.items():
        print(f"  {cat:30s} {n}")
    print()

    # Set up vLLM
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[vllm] Loading model + tokenizer ...")
    t_load = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"[vllm] Loaded in {time.time() - t_load:.1f}s")
    print()

    # Build prompts via apply_chat_template + enable_thinking
    print(f"[prompts] Building {len(df)} prompts ...")
    prompts = []
    for problem_text in df["prompt"]:
        user_content = problem_text + BOXED_INSTRUCTION
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    lora = LoRARequest("adapter", 1, str(args.adapter_dir))

    print(f"[generate] Submitting {len(prompts)} prompts to vLLM ...")
    t_gen = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora)
    gen_elapsed = time.time() - t_gen
    print(f"[generate] Done in {gen_elapsed:.1f}s ({gen_elapsed / len(prompts):.2f}s/sample)")
    print()

    # Score
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    with open(args.output_jsonl, "w") as fout:
        for i, output in enumerate(outputs):
            row = df.iloc[i]
            text = output.outputs[0].text
            pred = extract_final_answer(text)
            correct = verify(str(row["answer"]), pred)

            per_cat[row["category"]]["total"] += 1
            if correct:
                per_cat[row["category"]]["correct"] += 1

            fout.write(json.dumps({
                "id": row["id"],
                "category": row["category"],
                "answer": row["answer"],
                "predicted": pred,
                "correct": correct,
                "num_chars": len(text),
                "num_tokens": len(output.outputs[0].token_ids),
            }) + "\n")

    # Summary
    grand_correct = sum(d["correct"] for d in per_cat.values())
    grand_total = sum(d["total"] for d in per_cat.values())
    summary = {
        "adapter_dir": str(args.adapter_dir),
        "train_csv": args.train_csv,
        "max_samples": args.max_samples,
        "total_correct": grand_correct,
        "total_samples": grand_total,
        "overall_accuracy": grand_correct / grand_total if grand_total else 0,
        "per_category": {},
        "config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
        },
    }
    for cat in sorted(per_cat):
        d = per_cat[cat]
        summary["per_category"][cat] = {
            "correct": d["correct"],
            "total": d["total"],
            "accuracy": d["correct"] / d["total"] if d["total"] else 0,
        }

    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print("=" * 80)
    print(f"{'Category':30s} {'Correct':>10s} {'Total':>10s} {'Acc':>10s}")
    print("-" * 80)
    for cat in sorted(per_cat):
        d = summary["per_category"][cat]
        print(f"{cat:30s} {d['correct']:>10d} {d['total']:>10d} {d['accuracy']*100:>9.1f}%")
    print("-" * 80)
    print(f"{'TOTAL':30s} {grand_correct:>10d} {grand_total:>10d} "
          f"{summary['overall_accuracy']*100:>9.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
