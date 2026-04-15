"""
distillation_training/eval_vllm.py — vLLM-based evaluation for LoRA checkpoints.

Uses vLLM for ~30x faster inference compared to HuggingFace generate().
Fully aligned with Kaggle evaluation: greedy decoding, max_tokens=7680.

Usage:
  python distillation_training/eval_vllm.py --checkpoint_dir <path>
  python distillation_training/eval_vllm.py --checkpoint_dir <path1> --checkpoint_dir <path2>
"""

import argparse
import csv
import json
import random
import sys
import time
import torch
from collections import defaultdict
from pathlib import Path

# Reuse official metric functions from baseline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from eval import extract_final_answer, verify, BOXED_INSTRUCTION

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = str(PROJECT_ROOT / "model")
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "by_category"
DISTILL_DATA_DIR = PROJECT_ROOT / "data" / "distillation"
COT_DATA_DIR     = PROJECT_ROOT / "data" / "cot"

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", action="append", default=[],
                    help="Path(s) to LoRA checkpoint(s). Can specify multiple.")
parser.add_argument("--scan_dir", type=str, default=None,
                    help="Auto-discover all checkpoint-* dirs under this path.")
parser.add_argument("--max_tokens", type=int, default=7680,
                    help="Max new tokens (Kaggle uses 7680).")
parser.add_argument("--eval_samples_per_class", type=int, default=5)
parser.add_argument("--train_samples_per_class", type=int, default=500,
                    help="Training samples to skip (same split as cot_training).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory for result JSONL files.")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
parser.add_argument("--tensorboard_dir", type=str, default=None,
                    help="TensorBoard log dir to write eval metrics into. "
                         "If --scan_dir is set and {scan_dir}/.tb_run_dir exists, "
                         "auto-detected from training run.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = str(PROJECT_ROOT / "output")

# Auto-discover checkpoints from scan_dir
if args.scan_dir:
    import re
    scan_path = Path(args.scan_dir)
    discovered = sorted(
        [d for d in scan_path.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")
         and (d / "adapter_config.json").exists()],
        key=lambda d: int(re.search(r'checkpoint-(\d+)', d.name).group(1)),
    )
    args.checkpoint_dir.extend(str(d) for d in discovered)
    print(f"[scan] Discovered {len(discovered)} checkpoints in {args.scan_dir}")

    # Auto-detect TensorBoard dir from training run
    if args.tensorboard_dir is None:
        tb_marker = scan_path / ".tb_run_dir"
        if tb_marker.exists():
            args.tensorboard_dir = tb_marker.read_text().strip()
            print(f"[scan] Auto-detected TensorBoard dir: {args.tensorboard_dir}")

if not args.checkpoint_dir:
    print("ERROR: No checkpoints specified. Use --checkpoint_dir or --scan_dir.")
    sys.exit(1)

# ── Collect training-side prompts (for leak filtering) ───────────────────────
def _load_training_prompts():
    """Load all prompts used in training (distillation + CoT) for dedup."""
    seen = set()
    # Distillation prompts
    for task_dir in sorted(DISTILL_DATA_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        jsonl_file = task_dir / "distilled.jsonl"
        if not jsonl_file.exists():
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                seen.add(json.loads(line)["prompt"])
    # CoT synthesis prompts
    for task_dir in sorted(COT_DATA_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        for level_file in sorted(task_dir.glob("*.jsonl")):
            with open(level_file, "r", encoding="utf-8") as f:
                for line in f:
                    seen.add(json.loads(line)["prompt"])
    return seen


# ── Load eval data ───────────────────────────────────────────────────────────
def load_eval_pool(data_dir, train_samples_per_class, eval_samples_per_class, seed):
    """Load eval samples using distillation-priority split (matches MixedDataset).

    Distillation prompts are counted as part of the training split, so eval
    is drawn only from non-distillation leftovers, then further filtered
    against CoT training prompts.
    """
    training_prompts = _load_training_prompts()

    # Load distillation prompts for priority split
    distill_prompts = set()
    for task_dir in sorted(DISTILL_DATA_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        jsonl_file = task_dir / "distilled.jsonl"
        if not jsonl_file.exists():
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                distill_prompts.add(json.loads(line)["prompt"])

    rng = random.Random(seed)
    eval_data = []
    n_filtered = 0
    for csv_file in sorted(Path(data_dir).glob("*.csv")):
        category = csv_file.stem
        rows = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["prompt"], row["answer"]))
        rng.shuffle(rows)

        # Same priority logic as MixedDataset: distill prompts fill train first
        has_distill = [(p, a) for p, a in rows if p in distill_prompts]
        no_distill = [(p, a) for p, a in rows if p not in distill_prompts]
        n_distill = min(len(has_distill), train_samples_per_class)
        n_regular_fill = train_samples_per_class - n_distill
        remaining = no_distill[n_regular_fill:]

        # Filter out any remaining prompts that appear in CoT training data
        chosen = []
        for p, a in remaining:
            if len(chosen) >= eval_samples_per_class:
                break
            if p in training_prompts:
                n_filtered += 1
                continue
            chosen.append((p, a))
        eval_data.extend([(p, a, category) for p, a in chosen])
    if n_filtered > 0:
        print(f"  [dedup] Filtered {n_filtered} eval candidates that overlap with training data")
    return eval_data

print(f"[config] checkpoints       = {args.checkpoint_dir}")
print(f"[config] max_tokens        = {args.max_tokens}")
print(f"[config] eval_samples/cls  = {args.eval_samples_per_class}")
print(f"[config] train_skip/cls    = {args.train_samples_per_class}")
print(f"[config] output_dir        = {args.output_dir}")
print(f"[config] tensorboard_dir   = {args.tensorboard_dir}")

# ── TensorBoard writer (optional) ───────────────────────────────────────────
tb_writer = None
if args.tensorboard_dir:
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        print(f"[tensorboard] Writing eval metrics to {args.tensorboard_dir}")
    except ImportError:
        print("[tensorboard] tensorboard not installed, skipping")

print("\n[1/4] Loading eval data ...")
eval_data = load_eval_pool(
    EVAL_DATA_DIR, args.train_samples_per_class,
    args.eval_samples_per_class, args.seed,
)
print(f"  {len(eval_data)} eval samples loaded")

# ── Build prompts (Kaggle-aligned chat template) ────────────────────────────
print("\n[2/4] Building prompts ...")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

prompts = []
for prompt_text, answer, category in eval_data:
    user_content = prompt_text + BOXED_INSTRUCTION
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompts.append(chat_input)

print(f"  {len(prompts)} prompts built")

# ── Load vLLM engine ────────────────────────────────────────────────────────
print("\n[3/4] Loading vLLM engine ...")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

t_load = time.time()
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=args.gpu_memory_utilization,
    dtype="auto",
    max_model_len=8192,
    trust_remote_code=True,
    enable_lora=True,
    max_lora_rank=32,
    enable_prefix_caching=True,
)
t_load = time.time() - t_load
print(f"  Engine loaded in {t_load:.1f}s")

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=args.max_tokens,
)

# ── Evaluate each checkpoint ────────────────────────────────────────────────
print(f"\n[4/4] Evaluating {len(args.checkpoint_dir)} checkpoint(s) ...")

all_results = {}

for ckpt_idx, ckpt_path in enumerate(args.checkpoint_dir):
    ckpt_name = Path(ckpt_path).name
    print(f"\n{'='*60}")
    print(f"Checkpoint {ckpt_idx+1}/{len(args.checkpoint_dir)}: {ckpt_name}")
    print(f"  Path: {ckpt_path}")
    print(f"{'='*60}")

    # Verify adapter exists
    if not Path(ckpt_path, "adapter_config.json").exists():
        print(f"  ERROR: adapter_config.json not found, skipping")
        continue

    lora_request = LoRARequest(
        lora_name=ckpt_name,
        lora_int_id=ckpt_idx + 1,
        lora_path=ckpt_path,
    )

    # Run inference
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    t_infer = time.time() - t0

    # Score results
    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []
    num_correct = 0
    total_tokens = 0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)
        total_tokens += n_tokens

        _, answer, category = eval_data[i]
        predicted = extract_final_answer(generated_text)
        correct = verify(str(answer), predicted)

        if correct:
            num_correct += 1
            per_category[category]["correct"] += 1
        per_category[category]["total"] += 1

        details.append({
            "category": category,
            "answer": answer,
            "predicted": predicted,
            "correct": correct,
            "generated_tokens": n_tokens,
            "generated_text_preview": generated_text[:500],
        })

    num_total = len(eval_data)
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    tok_per_sec = total_tokens / t_infer if t_infer > 0 else 0

    # Print results
    print(f"\nOverall: {num_correct}/{num_total} ({accuracy*100:.1f}%)")
    print(f"Time: {t_infer:.1f}s | {tok_per_sec:.1f} tok/s | {t_infer/num_total:.1f}s/sample")
    print(f"\n{'Category':<22} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 50)
    for cat in sorted(per_category):
        c = per_category[cat]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"{cat:<22} {c['correct']:>8} {c['total']:>6} {acc:>10.1%}")

    # Save detailed results
    output_file = Path(args.output_dir) / f"eval_vllm_{ckpt_name}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nResults saved to: {output_file}")

    # Show sample outputs
    print(f"\nSample outputs (first 2 per category):")
    shown = defaultdict(int)
    for d in details:
        cat = d["category"]
        if shown[cat] >= 2:
            continue
        shown[cat] += 1
        status = "correct" if d["correct"] else "WRONG"
        print(f"  [{cat}] {status} | expected={d['answer']} predicted={d['predicted']} | {d['generated_tokens']} tokens")
        print(f"    {d['generated_text_preview'][:200]}")

    all_results[ckpt_name] = {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "time": t_infer,
        "tok_per_sec": tok_per_sec,
        "per_category": {cat: dict(v) for cat, v in per_category.items()},
    }

    # Write to TensorBoard (same run as training)
    if tb_writer is not None:
        import re as _re
        step_match = _re.search(r'checkpoint-(\d+)', ckpt_name)
        if step_match:
            step = int(step_match.group(1))
            tb_writer.add_scalar("eval/accuracy", accuracy, step)
            for cat in sorted(per_category):
                c = per_category[cat]
                cat_acc = c["correct"] / c["total"] if c["total"] > 0 else 0
                tb_writer.add_scalar(f"eval/accuracy_{cat}", cat_acc, step)
            tb_writer.flush()
            print(f"  [tensorboard] Wrote eval metrics at step {step}")

# ── Summary comparison ──────────────────────────────────────────────────────
if len(all_results) > 1:
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Checkpoint':<30} {'Accuracy':>10} {'Time':>8} {'tok/s':>8}")
    print("-" * 60)
    for name, r in all_results.items():
        print(f"{name:<30} {r['num_correct']}/{r['num_total']} ({r['accuracy']*100:.1f}%) {r['time']:>7.1f}s {r['tok_per_sec']:>7.1f}")

if tb_writer is not None:
    tb_writer.close()
    print(f"[tensorboard] Writer closed.")

print(f"\n[done] Evaluation complete.")
