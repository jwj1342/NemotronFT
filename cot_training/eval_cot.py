"""
cot_training/eval_cot.py — Evaluate CoT checkpoint with Kaggle-aligned token budget.

Runs inference on the validation set using max_new_tokens=7680 (matching Kaggle's
official evaluation) to get accurate per-category accuracy for CoT checkpoints.

Also saves generated outputs to a JSONL file for manual inspection.

Usage:
  python cot_training/eval_cot.py --checkpoint_dir <path> [--max_new_tokens 7680]
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Compatibility patch ────────────────────────────────────────────────────────
import transformers.generation as _tg
if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
    @dataclass
    class _Dummy:
        pass
    _tg.GreedySearchDecoderOnlyOutput = _Dummy
    _tg.SampleDecoderOnlyOutput = _Dummy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import eval logic from baseline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from eval import extract_final_answer, verify, BOXED_INSTRUCTION

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = str(PROJECT_ROOT / "model")
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "by_category"

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", required=True,
                    help="Path to LoRA checkpoint to evaluate.")
parser.add_argument("--max_new_tokens", type=int, default=7680,
                    help="Max new tokens for generation (Kaggle uses 7680).")
parser.add_argument("--eval_samples_per_class", type=int, default=5,
                    help="Number of eval samples per category.")
parser.add_argument("--train_samples_per_class", type=int, default=400,
                    help="Number of training samples to skip (same split as baseline).")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for generation (1 for long sequences).")
parser.add_argument("--output_file", type=str, default=None,
                    help="Path to save detailed results JSONL.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

checkpoint_name = Path(args.checkpoint_dir).name
if args.output_file is None:
    args.output_file = str(PROJECT_ROOT / "output" / f"eval_results_{checkpoint_name}.jsonl")

print(f"[config] checkpoint_dir    = {args.checkpoint_dir}")
print(f"[config] max_new_tokens    = {args.max_new_tokens}")
print(f"[config] eval_samples/cls  = {args.eval_samples_per_class}")
print(f"[config] batch_size        = {args.batch_size}")
print(f"[config] output_file       = {args.output_file}")
print(f"[config] seed              = {args.seed}")
print(f"[config] torch version     = {torch.__version__}")
print(f"[config] GPUs available    = {torch.cuda.device_count()}")

# ── Load eval data ───────────────────────────────────────────────────────────
def load_eval_pool(data_dir, train_samples_per_class, eval_samples_per_class, seed):
    rng = random.Random(seed)
    eval_data = []
    for csv_file in sorted(Path(data_dir).glob("*.csv")):
        category = csv_file.stem
        rows = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["prompt"], row["answer"]))
        rng.shuffle(rows)
        remaining = rows[train_samples_per_class:]
        chosen = remaining[:eval_samples_per_class]
        eval_data.extend([(p, a, category) for p, a in chosen])
    return eval_data

print("\n[1/3] Loading eval data ...")
eval_data = load_eval_pool(
    EVAL_DATA_DIR, args.train_samples_per_class,
    args.eval_samples_per_class, args.seed,
)
print(f"  {len(eval_data)} eval samples loaded")

# ── Load model + checkpoint ──────────────────────────────────────────────────
print("\n[2/3] Loading model + LoRA adapter ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="balanced",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, args.checkpoint_dir)
model.eval()
print(f"  Model loaded on {model.device}")

# ── Run evaluation ───────────────────────────────────────────────────────────
print(f"\n[3/3] Running evaluation (max_new_tokens={args.max_new_tokens}) ...")

from collections import defaultdict
per_category = defaultdict(lambda: {"correct": 0, "total": 0})
details = []
num_correct = 0
num_total = len(eval_data)

# Pre-build chat strings
chat_strings = []
for prompt, answer, category in eval_data:
    user_content = prompt + BOXED_INSTRUCTION
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    chat_strings.append(chat_input)

t0 = time.time()
for i in range(0, num_total, args.batch_size):
    batch_end = min(i + args.batch_size, num_total)
    batch_chats = chat_strings[i:batch_end]
    batch_meta = eval_data[i:batch_end]
    cur_bs = len(batch_chats)

    inputs = tokenizer(
        batch_chats,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_p=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    for j in range(cur_bs):
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[j][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        _, answer, category = batch_meta[j]
        predicted = extract_final_answer(generated_text)
        correct = verify(str(answer), predicted)

        if correct:
            num_correct += 1
            per_category[category]["correct"] += 1
        per_category[category]["total"] += 1

        # Save detail with truncated generation for inspection
        detail = {
            "category": category,
            "answer": answer,
            "predicted": predicted,
            "correct": correct,
            "generated_tokens": len(generated_ids),
            "generated_text_preview": generated_text[:500] + ("..." if len(generated_text) > 500 else ""),
        }
        details.append(detail)

    processed = batch_end
    elapsed = time.time() - t0
    avg = elapsed / processed
    eta = avg * (num_total - processed)
    sys.stdout.write(
        f"\r[eval] {processed}/{num_total} "
        f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left, {avg:.1f}s/sample] "
        f"acc={num_correct}/{processed}   "
    )
    sys.stdout.flush()

print()
elapsed_total = time.time() - t0

# ── Print results ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"EVALUATION RESULTS — {Path(args.checkpoint_dir).name}")
print(f"max_new_tokens={args.max_new_tokens} (Kaggle-aligned)")
print(f"{'='*60}")
print(f"\nOverall: {num_correct}/{num_total} ({num_correct/num_total*100:.1f}%)")
print(f"Time: {elapsed_total:.0f}s ({elapsed_total/num_total:.1f}s/sample)\n")

print(f"{'Category':<22} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
print("-" * 50)
for cat in sorted(per_category):
    c = per_category[cat]
    acc = c["correct"] / c["total"] if c["total"] > 0 else 0
    print(f"{cat:<22} {c['correct']:>8} {c['total']:>6} {acc:>10.1%}")

# ── Save detailed results ────────────────────────────────────────────────────
Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
with open(args.output_file, "w", encoding="utf-8") as f:
    for d in details:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"\nDetailed results saved to: {args.output_file}")

# ── Print sample outputs for inspection ──────────────────────────────────────
print(f"\n{'='*60}")
print("SAMPLE OUTPUTS (first 2 per category)")
print(f"{'='*60}")
shown = defaultdict(int)
for d in details:
    cat = d["category"]
    if shown[cat] >= 2:
        continue
    shown[cat] += 1
    status = "✓" if d["correct"] else "✗"
    print(f"\n[{cat}] {status} expected={d['answer']} predicted={d['predicted']} tokens={d['generated_tokens']}")
    print(f"  Output: {d['generated_text_preview'][:300]}")
