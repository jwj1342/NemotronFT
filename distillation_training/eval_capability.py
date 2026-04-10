"""
distillation_training/eval_capability.py — Capability audit across categories.

Evaluates a checkpoint on a random sample from each category.  Reports accuracy
split by two cohorts:

  - TRAIN: prompts the model saw during training (regular SFT + upgraded distill)
  - UNSEEN: prompts the model never saw

This uses the same seed=42 + distillation-priority logic as MixedDataset to
reconstruct exactly which prompts landed in the training split.

Usage:
  python distillation_training/eval_capability.py \
      --checkpoint_dir <path> \
      --samples_per_class 2000 \
      --categories bit_ops cipher equation_numeric equation_symbolic
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from eval import extract_final_answer, verify, BOXED_INSTRUCTION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = str(PROJECT_ROOT / "model")
DATA_DIR     = PROJECT_ROOT / "data" / "by_category"
DISTILL_DIR  = PROJECT_ROOT / "data" / "distillation"
COT_DIR      = PROJECT_ROOT / "data" / "cot"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True)
parser.add_argument("--samples_per_class", type=int, default=2000,
                    help="Samples per category. 0 = ALL.")
parser.add_argument("--max_tokens", type=int, default=7680)
parser.add_argument("--seed", type=int, default=123,
                    help="Seed for sampling eval data (NOT the training split seed).")
parser.add_argument("--train_seed", type=int, default=42,
                    help="Seed used by MixedDataset to split train/eval (must match).")
parser.add_argument("--train_samples_per_class", type=int, default=600,
                    help="Number of training samples per class (must match training config).")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--categories", type=str, nargs="+", default=None,
                    help="Only evaluate these categories (default: all).")
args = parser.parse_args()


# ── 0. Reconstruct training prompts ─────────────────────────────────────────
# Replicate the exact same split logic as MixedDataset (seed=42, distill priority)
# so we know which prompts were in the training set.

print("[0/3] Reconstructing training split ...")

# Load distillation prompts
distill_prompts = set()
for task_dir in sorted(DISTILL_DIR.iterdir()):
    if not task_dir.is_dir():
        continue
    jsonl_file = task_dir / "distilled.jsonl"
    if not jsonl_file.exists():
        continue
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            distill_prompts.add(json.loads(line)["prompt"])

# Load CoT prompts
cot_prompts = set()
for task_dir in sorted(COT_DIR.iterdir()):
    if not task_dir.is_dir():
        continue
    for level_file in sorted(task_dir.glob("*.jsonl")):
        with open(level_file, "r", encoding="utf-8") as f:
            for line in f:
                cot_prompts.add(json.loads(line)["prompt"])

# Reconstruct per-category training prompts using same logic as MixedDataset
train_rng = random.Random(args.train_seed)
training_prompts = set()  # all prompts that entered the training set

for csv_file in sorted(DATA_DIR.glob("*.csv")):
    category = csv_file.stem
    rows = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row["prompt"])
    train_rng.shuffle(rows)

    # Same priority logic: distill prompts fill train split first
    has_distill = [p for p in rows if p in distill_prompts]
    no_distill = [p for p in rows if p not in distill_prompts]

    n = args.train_samples_per_class
    n_distill = min(len(has_distill), n)
    n_regular_fill = n - n_distill

    # These prompts went into training
    for p in has_distill[:n_distill]:
        training_prompts.add(p)
    for p in no_distill[:n_regular_fill]:
        training_prompts.add(p)

# Also add CoT prompts to training set (they were all used in training)
training_prompts.update(cot_prompts)

print(f"  Reconstructed training set: {len(training_prompts)} prompts")
print(f"    Regular+Upgraded (from CSVs): {len(training_prompts - cot_prompts)}")
print(f"    CoT synthesis: {len(cot_prompts)}")


# ── 1. Load eval samples ────────────────────────────────────────────────────
print(f"\n[1/3] Loading samples ...")
eval_rng = random.Random(args.seed)
eval_data = []  # (prompt, answer, category, is_train)

for csv_file in sorted(DATA_DIR.glob("*.csv")):
    category = csv_file.stem
    if args.categories and category not in args.categories:
        continue
    rows = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["prompt"], row["answer"]))
    eval_rng.shuffle(rows)
    if args.samples_per_class > 0:
        chosen = rows[:args.samples_per_class]
    else:
        chosen = rows

    n_train = sum(1 for p, _ in chosen if p in training_prompts)
    n_unseen = len(chosen) - n_train
    for p, a in chosen:
        is_train = p in training_prompts
        eval_data.append((p, a, category, is_train))
    print(f"  {category:22s}: {len(chosen):5d} samples "
          f"({n_train} train + {n_unseen} unseen)")

total_samples = len(eval_data)
total_train = sum(1 for _, _, _, t in eval_data if t)
total_unseen = total_samples - total_train
print(f"  Total: {total_samples} ({total_train} train + {total_unseen} unseen)")


# ── 2. Run inference ─────────────────────────────────────────────────────────
print(f"\n[2/3] Loading model + LoRA from {args.checkpoint_dir} ...")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model=MODEL_PATH,
    enable_lora=True,
    max_lora_rank=64,
    max_model_len=8192,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True,
    dtype="bfloat16",
    enforce_eager=True,
)

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=args.max_tokens,
    stop=["<|im_end|>"],
)

lora_req = LoRARequest("adapter", 1, args.checkpoint_dir)

tokenizer = llm.get_tokenizer()
prompts = []
for prompt_text, answer, category, is_train in eval_data:
    user_content = prompt_text + BOXED_INSTRUCTION
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompts.append(text)

print(f"  Running inference on {len(prompts)} samples ...")
t0 = time.time()
outputs = llm.generate(prompts, sampling, lora_request=lora_req)
elapsed = time.time() - t0
print(f"  Inference done in {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/sample)")


# ── 3. Score (dual cohort) ──────────────────────────────────────────────────
print(f"\n[3/3] Scoring ...")

# Stats: results_by_cat[category][cohort] = {correct, total}
results = defaultdict(lambda: {
    "train": {"correct": 0, "total": 0},
    "unseen": {"correct": 0, "total": 0},
    "all": {"correct": 0, "total": 0},
})
output_records = []

for (prompt_text, expected, category, is_train), output in zip(eval_data, outputs):
    generated = output.outputs[0].text
    predicted = extract_final_answer(generated)
    is_correct = verify(expected, predicted)

    cohort = "train" if is_train else "unseen"
    results[category][cohort]["total"] += 1
    results[category]["all"]["total"] += 1
    if is_correct:
        results[category][cohort]["correct"] += 1
        results[category]["all"]["correct"] += 1

    output_records.append({
        "category": category,
        "cohort": cohort,
        "expected": expected,
        "predicted": predicted,
        "correct": is_correct,
        "tokens": len(output.outputs[0].token_ids),
    })


# ── Print results ────────────────────────────────────────────────────────────
def pct(c, t):
    return f"{100*c/t:.1f}%" if t > 0 else "N/A"

print(f"\n{'='*90}")
print(f"CAPABILITY AUDIT — {Path(args.checkpoint_dir).name}")
print(f"{'='*90}")

# Header
print(f"\n{'Category':<22s} "
      f"{'--- TRAIN (seen) ---':>22s}  "
      f"{'--- UNSEEN ---':>22s}  "
      f"{'--- ALL ---':>18s}")
print(f"{'':22s} "
      f"{'correct/total':>13s} {'acc':>7s}  "
      f"{'correct/total':>13s} {'acc':>7s}  "
      f"{'correct/total':>13s} {'acc':>7s}")
print("-" * 90)

grand = {"train": {"correct": 0, "total": 0},
         "unseen": {"correct": 0, "total": 0},
         "all": {"correct": 0, "total": 0}}

for category in sorted(results.keys()):
    r = results[category]
    tr = r["train"]
    un = r["unseen"]
    al = r["all"]
    print(f"{category:<22s} "
          f"{tr['correct']:>5d}/{tr['total']:<5d}  {pct(tr['correct'],tr['total']):>7s}  "
          f"{un['correct']:>5d}/{un['total']:<5d}  {pct(un['correct'],un['total']):>7s}  "
          f"{al['correct']:>5d}/{al['total']:<5d}  {pct(al['correct'],al['total']):>7s}")
    for k in grand:
        grand[k]["correct"] += r[k]["correct"]
        grand[k]["total"] += r[k]["total"]

print("-" * 90)
tr, un, al = grand["train"], grand["unseen"], grand["all"]
print(f"{'TOTAL':<22s} "
      f"{tr['correct']:>5d}/{tr['total']:<5d}  {pct(tr['correct'],tr['total']):>7s}  "
      f"{un['correct']:>5d}/{un['total']:<5d}  {pct(un['correct'],un['total']):>7s}  "
      f"{al['correct']:>5d}/{al['total']:<5d}  {pct(al['correct'],al['total']):>7s}")

print(f"\nInference time: {elapsed:.1f}s | {elapsed/len(prompts):.2f}s/sample")

# ── RL feasibility ───────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("RL FEASIBILITY (based on UNSEEN accuracy)")
print(f"{'='*90}")
for category in sorted(results.keys()):
    un = results[category]["unseen"]
    p = 100 * un["correct"] / un["total"] if un["total"] > 0 else 0
    if p == 0:
        status = "BLOCKED — 0% on unseen data, RL cannot bootstrap"
    elif p < 5:
        status = f"RISKY  — only {un['correct']} correct, RL signal very sparse"
    elif p < 20:
        status = f"OK     — {un['correct']} correct samples, RL feasible"
    else:
        status = f"GOOD   — {p:.0f}% baseline, strong RL signal"
    print(f"  {category:<22s} {p:5.1f}%  {status}")

# ── Memorization check ───────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("MEMORIZATION GAP (train acc − unseen acc)")
print(f"{'='*90}")
for category in sorted(results.keys()):
    tr = results[category]["train"]
    un = results[category]["unseen"]
    tr_pct = 100 * tr["correct"] / tr["total"] if tr["total"] > 0 else 0
    un_pct = 100 * un["correct"] / un["total"] if un["total"] > 0 else 0
    gap = tr_pct - un_pct
    if gap > 15:
        note = "HIGH — likely memorizing, not generalizing"
    elif gap > 5:
        note = "MODERATE — some overfitting"
    else:
        note = "LOW — good generalization"
    print(f"  {category:<22s} {tr_pct:5.1f}% → {un_pct:5.1f}%  "
          f"gap={gap:+.1f}pp  {note}")

# ── Save detailed results ────────────────────────────────────────────────────
if args.output_file:
    with open(args.output_file, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nDetailed results saved to {args.output_file}")
