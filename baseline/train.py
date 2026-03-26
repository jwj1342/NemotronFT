"""
baseline/train.py — LoRA fine-tuning baseline for NVIDIA Nemotron-3-Nano-30B.

The script:
  1. Loads the base model in BF16, split across available GPUs via device_map="balanced".
  2. Wraps it with a PEFT LoRA adapter (rank=32, alpha=16).
  3. Stratified-samples train.csv (default: 200 samples per task category).
  4. Trains on a supervised next-token objective, masking the user turn so only the
     assistant response (the \\boxed{answer}) contributes to the loss.
  5. Uses NEFTune (noise_alpha=5) to regularise embedding space.
  6. Saves the adapter to OUTPUT_DIR.

Usage (usually invoked by run_baseline.sbatch):
  python baseline/train.py [--max_steps N] [--output_dir PATH]
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# ── Compatibility patch ────────────────────────────────────────────────────────
# mamba-ssm <= 2.2.4 imports GreedySearchDecoderOnlyOutput / SampleDecoderOnlyOutput
# from transformers.generation, but these were removed in transformers >= 4.40.
# We inject dummy classes so the import chain doesn't fail at model load time.
import transformers.generation as _tg
if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
    @dataclass
    class _Dummy:
        pass
    _tg.GreedySearchDecoderOnlyOutput = _Dummy
    _tg.SampleDecoderOnlyOutput = _Dummy

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = str(PROJECT_ROOT / "model")
DATA_PATH    = str(PROJECT_ROOT / "data" / "train.csv")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",        default=str(PROJECT_ROOT / "output" / "baseline-lora"))
parser.add_argument("--max_steps",   type=int,   default=-1,
                    help="Training steps. -1 = train for a full epoch.")
parser.add_argument("--samples_per_class", type=int, default=200,
                    help="Samples to draw per task category (stratified sampling).")
parser.add_argument("--batch_size",  type=int,   default=4)
parser.add_argument("--grad_acc",    type=int,   default=4)
parser.add_argument("--lr",          type=float, default=2e-4)
parser.add_argument("--max_length",  type=int,   default=512)
parser.add_argument("--lora_rank",   type=int,   default=32)
parser.add_argument("--lora_alpha",  type=int,   default=16)
parser.add_argument("--neftune_alpha", type=float, default=5.0,
                    help="NEFTune noise alpha. Set 0 to disable.")
args = parser.parse_args()

print(f"[config] output_dir        = {args.output_dir}")
print(f"[config] max_steps         = {args.max_steps}")
print(f"[config] samples_per_class = {args.samples_per_class}")
print(f"[config] batch_size        = {args.batch_size}")
print(f"[config] grad_acc          = {args.grad_acc}")
print(f"[config] lr                = {args.lr}")
print(f"[config] max_length        = {args.max_length}")
print(f"[config] lora_rank         = {args.lora_rank}")
print(f"[config] lora_alpha        = {args.lora_alpha}")
print(f"[config] neftune_alpha     = {args.neftune_alpha}")
print(f"[config] torch version     = {torch.__version__}")
print(f"[config] GPUs available    = {torch.cuda.device_count()}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("\n[1/4] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Dataset ───────────────────────────────────────────────────────────────────
BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Think-token fragments that Nemotron's chat template may append to the
# generation prefix when enable_thinking is False. We strip them so the model
# learns to output \boxed{answer} directly without any think-token context.
_THINK_PREFIXES = (
    "<think>\n</think>\n",
    "<think></think>\n",
    "<think>\n\n</think>\n\n",
)


def classify_task(prompt: str) -> str:
    """Heuristic task-type classifier based on prompt keywords."""
    p = prompt.lower()
    if "bit manipulation" in p or "bit shift" in p or "binary" in p:
        return "bit_ops"
    if "encryption" in p or "cipher" in p:
        return "cipher"
    if "numeral system" in p or "number system" in p:
        return "numeral"
    if "gravitational" in p:
        return "gravity"
    if "transformation rules" in p:
        return "symbol"
    if "unit conversion" in p:
        return "unit_conv"
    return "other"


def stratified_sample(rows: list, samples_per_class: int, seed: int = 42) -> list:
    """Return a stratified subsample: up to `samples_per_class` rows per category."""
    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    for row in rows:
        buckets[classify_task(row[0])].append(row)

    sampled = []
    for task in sorted(buckets):
        task_rows = buckets[task]
        rng.shuffle(task_rows)
        chosen = task_rows[:samples_per_class]
        sampled.extend(chosen)
        print(f"[dataset]   {task:12s}: {len(chosen):4d} / {len(task_rows):5d} sampled")

    rng.shuffle(sampled)
    return sampled


def build_conversation(prompt: str, answer: str) -> tuple[str, int]:
    """
    Returns (full_text, prefix_len) where prefix_len is the number of tokens
    that should be masked out (user turn + assistant-start tokens, including
    any think-token preamble added by the chat template).

    The training target is exactly: \\boxed{answer}<|im_end|>
    No <think></think> tokens appear in the supervised signal.
    """
    user_content = prompt + BOXED_INSTRUCTION

    # Get everything up to (and including) the start of the assistant turn.
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Strip any think-token preamble that the template appended to the
    # generation prefix so it doesn't bleed into the training target.
    for think_frag in _THINK_PREFIXES:
        if prefix_text.endswith(think_frag):
            prefix_text = prefix_text[:-len(think_frag)]
            break

    # Append the answer directly — no think tokens in the training target.
    full_text = prefix_text + f"\\boxed{{{answer}}}<|im_end|>\n"

    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    return full_text, len(prefix_ids)


class NemotronSFTDataset(Dataset):
    def __init__(self, csv_path: str, samples_per_class: int = 200, max_length: int = 1024):
        print(f"[dataset] Loading {csv_path} ...")
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["prompt"], row["answer"]))
        print(f"[dataset] Total rows in CSV: {len(rows)}")
        print(f"[dataset] Stratified sampling ({samples_per_class} per class) ...")
        sampled = stratified_sample(rows, samples_per_class)
        print(f"[dataset] {len(sampled)} samples after stratified sampling.")

        # Pre-tokenize all samples once at init time so __getitem__ is a
        # simple list lookup. This moves CPU-heavy work (Jinja2 templating,
        # tokenisation) out of the training loop and eliminates GPU starvation.
        print(f"[dataset] Pre-tokenizing {len(sampled)} samples (max_length={max_length}) ...")
        self.samples = []
        skipped = 0
        for prompt, answer in sampled:
            full_text, prefix_len = build_conversation(prompt, answer)
            encoding = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            labels = list(input_ids)
            mask_len = min(prefix_len, len(labels))
            for i in range(mask_len):
                labels[i] = -100

            if all(l == -100 for l in labels):
                labels[-1] = input_ids[-1]
                skipped += 1

            self.samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

        print(f"[dataset] Pre-tokenization done. "
              f"({skipped} samples had fully-masked labels and were patched.)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
print("\n[2/4] Loading base model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="balanced",        # split evenly across all visible GPUs
    trust_remote_code=True,
)
# Prevent Trainer from wrapping the model in DataParallel (incompatible with device_map).
model.is_parallelizable = True
model.model_parallel = True

# ── LoRA ──────────────────────────────────────────────────────────────────────
print(f"\n[3/4] Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}) ...")
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing to reduce activation memory.
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ── NEFTune ───────────────────────────────────────────────────────────────────
# Inject uniform noise into the embedding output during training.
# This improves generalisation on small fine-tuning datasets (NEFTune, 2023).
if args.neftune_alpha > 0:
    _neftune_alpha = args.neftune_alpha

    def _neftune_hook(module, input, output):
        if module.training:
            seq_len = output.shape[1]
            hidden  = output.shape[2]
            mag = _neftune_alpha / (seq_len * hidden) ** 0.5
            noise = torch.zeros_like(output).uniform_(-mag, mag)
            return output + noise
        return output

    model.get_input_embeddings().register_forward_hook(_neftune_hook)
    print(f"[neftune] Registered embedding noise hook (alpha={_neftune_alpha})")

# ── Data ──────────────────────────────────────────────────────────────────────
dataset = NemotronSFTDataset(
    DATA_PATH,
    samples_per_class=args.samples_per_class,
    max_length=args.max_length,
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

# ── Training ──────────────────────────────────────────────────────────────────
print(f"\n[4/4] Starting training ...")
os.makedirs(args.output_dir, exist_ok=True)

# Warmup ratio: 5% of total steps (防止训练初期大梯度破坏预训练权重).
training_args = TrainingArguments(
    output_dir=args.output_dir,
    # Steps / epochs
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    num_train_epochs=1 if args.max_steps <= 0 else 100,  # epochs ignored when max_steps set
    # Batch
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc,
    # Optimizer — cosine decay + warmup
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    # Precision
    bf16=True,
    fp16=False,
    # Memory
    gradient_checkpointing=False,   # already enabled manually above
    dataloader_pin_memory=True,
    dataloader_num_workers=0,
    # Logging / saving
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()

# ── Save adapter ──────────────────────────────────────────────────────────────
print(f"\n[done] Saving LoRA adapter to {args.output_dir} ...")
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("[done] Training complete.")
