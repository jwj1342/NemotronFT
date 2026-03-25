"""
baseline/train.py — LoRA fine-tuning baseline for NVIDIA Nemotron-3-Nano-30B.

The script:
  1. Loads the base model in BF16, split across available GPUs via device_map="balanced".
  2. Wraps it with a PEFT LoRA adapter (rank=32, matching the competition target modules).
  3. Trains on train.csv with a supervised next-token objective, masking the user turn
     so only the assistant response (the \\boxed{answer}) contributes to the loss.
  4. Saves the adapter to OUTPUT_DIR.

Usage (usually invoked by run_baseline.sbatch):
  python baseline/train.py [--max_steps N] [--output_dir PATH]
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Compatibility patch ────────────────────────────────────────────────────────
# mamba-ssm <= 2.2.4 imports GreedySearchDecoderOnlyOutput / SampleDecoderOnlyOutput
# from transformers.generation, but these were removed in transformers >= 4.40.
# We inject dummy classes so the import chain doesn't fail at model load time.
# Training doesn't use mamba-ssm's generation utilities at all.
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
parser.add_argument("--output_dir",          default=str(PROJECT_ROOT / "output" / "baseline-lora"))
parser.add_argument("--max_steps",     type=int, default=500,
                    help="Training steps. Use -1 to train for a full epoch.")
parser.add_argument("--num_train_samples", type=int, default=-1,
                    help="Subsample training set. -1 = use all.")
parser.add_argument("--batch_size",    type=int, default=1)
parser.add_argument("--grad_acc",      type=int, default=8)
parser.add_argument("--lr",            type=float, default=2e-4)
parser.add_argument("--max_length",    type=int, default=2048)
parser.add_argument("--lora_rank",     type=int, default=32)
parser.add_argument("--lora_alpha",    type=int, default=64)
args = parser.parse_args()

print(f"[config] output_dir       = {args.output_dir}")
print(f"[config] max_steps        = {args.max_steps}")
print(f"[config] batch_size       = {args.batch_size}")
print(f"[config] grad_acc         = {args.grad_acc}")
print(f"[config] lr               = {args.lr}")
print(f"[config] max_length       = {args.max_length}")
print(f"[config] lora_rank        = {args.lora_rank}")
print(f"[config] torch version    = {torch.__version__}")
print(f"[config] GPUs available   = {torch.cuda.device_count()}")

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

def build_conversation(prompt: str, answer: str) -> tuple[str, int]:
    """
    Returns (full_text, prefix_len) where prefix_len is the number of tokens
    that should be masked out (user turn + assistant-start tokens).
    """
    user_content = prompt + BOXED_INSTRUCTION

    # Build the prefix (user message + start of assistant turn).
    # apply_chat_template with add_generation_prompt=True gives us the prompt
    # up to (but not including) the model's actual reply.
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # baseline: skip chain-of-thought
    )

    # Build the full conversation including the answer.
    # The chat template prepends <think></think> to assistant content automatically.
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Tokenize both to get the boundary index.
    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    return full_text, len(prefix_ids)


class NemotronSFTDataset(Dataset):
    def __init__(self, csv_path: str, max_samples: int = -1, max_length: int = 2048):
        print(f"[dataset] Loading {csv_path} ...")
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["prompt"], row["answer"]))
        if max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows
        self.max_length = max_length
        print(f"[dataset] {len(self.rows)} samples loaded.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        prompt, answer = self.rows[idx]
        full_text, prefix_len = build_conversation(prompt, answer)

        encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Labels: mask the user prefix with -100; train only on the assistant response.
        labels = list(input_ids)
        mask_len = min(prefix_len, len(labels))
        for i in range(mask_len):
            labels[i] = -100

        # If the entire sequence was masked (very long prompt), skip gracefully.
        if all(l == -100 for l in labels):
            # Fall back: unmask last token so loss doesn't become NaN.
            labels[-1] = input_ids[-1]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


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
print(f"\n[3/4] Applying LoRA (rank={args.lora_rank}) ...")
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    # Mirrors the competition's submission demo target modules.
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

# ── Data ──────────────────────────────────────────────────────────────────────
dataset = NemotronSFTDataset(
    DATA_PATH,
    max_samples=args.num_train_samples,
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

training_args = TrainingArguments(
    output_dir=args.output_dir,
    # Steps / epochs
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    num_train_epochs=1 if args.max_steps <= 0 else 100,  # epochs ignored when max_steps set
    # Batch
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc,
    # Optimizer
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    # Precision
    bf16=True,
    fp16=False,
    # Memory
    gradient_checkpointing=False,   # already enabled manually above
    dataloader_pin_memory=False,    # required with device_map
    dataloader_num_workers=4,
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
