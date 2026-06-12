"""
v6_training/train_v6.py — FSDP LoRA training mirroring the proven Unsloth 0.81 recipe.

Differences from train_v3.py (the failed v5):
- LoRA target_modules includes in_proj / out_proj (Mamba layers) -- v3 missed these
- alpha == rank (32/32), dropout=0   (v3 had 32/64, 0.05)
- max_length=8192                    (v3 had 4096)
- lr=2e-4 linear schedule, no warmup (v3 had cosine + warmup=20)
- adam_beta2=0.95                    (v3 had default 0.999)
- max_grad_norm=1e9 (disabled)       (v3 had default 1.0)
- No data filtering                  (v3 filtered response<=2500)
- Re-tokenized via apply_chat_template enable_thinking=True
  (v3 used Huikang's pre-tokenized data directly)
- Stratified batches by category
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Compatibility patches for torch 2.5.1 + transformers 5.3.0
import transformers.generation as _tg
if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
    @dataclass
    class _Dummy:
        pass
    _tg.GreedySearchDecoderOnlyOutput = _Dummy
    _tg.SampleDecoderOnlyOutput = _Dummy

import torch
import torch.distributed.fsdp as _fsdp_mod
if not hasattr(_fsdp_mod, "register_fsdp_forward_method"):
    _fsdp_mod.register_fsdp_forward_method = lambda *a, **kw: None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import HuikangCoTDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "model")


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "output" / "sft"))
parser.add_argument("--csv_path", default=str(PROJECT_ROOT / "training" / "train_with_cot.csv"))
parser.add_argument("--lora_rank", type=int, default=32)
parser.add_argument("--lora_alpha", type=int, default=32,
                    help="Proven Unsloth 0.81 recipe uses alpha=rank, NOT 2*rank.")
parser.add_argument("--lora_dropout", type=float, default=0.0)
parser.add_argument("--max_length", type=int, default=8192)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--all_linear", action="store_true",
                    help="Use target_modules='all-linear' (kienngx 0.86 recipe) "
                         "instead of the explicit attn+mamba+moe+lm_head list.")
parser.add_argument("--plain_answer", action="store_true",
                    help="kienngx recipe: train on raw `answer` (no CoT, "
                         "enable_thinking=False). Reads answer col, ignores generated_cot.")
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=2,
                    help="Per-device micro batch. 2 fits L40s; total = 2*8*4 = 64.")
parser.add_argument("--grad_acc", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--save_steps", type=int, default=20)
parser.add_argument("--save_total_limit", type=int, default=-1)
parser.add_argument("--save_only_model", action="store_true")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--tensorboard_dir", type=str, default="/home/jwj/tensorboard_logs")
parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="DEPRECATED for PEFT: HF Trainer's resume_from_checkpoint cannot "
                         "detect PEFT adapter after FSDP wrap. Use --resume_adapter_dir instead.")
parser.add_argument("--resume_adapter_dir", type=str, default=None,
                    help="Existing checkpoint dir to manually load LoRA adapter from. "
                         "Use --max_steps + --lr to control the remaining LR schedule.")
args = parser.parse_args()


local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_main = rank == 0


if is_main:
    print("=" * 60)
    print("v6 Training — Configuration (Unsloth 0.81 recipe)")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:30s} = {v}")
    print(f"  {'torch_version':30s} = {torch.__version__}")
    print(f"  {'world_size':30s} = {world_size}")
    print(f"  {'effective_batch':30s} = {args.batch_size * args.grad_acc * world_size}")
    print("=" * 60)


# Tokenizer
if is_main:
    print("\n[1/5] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Dataset (loads on every rank — small CSV)
if is_main:
    print("\n[2/5] Loading CoT dataset ...")
dataset = HuikangCoTDataset(
    csv_path=args.csv_path,
    tokenizer=tokenizer,
    max_length=args.max_length,
    mask_prompt=True,
    plain_answer=args.plain_answer,
)
if is_main:
    print(dataset.summary())


# Stratified batches by category (mirrors Huikang's _stratified_batches in train_sft.py)
def make_stratified_order(categories: list[str], effective_batch: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(categories):
        by_cat[c].append(i)
    for indices in by_cat.values():
        rng.shuffle(indices)

    n = len(categories)
    n_batches = max(1, math.ceil(n / effective_batch))
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)

    assigned = 0
    for cat in sorted(by_cat.keys()):
        for idx in by_cat[cat]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1
    return [idx for batch in batches for idx in batch]


effective_batch = args.batch_size * args.grad_acc * world_size
stratified_order = make_stratified_order(dataset.categories, effective_batch, args.seed)


class StratifiedTrainer(Trainer):
    """HF Trainer that uses a precomputed stratified index order."""

    def __init__(self, *a, stratified_order=None, **kw):
        super().__init__(*a, **kw)
        self.stratified_order = stratified_order

    def _get_train_sampler(self, *args, **kwargs):
        # Disable HF's default sampler; we use a sampler that returns indices
        # in the precomputed stratified order. Distributed rank slicing is
        # handled by Trainer's DataLoader collator -- but for FSDP we shard
        # the model not the data, so each rank reads the full order and
        # batch index = rank_offset within effective batch.
        from torch.utils.data import Sampler

        class _Sampler(Sampler):
            def __init__(self, order):
                self.order = order
            def __iter__(self):
                return iter(self.order)
            def __len__(self):
                return len(self.order)
        return _Sampler(self.stratified_order)


collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)


# Model + LoRA (fresh adapter — cold start, per Unsloth recipe)
if is_main:
    print("\n[3/5] Loading base model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

if args.resume_adapter_dir:
    if is_main:
        print(f"[3/5] Loading LoRA from {args.resume_adapter_dir} ...")
    model = PeftModel.from_pretrained(model, args.resume_adapter_dir, is_trainable=True)
else:
    if is_main:
        print(f"[3/5] Creating LoRA (r={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
        print(f"      target_modules covers attn + mamba (in_proj, out_proj) + MoE MLP + lm_head")
    # CRITICAL: include in_proj and out_proj for Mamba layers.
    # all_linear (kienngx 0.86 recipe) auto-covers every nn.Linear incl. Mamba
    # in_proj/out_proj and per-expert up/down_proj — and produces Kaggle-format
    # per-expert LoRA keys directly (our base model is per-expert, verified), so
    # NO lossy Tinker→Kaggle SVD conversion is needed.
    target_modules = "all-linear" if args.all_linear else [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "in_proj", "out_proj",                   # Mamba layers
        "up_proj", "down_proj",                  # MoE expert MLPs
        "lm_head",
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

if is_main:
    model.print_trainable_parameters()

model = model.to(torch.bfloat16)
model.enable_input_require_grads()


# Training args
if is_main:
    print("\n[4/5] Configuring training ...")
os.makedirs(args.output_dir, exist_ok=True)


training_args = TrainingArguments(
    output_dir=args.output_dir,
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    num_train_epochs=args.num_epochs if args.max_steps <= 0 else 100,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc,
    learning_rate=args.lr,
    lr_scheduler_type="linear",
    warmup_steps=args.warmup_steps,
    optim="adamw_torch_fused",
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=0.0,
    max_grad_norm=1e9,  # disabled
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    logging_steps=1,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
    save_only_model=args.save_only_model,
    report_to="tensorboard",
    logging_dir=args.tensorboard_dir,
    remove_unused_columns=False,
    seed=args.seed,
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_min_num_params": 1_000_000,
        "fsdp_transformer_layer_cls_to_wrap": None,
        "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
        "fsdp_forward_prefetch": False,
        "fsdp_offload_params": False,
        "fsdp_use_orig_params": True,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_sync_module_states": True,
        "activation_checkpointing": False,
    },
)


if is_main:
    print("\n[5/5] Starting v6 training ...")

trainer = StratifiedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    stratified_order=stratified_order,
)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if is_main:
    print(f"\n[done] Saving final adapter to {args.output_dir} ...")
trainer.save_model(args.output_dir)
if is_main:
    tokenizer.save_pretrained(args.output_dir)
    print("[done] v6 training complete.")
