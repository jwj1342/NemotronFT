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
    TrainerCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import evaluate_samples

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = str(PROJECT_ROOT / "model")
DATA_DIR     = PROJECT_ROOT / "data" / "by_category"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",        default=str(PROJECT_ROOT / "output" / "baseline-lora"))
parser.add_argument("--max_steps",   type=int,   default=-1,
                    help="Training steps. -1 = train for a full epoch.")
parser.add_argument("--samples_per_class", type=int, default=400,
                    help="Samples to draw per task category (stratified sampling).")
parser.add_argument("--batch_size",  type=int,   default=4)
parser.add_argument("--grad_acc",    type=int,   default=4)
parser.add_argument("--lr",          type=float, default=2e-4)
parser.add_argument("--max_length",  type=int,   default=512)
parser.add_argument("--lora_rank",   type=int,   default=32)
parser.add_argument("--lora_alpha",  type=int,   default=16)
parser.add_argument("--neftune_alpha", type=float, default=5.0,
                    help="NEFTune noise alpha. Set 0 to disable.")
parser.add_argument("--eval_steps", type=int, default=25,
                    help="Run evaluation every N steps. 0 to disable.")
parser.add_argument("--eval_samples_per_class", type=int, default=5,
                    help="Number of eval samples per category.")
parser.add_argument("--eval_batch_size", type=int, default=4,
                    help="Batch size for eval generation.")
parser.add_argument("--tensorboard_dir", type=str,
                    default="/home/jwj/tensorboard_logs",
                    help="TensorBoard log directory.")
parser.add_argument("--enable_thinking", action="store_true",
                    help="Enable <think> mode: train on CoT reasoning inside "
                         "<think>...</think> tags. Requires CoT-formatted data.")
parser.add_argument("--cot_data_dir", type=str, default=None,
                    help="Optional directory containing *_think.csv files with "
                         "CoT training data. These are added to the training set "
                         "alongside the original data from --data_dir.")
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
print(f"[config] eval_steps        = {args.eval_steps}")
print(f"[config] eval_samples/cls  = {args.eval_samples_per_class}")
print(f"[config] eval_batch_size  = {args.eval_batch_size}")
print(f"[config] tensorboard_dir   = {args.tensorboard_dir}")
print(f"[config] enable_thinking   = {args.enable_thinking}")
print(f"[config] cot_data_dir      = {args.cot_data_dir}")
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



def build_conversation(prompt: str, answer: str) -> tuple[str, int]:
    r"""
    Returns (full_text, prefix_len) where prefix_len is the number of tokens
    that should be masked out (user turn + assistant-start tokens).

    Two modes controlled by --enable_thinking:

    1. enable_thinking=False (original, answer-only):
       Training target: \boxed{answer}<|im_end|>
       No <think> tokens in the supervised signal.

    2. enable_thinking=True (CoT reasoning):
       Training target: <reasoning>\n</think>\n\boxed{answer}<|im_end|>
       The prefix includes the template's <think>\n preamble (masked out).
       If the answer already contains </think>, it's used as-is (CoT data).
       Otherwise it's auto-wrapped as </think>\n\boxed{answer} (direct data).
    """
    user_content = prompt + BOXED_INSTRUCTION

    if args.enable_thinking:
        # ── Thinking mode: train on <think>reasoning</think>\boxed{ans} ──
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # The template's prefix ends with "<think>\n".
        # answer is the training target that follows.
        if "</think>" in answer:
            # CoT data: answer = "reasoning\n</think>\n\boxed{ans}"
            full_text = prefix_text + answer + "<|im_end|>\n"
        else:
            # Direct data: auto-wrap plain answer into thinking format
            full_text = (prefix_text
                         + f"</think>\n\\boxed{{{answer}}}<|im_end|>\n")
    else:
        # ── Original mode: answer-only, no think tokens ──
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for think_frag in _THINK_PREFIXES:
            if prefix_text.endswith(think_frag):
                prefix_text = prefix_text[:-len(think_frag)]
                break
        full_text = prefix_text + f"\\boxed{{{answer}}}<|im_end|>\n"

    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    return full_text, len(prefix_ids)


class NemotronSFTDataset(Dataset):
    def __init__(self, data_dir: str, samples_per_class: int = 400,
                 max_length: int = 512, cot_data_dir: str = None):
        print(f"[dataset] Loading per-category CSVs from {data_dir} ...")
        rng = random.Random(42)
        sampled = []
        self.eval_pool = {}  # {category: [(prompt, answer), ...]}
        for csv_file in sorted(Path(data_dir).glob("*.csv")):
            category = csv_file.stem
            rows = []
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append((row["prompt"], row["answer"]))
            rng.shuffle(rows)
            chosen = rows[:samples_per_class]
            remaining = rows[samples_per_class:]
            sampled.extend(chosen)
            self.eval_pool[category] = remaining
            print(f"[dataset]   {category:22s}: {len(chosen):4d} train / {len(remaining):4d} eval / {len(rows):5d} total")

        # ── Load CoT data (additional training samples) ──────────────────────
        if cot_data_dir:
            cot_dir = Path(cot_data_dir)
            cot_count = 0
            for csv_file in sorted(cot_dir.rglob("*_think.csv")):
                rows = []
                with open(csv_file, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        rows.append((row["prompt"], row["answer"]))
                sampled.extend(rows)
                cot_count += len(rows)
                print(f"[dataset]   CoT {csv_file.stem:18s}: {len(rows):4d} samples")
            print(f"[dataset] {cot_count} CoT samples added to training set.")

        rng.shuffle(sampled)
        print(f"[dataset] {len(sampled)} train samples, "
              f"{sum(len(v) for v in self.eval_pool.values())} eval pool samples.")

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
    str(DATA_DIR),
    samples_per_class=args.samples_per_class,
    max_length=args.max_length,
    cot_data_dir=args.cot_data_dir,
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
    report_to="tensorboard",
    logging_dir=args.tensorboard_dir,
    remove_unused_columns=False,
)

# ── Eval callback ────────────────────────────────────────────────────────────
callbacks = []
if args.eval_steps > 0:
    eval_data = []
    for category, pool in sorted(dataset.eval_pool.items()):
        chosen = pool[:args.eval_samples_per_class]
        eval_data.extend([(p, a, category) for p, a in chosen])
    print(f"[eval] {len(eval_data)} eval samples across "
          f"{len(dataset.eval_pool)} categories (eval every {args.eval_steps} steps)")

    class EvalCallback(TrainerCallback):
        def __init__(self, eval_steps, eval_batch_size):
            self.eval_steps = eval_steps
            self.eval_batch_size = eval_batch_size
            self._tb_writer = None

        def _get_tb_writer(self, trainer):
            """Lazily retrieve the TensorBoard writer from Trainer."""
            if self._tb_writer is None:
                for cb in trainer.callback_handler.callbacks:
                    if hasattr(cb, "tb_writer") and cb.tb_writer is not None:
                        self._tb_writer = cb.tb_writer
                        break
            return self._tb_writer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % self.eval_steps == 0:
                import time
                t0 = time.time()
                results = evaluate_samples(model, tokenizer, eval_data,
                                           batch_size=self.eval_batch_size)
                elapsed = time.time() - t0

                tb_writer = self._get_tb_writer(kwargs.get("trainer") or trainer)
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/accuracy", results["accuracy"], state.global_step)
                    for cat, cr in results["per_category"].items():
                        tb_writer.add_scalar(f"eval/accuracy_{cat}", cr["accuracy"], state.global_step)
                    tb_writer.flush()

                cat_str = " | ".join(
                    f"{c}: {r['correct']}/{r['total']}"
                    for c, r in sorted(results["per_category"].items())
                )
                print(f"\n[eval @ step {state.global_step}] "
                      f"accuracy={results['accuracy']:.3f} "
                      f"({results['num_correct']}/{results['num_total']}) "
                      f"[{elapsed:.1f}s]")
                print(f"[eval]   {cat_str}")

    eval_cb = EvalCallback(args.eval_steps, args.eval_batch_size)
    callbacks.append(eval_cb)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    callbacks=callbacks,
)

trainer.train()

# ── Final eval ────────────────────────────────────────────────────────────────
if args.eval_steps > 0:
    print("\n[final eval] Running final evaluation ...")
    final_results = evaluate_samples(model, tokenizer, eval_data, batch_size=args.eval_batch_size)

    tb_writer = eval_cb._get_tb_writer(trainer)
    if tb_writer is not None:
        tb_writer.add_scalar("eval/accuracy", final_results["accuracy"], trainer.state.global_step)
        for cat, cr in final_results["per_category"].items():
            tb_writer.add_scalar(f"eval/accuracy_{cat}", cr["accuracy"], trainer.state.global_step)
        tb_writer.flush()

    print(f"[final eval] accuracy={final_results['accuracy']:.3f} "
          f"({final_results['num_correct']}/{final_results['num_total']})")
    for cat, cr in sorted(final_results["per_category"].items()):
        print(f"[final eval]   {cat:22s}: {cr['correct']}/{cr['total']} = {cr['accuracy']:.3f}")

# ── Save adapter ──────────────────────────────────────────────────────────────
print(f"\n[done] Saving LoRA adapter to {args.output_dir} ...")
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("[done] Training complete.")
