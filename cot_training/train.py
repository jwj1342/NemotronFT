"""
cot_training/train.py — Mixed CoT + SFT training with per-token loss weighting.

Combines two data sources into a single shuffled training set:
  - Regular SFT samples (7 categories × 500 from train.csv)
      Target: </think>\\n\\n\\boxed{answer}<|im_end|>
  - Synthetic CoT samples (bit_ops, equation_symbolic, cipher)
      Target: {reasoning}\\n</think>\\n\\n\\boxed{answer}<|im_end|>

The \\boxed{answer} portion of CoT samples receives higher loss weight
(--boxed_loss_weight) than the reasoning portion (--reasoning_loss_weight),
so the model prioritises correct final answers over imitating reasoning patterns.

Continues training from the best baseline LoRA checkpoint.

Usage:
  python cot_training/train.py [--checkpoint_dir PATH] [--output_dir PATH]
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Compatibility patch ──────────────────────────────────────────────────────
import transformers.generation as _tg
if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
    @dataclass
    class _Dummy:
        pass
    _tg.GreedySearchDecoderOnlyOutput = _Dummy
    _tg.SampleDecoderOnlyOutput = _Dummy

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from peft import PeftModel

# Local modules
from dataset import MixedDataset, load_eval_pool
from weighted_trainer import WeightedDataCollator, WeightedTrainer

# Eval from baseline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from eval import evaluate_samples

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = str(PROJECT_ROOT / "model")
REGULAR_DATA_DIR = PROJECT_ROOT / "data" / "by_category"
COT_DATA_DIR     = PROJECT_ROOT / "data" / "cot"
DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "output" / "baseline-lora" / "checkpoint-100")

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "output" / "cot-lora"))
parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT,
                    help="Baseline LoRA checkpoint to continue from.")
# Data
parser.add_argument("--regular_samples_per_class", type=int, default=500,
                    help="Regular SFT samples per category.")
parser.add_argument("--eval_samples_per_class", type=int, default=5,
                    help="Eval samples per category.")
parser.add_argument("--max_length", type=int, default=2048,
                    help="Max sequence length.")
# Loss weighting
parser.add_argument("--boxed_loss_weight", type=float, default=5.0,
                    help="Loss weight for \\boxed{} tokens in CoT samples.")
parser.add_argument("--reasoning_loss_weight", type=float, default=1.0,
                    help="Loss weight for reasoning tokens in CoT samples.")
# Training
parser.add_argument("--max_steps", type=int, default=-1,
                    help="Training steps. -1 = full epoch(s).")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to checkpoint dir to resume training from. "
                         "Restores optimizer, scheduler, step count and RNG state.")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Per-device batch size (1 due to long CoT sequences).")
parser.add_argument("--grad_acc", type=int, default=16,
                    help="Gradient accumulation steps.")
parser.add_argument("--lr", type=float, default=5e-5,
                    help="Learning rate.")
parser.add_argument("--neftune_alpha", type=float, default=5.0,
                    help="NEFTune noise alpha. 0 to disable.")
# Eval
parser.add_argument("--eval_steps", type=int, default=200,
                    help="Run eval every N steps. 0 to disable.")
parser.add_argument("--eval_max_new_tokens", type=int, default=7680,
                    help="Max new tokens for eval generation (Kaggle uses 7680).")
parser.add_argument("--eval_batch_size", type=int, default=1,
                    help="Batch size for eval (1 for long generation).")
# Infra
parser.add_argument("--tensorboard_dir", type=str,
                    default="/home/jwj/tensorboard_logs")
parser.add_argument("--save_steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Print config ─────────────────────────────────────────────────────────────
print("=" * 60)
print("CoT Mixed Training — Configuration")
print("=" * 60)
for k, v in sorted(vars(args).items()):
    print(f"  {k:30s} = {v}")
print(f"  {'torch_version':30s} = {torch.__version__}")
print(f"  {'gpu_count':30s} = {torch.cuda.device_count()}")
print("=" * 60)

# ── Tokenizer ────────────────────────────────────────────────────────────────
print("\n[1/5] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Dataset ──────────────────────────────────────────────────────────────────
print("\n[2/5] Loading mixed dataset ...")
dataset = MixedDataset(
    tokenizer=tokenizer,
    regular_data_dir=str(REGULAR_DATA_DIR),
    cot_data_dir=str(COT_DATA_DIR),
    regular_samples_per_class=args.regular_samples_per_class,
    eval_samples_per_class=args.eval_samples_per_class,
    max_length=args.max_length,
    boxed_loss_weight=args.boxed_loss_weight,
    reasoning_loss_weight=args.reasoning_loss_weight,
    seed=args.seed,
)

collator = WeightedDataCollator(
    tokenizer=tokenizer,
    model=None,  # will be set after model load
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

# ── Model + LoRA ─────────────────────────────────────────────────────────────
print("\n[3/5] Loading base model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="balanced",
    trust_remote_code=True,
)
model.is_parallelizable = True
model.model_parallel = True

print(f"[3/5] Loading LoRA adapter from {args.checkpoint_dir} ...")
model = PeftModel.from_pretrained(
    model,
    args.checkpoint_dir,
    is_trainable=True,
)
model.print_trainable_parameters()

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ── NEFTune ──────────────────────────────────────────────────────────────────
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

# ── Training args ────────────────────────────────────────────────────────────
print("\n[4/5] Configuring training ...")
os.makedirs(args.output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    num_train_epochs=args.num_epochs if args.max_steps <= 0 else 100,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,  # already enabled manually
    dataloader_pin_memory=True,
    dataloader_num_workers=0,
    logging_steps=5,
    save_steps=args.save_steps,
    save_total_limit=3,
    report_to="tensorboard",
    logging_dir=args.tensorboard_dir,
    remove_unused_columns=False,  # CRITICAL: keep loss_weights
)

# ── Eval callback ────────────────────────────────────────────────────────────
callbacks = []
eval_data = None

if args.eval_steps > 0:
    eval_data = load_eval_pool(
        str(REGULAR_DATA_DIR),
        train_samples_per_class=args.regular_samples_per_class,
        eval_samples_per_class=args.eval_samples_per_class,
        seed=args.seed,
    )
    print(f"[eval] {len(eval_data)} eval samples "
          f"(every {args.eval_steps} steps, max_new_tokens={args.eval_max_new_tokens}, "
          f"batch_size={args.eval_batch_size})")

    class EvalCallback(TrainerCallback):
        def __init__(self, eval_steps, eval_batch_size, eval_max_new_tokens):
            self.eval_steps = eval_steps
            self.eval_batch_size = eval_batch_size
            self.eval_max_new_tokens = eval_max_new_tokens
            self._tb_writer = None

        def _get_tb_writer(self, **kwargs):
            """Retrieve TensorBoard writer from the Trainer's callback list."""
            if self._tb_writer is None:
                # on_step_end receives model= but not trainer=.
                # Walk the callback list via the model's reference is not possible,
                # so we look for tb_writer among sibling callbacks passed in kwargs.
                # Fallback: search all callbacks if a 'callbacks' key exists.
                model = kwargs.get("model")
                if model is not None and hasattr(model, "_trainer"):
                    for cb in model._trainer.callback_handler.callbacks:
                        if hasattr(cb, "tb_writer") and cb.tb_writer is not None:
                            self._tb_writer = cb.tb_writer
                            break
            return self._tb_writer

        def set_tb_writer(self, writer):
            """Directly set the TensorBoard writer (called after trainer init)."""
            self._tb_writer = writer

        def on_step_end(self, _args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % self.eval_steps == 0:
                import time
                t0 = time.time()
                results = evaluate_samples(
                    model, tokenizer, eval_data,
                    max_new_tokens=self.eval_max_new_tokens,
                    batch_size=self.eval_batch_size,
                )
                elapsed = time.time() - t0

                tb_writer = self._tb_writer
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/accuracy", results["accuracy"],
                                         state.global_step)
                    for cat, cr in results["per_category"].items():
                        tb_writer.add_scalar(f"eval/accuracy_{cat}",
                                             cr["accuracy"], state.global_step)
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

    eval_cb = EvalCallback(args.eval_steps, args.eval_batch_size,
                           args.eval_max_new_tokens)
    callbacks.append(eval_cb)

# ── Train ────────────────────────────────────────────────────────────────────
print(f"\n[5/5] Starting mixed CoT training ...")

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    callbacks=callbacks,
)

# Inject TensorBoard writer into eval callback.
# TensorBoardCallback is auto-registered by Trainer when report_to="tensorboard".
# We grab its writer and pass it to our EvalCallback so eval metrics land in
# the same event file as training metrics.
if eval_data is not None:
    for cb in trainer.callback_handler.callbacks:
        if hasattr(cb, "_SummaryWriter") and hasattr(cb, "tb_writer"):
            # Force TensorBoardCallback to initialise its writer now
            if cb.tb_writer is None:
                if cb.logging_dir is None:
                    from transformers.integrations.integration_utils import default_logdir
                    cb.logging_dir = os.path.join(args.output_dir, default_logdir())
                cb._init_summary_writer(training_args)
            if cb.tb_writer is not None:
                eval_cb.set_tb_writer(cb.tb_writer)
                print(f"[tensorboard] eval callback connected (log_dir={cb.logging_dir})")
            break

if args.resume_from_checkpoint:
    print(f"[resume] Resuming from {args.resume_from_checkpoint}")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# ── Final eval ───────────────────────────────────────────────────────────────
if eval_data is not None:
    print("\n[final eval] Running final evaluation ...")
    final_results = evaluate_samples(
        model, tokenizer, eval_data,
        max_new_tokens=args.eval_max_new_tokens,
        batch_size=args.eval_batch_size,
    )

    tb_writer = eval_cb._tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar("eval/accuracy", final_results["accuracy"],
                             trainer.state.global_step)
        for cat, cr in final_results["per_category"].items():
            tb_writer.add_scalar(f"eval/accuracy_{cat}", cr["accuracy"],
                                 trainer.state.global_step)
        tb_writer.flush()

    print(f"[final eval] accuracy={final_results['accuracy']:.3f} "
          f"({final_results['num_correct']}/{final_results['num_total']})")
    for cat, cr in sorted(final_results["per_category"].items()):
        print(f"[final eval]   {cat:22s}: "
              f"{cr['correct']}/{cr['total']} = {cr['accuracy']:.3f}")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\n[done] Saving LoRA adapter to {args.output_dir} ...")
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("[done] Mixed CoT training complete.")
