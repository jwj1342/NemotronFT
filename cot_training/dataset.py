"""
cot_training/dataset.py — Mixed dataset: regular SFT + CoT samples with per-token loss weights.

Regular samples:  train on  </think>\n\n\boxed{answer}<|im_end|>
CoT samples:      train on  {reasoning}\n</think>\n\n\boxed{answer}<|im_end|>

Each sample carries a `loss_weights` vector so the trainer can emphasise
\boxed{} tokens over reasoning tokens.
"""

import csv
import json
import random
from pathlib import Path

from torch.utils.data import Dataset

# ── Constants ────────────────────────────────────────────────────────────────

BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


# ── Conversation builders ────────────────────────────────────────────────────

def build_regular_conversation(tokenizer, prompt: str, answer: str):
    """Build a regular sample with enable_thinking=True.

    Prefix (masked):  ... <think>\n</think>\n\n
    Target (supervised):  \boxed{answer}<|im_end|>\n

    We don't have CoT ground truth for these categories, so the entire
    thinking region (including </think>) is masked.  The model is free
    to produce whatever reasoning it wants at inference time — we only
    supervise the final boxed answer.

    Returns (full_text, prefix_len, boxed_suffix_text).
    """
    user_content = prompt + BOXED_INSTRUCTION

    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Extend prefix to cover </think>\n\n — all masked, no supervision
    prefix_with_think_close = prefix_text + "</think>\n\n"
    boxed_suffix = f"\\boxed{{{answer}}}<|im_end|>\n"
    full_text = prefix_with_think_close + boxed_suffix

    prefix_ids = tokenizer(
        prefix_with_think_close, add_special_tokens=False, truncation=False
    ).input_ids
    return full_text, len(prefix_ids), boxed_suffix


def build_cot_conversation(tokenizer, prompt: str, cot_answer: str, answer: str):
    """Build a CoT sample with full reasoning chain.

    Prefix (masked):  ... <think>\n
    Target (supervised):  {cot_reasoning}\n</think>\n\n\boxed{answer}<|im_end|>\n

    Returns (full_text, prefix_len, boxed_suffix_text).
    """
    user_content = prompt + BOXED_INSTRUCTION

    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    boxed_suffix = f"\\boxed{{{answer}}}<|im_end|>\n"
    full_text = (
        prefix_text
        + cot_answer
        + "\n</think>\n\n"
        + boxed_suffix
    )

    prefix_ids = tokenizer(
        prefix_text, add_special_tokens=False, truncation=False
    ).input_ids
    return full_text, len(prefix_ids), boxed_suffix


# ── Dataset ──────────────────────────────────────────────────────────────────

class MixedDataset(Dataset):
    """Loads regular SFT + CoT data, shuffles, and pre-tokenizes with loss weights.

    Args:
        tokenizer:                 The tokenizer.
        regular_data_dir:          Path to by_category CSVs.
        cot_data_dir:              Path to cot/ directory with task subdirs.
        regular_samples_per_class: Number of regular samples per category.
        eval_samples_per_class:    Number of eval samples per category.
        max_length:                Max token length for truncation.
        boxed_loss_weight:         Loss weight for \\boxed{} tokens in CoT samples.
        reasoning_loss_weight:     Loss weight for reasoning tokens in CoT samples.
        seed:                      Random seed for reproducible splits.
    """

    def __init__(
        self,
        tokenizer,
        regular_data_dir: str,
        cot_data_dir: str,
        regular_samples_per_class: int = 500,
        eval_samples_per_class: int = 5,
        max_length: int = 2048,
        boxed_loss_weight: float = 5.0,
        reasoning_loss_weight: float = 1.0,
        seed: int = 42,
    ):
        rng = random.Random(seed)
        self.eval_pool = {}  # {category: [(prompt, answer), ...]}

        # ── 1. Load regular data ─────────────────────────────────────────
        regular_raw = []  # [(prompt, answer, category)]
        print(f"[dataset] Loading regular data from {regular_data_dir} ...")
        for csv_file in sorted(Path(regular_data_dir).glob("*.csv")):
            category = csv_file.stem
            rows = []
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append((row["prompt"], row["answer"]))
            rng.shuffle(rows)
            chosen = rows[:regular_samples_per_class]
            remaining = rows[regular_samples_per_class:]
            self.eval_pool[category] = remaining[:eval_samples_per_class]
            regular_raw.extend([(p, a, category) for p, a in chosen])
            print(f"[dataset]   {category:22s}: {len(chosen):4d} train / "
                  f"{min(len(remaining), eval_samples_per_class):4d} eval / "
                  f"{len(rows):5d} total")
        print(f"[dataset] Regular: {len(regular_raw)} samples")

        # ── 2. Load CoT data ─────────────────────────────────────────────
        cot_raw = []  # [(prompt, cot_answer, answer, task)]
        cot_root = Path(cot_data_dir)
        print(f"[dataset] Loading CoT data from {cot_data_dir} ...")
        task_counts = {}
        for task_dir in sorted(cot_root.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            count = 0
            for level_file in sorted(task_dir.glob("level_*.jsonl")):
                with open(level_file, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        cot_raw.append((
                            obj["prompt"],
                            obj["cot_answer"],
                            obj["answer"],
                            task,
                        ))
                        count += 1
            task_counts[task] = count
            print(f"[dataset]   CoT {task:22s}: {count:4d} samples")
        print(f"[dataset] CoT: {len(cot_raw)} samples "
              f"({', '.join(f'{t}: {c}' for t, c in sorted(task_counts.items()))})")

        # ── 3. Pre-tokenize ──────────────────────────────────────────────
        print(f"[dataset] Pre-tokenizing (max_length={max_length}, "
              f"boxed_weight={boxed_loss_weight}, reasoning_weight={reasoning_loss_weight}) ...")

        all_items = []  # will be shuffled

        # Regular samples
        reg_truncated = 0
        for prompt, answer, category in regular_raw:
            full_text, prefix_len, boxed_suffix = build_regular_conversation(
                tokenizer, prompt, answer
            )
            item = self._tokenize_with_weights(
                tokenizer, full_text, prefix_len, boxed_suffix,
                boxed_weight=1.0,       # no special weighting for regular
                reasoning_weight=1.0,
                max_length=max_length,
            )
            if item is None:
                continue
            if item["truncated"]:
                reg_truncated += 1
            all_items.append(item["sample"])

        # CoT samples
        cot_truncated = 0
        for prompt, cot_answer, answer, task in cot_raw:
            full_text, prefix_len, boxed_suffix = build_cot_conversation(
                tokenizer, prompt, cot_answer, answer
            )
            item = self._tokenize_with_weights(
                tokenizer, full_text, prefix_len, boxed_suffix,
                boxed_weight=boxed_loss_weight,
                reasoning_weight=reasoning_loss_weight,
                max_length=max_length,
            )
            if item is None:
                continue
            if item["truncated"]:
                cot_truncated += 1
            all_items.append(item["sample"])

        rng.shuffle(all_items)
        self.samples = all_items

        print(f"[dataset] Total: {len(self.samples)} samples (shuffled)")
        print(f"[dataset]   Regular: {len(regular_raw)} ({reg_truncated} truncated)")
        print(f"[dataset]   CoT:     {len(cot_raw)} ({cot_truncated} truncated)")

    @staticmethod
    def _tokenize_with_weights(
        tokenizer,
        full_text: str,
        prefix_len: int,
        boxed_suffix: str,
        boxed_weight: float,
        reasoning_weight: float,
        max_length: int,
    ):
        """Tokenize a sample and compute per-token loss weights.

        The weights are normalised so that every sample (regardless of length)
        contributes equally to the loss.  Within each sample, the relative
        ratio between reasoning tokens and boxed tokens is preserved.

        Without normalisation, a 1200-token CoT sample would dominate a
        11-token regular sample by ~100×, drowning out the answer-supervision
        signal from the regular data.

        Returns dict with 'sample' and 'truncated', or None if fully masked.
        """
        encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        seq_len = len(input_ids)
        truncated = (seq_len == max_length)

        # Build labels: mask prefix with -100
        labels = list(input_ids)
        mask_len = min(prefix_len, seq_len)
        for i in range(mask_len):
            labels[i] = -100

        # Safety: if everything is masked, keep last token
        if all(l == -100 for l in labels):
            labels[-1] = input_ids[-1]

        # Build raw loss_weights: default reasoning_weight for supervised tokens
        loss_weights = [0.0] * seq_len
        for i in range(seq_len):
            if labels[i] != -100:
                loss_weights[i] = reasoning_weight

        # Find \boxed{...}<|im_end|>\n suffix and apply boxed_weight
        if boxed_weight != reasoning_weight:
            boxed_ids = tokenizer(
                boxed_suffix, add_special_tokens=False, truncation=False
            ).input_ids
            boxed_len = len(boxed_ids)

            # Check if the tail of input_ids matches boxed_ids (not truncated away)
            if seq_len >= boxed_len and input_ids[-boxed_len:] == boxed_ids:
                for i in range(seq_len - boxed_len, seq_len):
                    if labels[i] != -100:
                        loss_weights[i] = boxed_weight

        # ── Sample-level normalisation ───────────────────────────────────
        # Scale weights so that sum(loss_weights) = 1.0 for every sample.
        # This ensures each sample contributes equally to the batch loss,
        # regardless of sequence length.
        weight_sum = sum(loss_weights)
        if weight_sum > 0:
            loss_weights = [w / weight_sum for w in loss_weights]

        return {
            "sample": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_weights": loss_weights,
            },
            "truncated": truncated,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Eval pool loader (standalone, for use outside MixedDataset) ──────────────

def load_eval_pool(data_dir: str, train_samples_per_class: int = 500,
                   eval_samples_per_class: int = 5, seed: int = 42):
    """Load eval samples from by_category CSVs, skipping the training split."""
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
