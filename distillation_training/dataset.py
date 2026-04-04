"""
distillation_training/dataset.py — Mixed dataset: regular SFT + CoT + distillation samples.

Three data sources, all shuffled into a single training set:
  1. Regular SFT samples (7 categories from by_category CSVs)
       Target: </think>\n\n\boxed{answer}<|im_end|>
  2. Synthetic CoT samples (from cot_synthesis)
       Target: {reasoning}\n</think>\n\n\boxed{answer}<|im_end|>
  3. Distillation samples (from distilled teacher model outputs)
       Target: {reasoning}\n</think>\n\n\boxed{answer}<|im_end|>

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
    """
    user_content = prompt + BOXED_INSTRUCTION

    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    prefix_with_think_close = prefix_text + "</think>\n\n"
    boxed_suffix = f"\\boxed{{{answer}}}<|im_end|>\n"
    full_text = prefix_with_think_close + boxed_suffix

    prefix_ids = tokenizer(
        prefix_with_think_close, add_special_tokens=False, truncation=False
    ).input_ids
    return full_text, len(prefix_ids), boxed_suffix


def build_cot_conversation(tokenizer, prompt: str, cot_answer: str, answer: str):
    """Build a CoT/distillation sample with full reasoning chain.

    Prefix (masked):  ... <think>\n
    Target (supervised):  {cot_reasoning}\n</think>\n\n\boxed{answer}<|im_end|>\n
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
    """Loads regular SFT + CoT + distillation data, shuffles, and pre-tokenizes.

    Args:
        tokenizer:                 The tokenizer.
        regular_data_dir:          Path to by_category CSVs.
        cot_data_dir:              Path to cot/ directory with task subdirs.
        distill_data_dir:          Path to distillation/ directory with task subdirs.
        regular_samples_per_class: Default number of regular samples per category.
        category_samples:          Per-category overrides, e.g. {"gravity": 200}.
        exclude_categories:        Set of category names to skip entirely.
        eval_samples_per_class:    Number of eval samples per category.
        max_length:                Max token length for truncation.
        boxed_loss_weight:         Loss weight for \\boxed{} tokens in CoT/distill samples.
        reasoning_loss_weight:     Loss weight for reasoning tokens.
        seed:                      Random seed for reproducible splits.
    """

    def __init__(
        self,
        tokenizer,
        regular_data_dir: str,
        cot_data_dir: str,
        distill_data_dir: str,
        regular_samples_per_class: int = 500,
        category_samples: dict | None = None,
        exclude_categories: set | None = None,
        eval_samples_per_class: int = 5,
        max_length: int = 2048,
        boxed_loss_weight: float = 1.0,
        reasoning_loss_weight: float = 1.0,
        seed: int = 42,
    ):
        rng = random.Random(seed)
        self.eval_pool = {}

        # ── 1. Load distillation data (indexed by prompt for dedup) ─────
        distill_by_prompt = {}  # prompt -> (cot_answer, answer, category)
        distill_root = Path(distill_data_dir)
        print(f"[dataset] Loading distillation data from {distill_data_dir} ...")
        distill_counts = {}
        for task_dir in sorted(distill_root.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            jsonl_file = task_dir / "distilled.jsonl"
            if not jsonl_file.exists():
                continue
            count = 0
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    distill_by_prompt[obj["prompt"]] = (
                        obj["cot_answer"], obj["answer"], task
                    )
                    count += 1
            distill_counts[task] = count
            print(f"[dataset]   Distill {task:19s}: {count:4d} samples")
        print(f"[dataset] Distillation: {len(distill_by_prompt)} samples "
              f"({', '.join(f'{t}: {c}' for t, c in sorted(distill_counts.items()))})")

        # ── 2. Load regular data (distillation prompts prioritized) ────
        #
        # Strategy: for each category, put ALL distillation-matched prompts
        # into the train split first, then fill remaining slots with regular
        # prompts.  Eval pool is drawn only from non-distillation leftovers
        # so there is zero overlap with distillation training data.
        regular_raw = []   # non-overlapping regular samples
        upgraded_raw = []  # regular prompts upgraded to distillation version
        exclude_categories = exclude_categories or set()
        if exclude_categories:
            print(f"[dataset] Excluding categories: {sorted(exclude_categories)}")
        print(f"[dataset] Loading regular data from {regular_data_dir} ...")
        for csv_file in sorted(Path(regular_data_dir).glob("*.csv")):
            category = csv_file.stem
            if category in exclude_categories:
                print(f"[dataset]   {category:22s}: SKIPPED (excluded)")
                continue
            rows = []
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append((row["prompt"], row["answer"]))
            rng.shuffle(rows)
            n = category_samples.get(category, regular_samples_per_class) if category_samples else regular_samples_per_class

            # Separate: prompts with distillation match vs without
            has_distill = [(p, a) for p, a in rows if p in distill_by_prompt]
            no_distill = [(p, a) for p, a in rows if p not in distill_by_prompt]

            # Priority: all distillation-matched first, then fill with regular
            n_distill = min(len(has_distill), n)
            n_regular_fill = n - n_distill
            chosen_distill = has_distill[:n_distill]
            chosen_regular = no_distill[:n_regular_fill]
            remaining = no_distill[n_regular_fill:]  # eval candidates (clean)

            self.eval_pool[category] = remaining[:eval_samples_per_class]

            for p, a in chosen_distill:
                cot_a, dist_a, dist_task = distill_by_prompt[p]
                upgraded_raw.append((p, cot_a, dist_a, category))

            for p, a in chosen_regular:
                regular_raw.append((p, a, category))

            print(f"[dataset]   {category:22s}: {len(chosen_regular):4d} regular + "
                  f"{len(chosen_distill):4d} upgraded "
                  f"({len(has_distill)} distill avail) / "
                  f"{min(len(remaining), eval_samples_per_class):4d} eval / "
                  f"{len(rows):5d} total")
        print(f"[dataset] Regular: {len(regular_raw)} samples "
              f"(+ {len(upgraded_raw)} upgraded to distillation)")

        # ── 3. Load CoT data ─────────────────────────────────────────────
        cot_raw = []
        cot_root = Path(cot_data_dir)
        print(f"[dataset] Loading CoT data from {cot_data_dir} ...")
        cot_counts = {}
        for task_dir in sorted(cot_root.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            if task in exclude_categories:
                print(f"[dataset]   CoT {task:22s}: SKIPPED (excluded)")
                continue
            count = 0
            for level_file in sorted(task_dir.glob("*.jsonl")):
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
            cot_counts[task] = count
            print(f"[dataset]   CoT {task:22s}: {count:4d} samples")
        print(f"[dataset] CoT: {len(cot_raw)} samples "
              f"({', '.join(f'{t}: {c}' for t, c in sorted(cot_counts.items()))})")

        # ── 4. Pre-tokenize ──────────────────────────────────────────────
        print(f"[dataset] Pre-tokenizing (max_length={max_length}) ...")

        all_items = []

        # Regular samples (no distillation match — only supervise boxed answer)
        reg_truncated = 0
        for prompt, answer, category in regular_raw:
            full_text, prefix_len, boxed_suffix = build_regular_conversation(
                tokenizer, prompt, answer
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
                reg_truncated += 1
            all_items.append(item["sample"])

        # Upgraded samples (regular prompts with distillation reasoning)
        upgraded_truncated = 0
        for prompt, cot_answer, answer, category in upgraded_raw:
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
                upgraded_truncated += 1
            all_items.append(item["sample"])

        # CoT synthesis samples
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
        print(f"[dataset]   Regular:      {len(regular_raw)} ({reg_truncated} truncated)")
        print(f"[dataset]   Upgraded:     {len(upgraded_raw)} ({upgraded_truncated} truncated)")
        print(f"[dataset]   CoT:          {len(cot_raw)} ({cot_truncated} truncated)")

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

        Weights are normalised so sum(loss_weights) = 1.0 per sample.
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

        labels = list(input_ids)
        mask_len = min(prefix_len, seq_len)
        for i in range(mask_len):
            labels[i] = -100

        if all(l == -100 for l in labels):
            labels[-1] = input_ids[-1]

        loss_weights = [0.0] * seq_len
        is_boxed = [False] * seq_len
        for i in range(seq_len):
            if labels[i] != -100:
                loss_weights[i] = reasoning_weight

        if boxed_weight != reasoning_weight:
            boxed_ids = tokenizer(
                boxed_suffix, add_special_tokens=False, truncation=False
            ).input_ids
            boxed_len = len(boxed_ids)

            if seq_len >= boxed_len and input_ids[-boxed_len:] == boxed_ids:
                for i in range(seq_len - boxed_len, seq_len):
                    if labels[i] != -100:
                        loss_weights[i] = boxed_weight
                        is_boxed[i] = True

        weight_sum = sum(loss_weights)
        if weight_sum > 0:
            loss_weights = [w / weight_sum for w in loss_weights]

        return {
            "sample": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_weights": loss_weights,
                "is_boxed": is_boxed,
            },
            "truncated": truncated,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Eval pool loader ─────────────────────────────────────────────────────────

def load_eval_pool(data_dir: str, train_samples_per_class: int = 500,
                   eval_samples_per_class: int = 5, seed: int = 42,
                   distill_data_dir: str | None = None):
    """Load eval samples from by_category CSVs, skipping the training split.

    Uses the same distillation-priority logic as MixedDataset to ensure the
    eval split is consistent: distillation prompts are counted as training
    samples, and eval is drawn only from non-distillation leftovers.
    """
    # Load distillation prompts if available
    distill_prompts = set()
    if distill_data_dir:
        distill_root = Path(distill_data_dir)
        for task_dir in sorted(distill_root.iterdir()):
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
    for csv_file in sorted(Path(data_dir).glob("*.csv")):
        category = csv_file.stem
        rows = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["prompt"], row["answer"]))
        rng.shuffle(rows)

        # Same priority logic: distillation prompts fill train split first
        has_distill = [(p, a) for p, a in rows if p in distill_prompts]
        no_distill = [(p, a) for p, a in rows if p not in distill_prompts]
        n_distill = min(len(has_distill), train_samples_per_class)
        n_regular_fill = train_samples_per_class - n_distill
        remaining = no_distill[n_regular_fill:]

        chosen = remaining[:eval_samples_per_class]
        eval_data.extend([(p, a, category) for p, a in chosen])
    return eval_data
