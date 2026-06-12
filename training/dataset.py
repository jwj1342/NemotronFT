"""
v6_training/dataset_v6.py — Load CoT CSV and build chat-templated training samples.

Each sample is (input_ids, labels, attention_mask). Labels mask the user/prompt
portion with -100 so loss is only computed on the assistant response (CoT + boxed).

The chat template uses enable_thinking=True so the assistant section opens with
`<think>\\n` and we close it with `</think>\\n\\boxed{answer}` — matching the
exact format the grader expects (Huikang's tinker-submission notebook uses the
same template at inference).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from torch.utils.data import Dataset


PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Remove any \boxed{...} inside the decoded CoT so we control where the final
# boxed answer appears. Matches the cleaning in the Unsloth 0.81 notebook.
_BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


class HuikangCoTDataset(Dataset):
    """Builds tokenized chat-template samples from a CoT CSV.

    The CSV must have columns: id, prompt, answer, generated_cot, type.
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        max_length: int = 8192,
        category_filter: set[str] | None = None,
        mask_prompt: bool = True,
        plain_answer: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        # plain_answer: kienngx 0.86 recipe — assistant = raw answer (NO CoT, NO
        # <think>), enable_thinking=False at train time. The base model's native
        # reasoning (enable_thinking=True at inference) does the solving; SFT only
        # aligns the output-format/task-distribution. Reads `answer`, ignores
        # `generated_cot`.
        self.plain_answer = plain_answer

        self.examples: list[dict] = []
        self.categories: list[str] = []
        skipped_empty = 0
        skipped_long = 0
        skipped_cat = 0
        cat_counts: dict[str, int] = {}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if category_filter and row.get("type") not in category_filter:
                    skipped_cat += 1
                    continue

                user_content = row["prompt"] + PROMPT_SUFFIX
                if self.plain_answer:
                    # kienngx recipe: assistant = raw final answer, no CoT wrapper.
                    assistant_content = row["answer"]
                    if not assistant_content.strip():
                        skipped_empty += 1
                        continue
                    input_ids, labels = self._encode(
                        user_content, assistant_content, enable_thinking=False)
                else:
                    cot = (row.get("generated_cot") or "").strip()
                    if len(cot) < 5:
                        skipped_empty += 1
                        continue
                    cot_cleaned = _BOXED_RE.sub("", cot).rstrip()
                    assistant_content = (
                        "<think>\n" + cot_cleaned
                        + "\n</think>\n"
                        + "\\boxed{" + row["answer"] + "}"
                    )
                    input_ids, labels = self._encode(user_content, assistant_content)
                if len(input_ids) > max_length:
                    skipped_long += 1
                    continue

                self.examples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(input_ids),
                })
                cat = row.get("type", "plain")
                self.categories.append(cat)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        self.cat_counts = cat_counts
        self.skipped_empty = skipped_empty
        self.skipped_long = skipped_long
        self.skipped_cat = skipped_cat

    def _encode(self, user_content: str, assistant_content: str,
                enable_thinking: bool = True) -> tuple[list[int], list[int]]:
        """Tokenize via chat template; produce labels with -100 for prompt portion."""
        # Token sequence with assistant content (full training sample)
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

        labels = list(full_ids)
        if self.mask_prompt:
            # Token sequence of user-only message with generation prompt — this
            # matches the inference-time prefix. The model only sees and trains
            # on tokens beyond this prefix.
            prefix_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False).input_ids
            n = len(prefix_ids)
            # Sanity: full_ids should start with prefix_ids
            for i in range(min(n, len(full_ids))):
                labels[i] = -100
        return full_ids, labels

    def summary(self) -> str:
        n = len(self.examples)
        total_tokens = sum(len(e["input_ids"]) for e in self.examples)
        supervised = sum(sum(1 for lb in e["labels"] if lb != -100) for e in self.examples)
        lines = [
            f"HuikangCoTDataset: {n} examples, {total_tokens:,} total tokens, "
            f"{supervised:,} supervised tokens, "
            f"skipped_empty={self.skipped_empty}, skipped_long={self.skipped_long}, "
            f"skipped_cat={self.skipped_cat}"
        ]
        for cat in sorted(self.cat_counts):
            lines.append(f"  {cat:30s} {self.cat_counts[cat]}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return dict(self.examples[idx])


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    tok = AutoTokenizer.from_pretrained(PROJECT_ROOT / "model", trust_remote_code=True)
    ds = HuikangCoTDataset(
        csv_path=PROJECT_ROOT / "v6_training" / "train_with_cot.csv",
        tokenizer=tok,
        max_length=8192,
    )
    print(ds.summary())
    print()
    sample = ds[0]
    print(f"Sample 0: input_ids len={len(sample['input_ids'])}, "
          f"supervised tokens={sum(1 for lb in sample['labels'] if lb != -100)}")
    # Show first non-masked label start
    first_unmasked = next(i for i, lb in enumerate(sample["labels"]) if lb != -100)
    print(f"First supervised token at index {first_unmasked}")
    print(f"Prefix decoded: {repr(tok.decode(sample['input_ids'][:first_unmasked])[-80:])}")
    print(f"First supervised tokens: {repr(tok.decode(sample['input_ids'][first_unmasked:first_unmasked+30]))}")
