"""
v6_training/build_cot_csv.py — Decode Huikang's pre-tokenized corpus into a text CSV.

Output: id, prompt, answer, generated_cot, type
Each row matches an entry in resource/winning-solution/train.csv (9500 rows),
with `generated_cot` being the decoded text from the corpus unmasked tokens
(with the closing </think>\\boxed{}<|im_end|> stripped).

This CSV is the input format that the proven Unsloth 0.81 recipe uses.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE = PROJECT_ROOT / "resource" / "winning-solution"
OUTPUT_CSV = PROJECT_ROOT / "training" / "train_with_cot.csv"
MODEL_PATH = PROJECT_ROOT / "model"


# Strip the closing chat-template tail: optional whitespace, optional </think>,
# optional \boxed{...}, then <|im_end|>. We do NOT strip embedded \boxed{} inside
# the CoT here — the dataset class will remove them and re-add a clean final one.
TAIL_RE = re.compile(
    r"\s*(?:</think>\s*)?(?:\\?boxed\{[^}]*\}\s*)?<\|im_end\|>\s*$"
)


def decode_segments(seg_path: Path, tokenizer) -> tuple[str, str]:
    """Return (decoded_masked_prompt, decoded_unmasked_response_with_tail)."""
    masked_tokens: list[int] = []
    unmasked_tokens: list[int] = []
    with open(seg_path) as f:
        for line in f:
            seg = json.loads(line)
            if seg["type"] == "masked":
                masked_tokens.extend(seg["tokens"])
            else:
                unmasked_tokens.extend(seg["tokens"])
    return tokenizer.decode(masked_tokens), tokenizer.decode(unmasked_tokens)


def main() -> None:
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading corpus index ...")
    corpus_entries: dict[str, dict] = {}
    with open(RESOURCE / "corpus.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e.get("included"):
                corpus_entries[e["problem_id"]] = e
    print(f"  included entries: {len(corpus_entries)}")

    print("Loading train.csv ...")
    with open(RESOURCE / "train.csv", newline="") as f:
        train_rows = list(csv.DictReader(f))
    print(f"  rows: {len(train_rows)}")

    print("Decoding CoT for each problem ...")
    out_rows = []
    cat_counts: Counter[str] = Counter()
    missing = 0
    long_resp = 0
    for i, row in enumerate(train_rows):
        pid = row["id"]
        if pid not in corpus_entries:
            missing += 1
            continue
        entry = corpus_entries[pid]
        seg_path = RESOURCE / "corpus" / pid / entry["segment"]
        if not seg_path.exists():
            missing += 1
            continue
        _, full_resp = decode_segments(seg_path, tokenizer)
        cot = TAIL_RE.sub("", full_resp)
        out_rows.append({
            "id": pid,
            "prompt": row["prompt"],
            "answer": row["answer"],
            "generated_cot": cot,
            "type": entry["category"],
        })
        cat_counts[entry["category"]] += 1
        if entry["token_count"] > 7680:
            long_resp += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(train_rows)} processed")

    print()
    print(f"Total output rows: {len(out_rows)}")
    print(f"Missing corpus entries: {missing}")
    print(f"Samples with total_tokens > 7680: {long_resp}")
    print()
    print("Category distribution:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s} {n}")

    print()
    print(f"Writing {OUTPUT_CSV} ...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot", "type"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows")


if __name__ == "__main__":
    main()
