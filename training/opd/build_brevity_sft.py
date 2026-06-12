"""Build the SHORT-TRACE (brevity) SFT dataset for the 0.87-recipe reproduction.

Mechanism we are betting on (per Codex re-judge 2026-06-08): NOT blank-recovery
(our base is already format-clean: only 14 non-crypt recoverable problems), but
GLOBAL BREVITY GENERALIZATION as private length-insurance. We train generic short
reasoning + the CORRECT answer on EASY categories the baseline already solves, and
bet the "be brief and decisive" behavior generalizes to compress the long-correct
reasoning on bit/equation (1490 of our correct answers use >6000/7680 tokens), so
they stop overrunning on a slightly-longer private set.

Design decisions (Codex-vetted):
- Train ONLY on easy-category problems the 0.85 baseline got CORRECT -> the short
  trace's boxed answer is guaranteed correct (zero risk of teaching wrong answers).
- EXCLUDE cryptarithm_deduce/guess + equation_numeric_guess (hard/under-determined;
  short traces there would teach guessing). Keep numeral/unit_conversion/gravity/cipher.
- Generic ~15-30 token reasoning, rotated over a few templates (avoid memorizing one
  string), no answer leakage in the cot (dataset.py appends \\boxed{answer}).

Output: data/crd_brevity_sft.csv  (cols id,prompt,answer,generated_cot,type) for
training/dataset.py CoT mode.
"""
from __future__ import annotations
import csv, json, glob, sys, zlib
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
TRAIN_CSV = PROJECT / "resource/winning-solution/train.csv"      # id,prompt,answer (raw)
EVAL_SHARDS = sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl")))
OUT = PROJECT / "data/crd_brevity_sft.csv"

EASY = {"numeral", "unit_conversion", "gravity", "cipher"}
EXCLUDE = {"cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"}

TEMPLATES = [
    "I identify the transformation rule from the examples and apply it to the query.",
    "From the given examples I infer the pattern, then apply it to the query.",
    "I determine the rule shown by the examples and use it on the query value.",
    "Reading the examples, I extract the underlying rule and compute the result.",
    "I work out the rule from the examples and apply it to obtain the answer.",
]


def main():
    csv.field_size_limit(10**7)
    prompts = {}
    with open(TRAIN_CSV) as f:
        for r in csv.DictReader(f):
            prompts[r["id"]] = r["prompt"]
    print(f"[load] {len(prompts)} prompts from {TRAIN_CSV.name}")

    rows = []
    skipped_noprompt = 0
    from collections import Counter
    bycat = Counter()
    for shard in EVAL_SHARDS:
        for line in open(shard):
            if not line.strip():
                continue
            r = json.loads(line)
            cat = r["category"]
            if cat in EXCLUDE or cat not in EASY:
                continue
            if not r.get("correct"):
                continue
            pid = r["id"]
            p = prompts.get(pid)
            if p is None:
                skipped_noprompt += 1
                continue
            # deterministic template pick by id (stable across reruns)
            t = TEMPLATES[zlib.crc32(pid.encode()) % len(TEMPLATES)]
            rows.append({"id": pid, "prompt": p, "answer": str(r["answer"]),
                         "generated_cot": t, "type": cat})
            bycat[cat] += 1

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot", "type"])
        w.writeheader()
        w.writerows(rows)

    print(f"[done] wrote {len(rows)} rows -> {OUT}")
    print(f"[skipped] {skipped_noprompt} (no prompt found)")
    for c in sorted(bycat):
        print(f"   {c:20s} {bycat[c]}")


if __name__ == "__main__":
    main()
