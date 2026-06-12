"""Build the bit GRPO validation split: a TRAIN pool (failure-weighted, where GRPO
learns) + a disjoint HOLDOUT (to judge net gain = recover - regress on UNSEEN bit
problems — the only valid salvageability test). Records: {id,prompt,gt,category,
greedy_correct}."""
from __future__ import annotations
import csv, json, glob
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
OUTDIR = PROJECT / "training/rl/runs/hotpool"
OUTDIR.mkdir(parents=True, exist_ok=True)
DIV_POOL_IDS = set()
# exclude the 50 ids already used in the diversity probe (keep splits clean)
dp = OUTDIR / "bit_diversity_pool.jsonl"
if dp.exists():
    for line in open(dp):
        if line.strip():
            DIV_POOL_IDS.add(json.loads(line)["id"])

csv.field_size_limit(10**7)
prompts = {}
with open(PROJECT / "resource/winning-solution/train.csv") as f:
    for r in csv.DictReader(f):
        prompts[r["id"]] = r["prompt"]

wrong, correct = [], []
for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
    for line in open(shard):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["category"] != "bit_manipulation" or r["id"] in DIV_POOL_IDS or r["id"] not in prompts:
            continue
        rec = {"id": r["id"], "prompt": prompts[r["id"]], "gt": str(r["answer"]),
               "category": "bit_manipulation", "greedy_correct": bool(r["correct"])}
        (correct if r["correct"] else wrong).append(rec)

# TRAIN: 90 failures + 60 correct  |  HOLDOUT: next 50 failures + 50 correct (disjoint)
train = wrong[:90] + correct[:60]
holdout = wrong[90:140] + correct[60:110]
for name, rows in [("bit_grpo_train.jsonl", train), ("bit_grpo_holdout.jsonl", holdout)]:
    with open(OUTDIR / name, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    nf = sum(not r["greedy_correct"] for r in rows)
    print(f"{name}: {len(rows)} ({nf} fail, {len(rows)-nf} correct)")
print(f"(excluded {len(DIV_POOL_IDS)} diversity-probe ids; bit pool avail: {len(wrong)} fail, {len(correct)} correct)")
