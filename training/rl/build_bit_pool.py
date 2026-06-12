"""Build a bit_manipulation diversity-probe pool: 30 greedy-WRONG (where GRPO would
learn) + 20 greedy-CORRECT (to measure regression risk at T>0). Records:
{id, prompt, gt, category, greedy_correct}. Source: baseline eval shards (correctness)
+ train.csv (prompts)."""
from __future__ import annotations
import csv, json, glob
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
OUT = PROJECT / "training/rl/runs/hotpool/bit_diversity_pool.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

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
        if r["category"] != "bit_manipulation":
            continue
        pid = r["id"]
        if pid not in prompts:
            continue
        rec = {"id": pid, "prompt": prompts[pid], "gt": str(r["answer"]),
               "category": "bit_manipulation", "greedy_correct": bool(r["correct"])}
        (correct if r["correct"] else wrong).append(rec)

pool = wrong[:30] + correct[:20]
with open(OUT, "w") as f:
    for rec in pool:
        f.write(json.dumps(rec) + "\n")
print(f"[done] {len(pool)} bit prompts ({sum(not r['greedy_correct'] for r in pool)} wrong, "
      f"{sum(r['greedy_correct'] for r in pool)} correct) -> {OUT}")
