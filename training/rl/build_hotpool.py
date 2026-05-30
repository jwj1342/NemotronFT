"""Build hot-pool for RL: prompts where the baseline (Huikang) gets the
answer right at T=0 greedy.  Restricted to the three RL-target categories
(cryptarithm_deduce, cryptarithm_guess, equation_numeric_guess).

Output:
  runs/hotpool/baseline_greedy_all.jsonl   — every prompt + greedy_correct bool
  runs/hotpool/tier1_hot.jsonl             — only correct ones (the actual hot pool)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
sys.path.insert(0, str(PROJECT / "eval"))
sys.path.insert(0, str(PROJECT / "training" / "rl"))
from category import detect_category  # type: ignore  # noqa: E402
from extract import extract_final_answer, verify, BOXED_INSTRUCTION  # type: ignore  # noqa: E402
from utils import write_jsonl  # type: ignore  # noqa: E402

MODEL_PATH = str(PROJECT / "model")
TARGET_CATS = {"cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv",
                    default=str(PROJECT / "resource" / "winning-solution" / "train.csv"))
    ap.add_argument("--adapter_dir",
                    default=str(PROJECT / "output" / "huikang-baseline-converted"))
    ap.add_argument("--out_dir",
                    default=str(PROJECT / "training" / "rl" / "runs" / "hotpool"))
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "baseline_greedy_all.jsonl"
    hot_path = out_dir / "tier1_hot.jsonl"

    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(args.train_csv) as f:
        for r in csv.DictReader(f):
            cat = detect_category(r["prompt"])
            if cat not in TARGET_CATS:
                continue
            rows.append({"id": r["id"], "prompt": r["prompt"], "gt": str(r["answer"]),
                         "category": cat})
    print(f"target prompts: {len(rows)}")
    by_cat = Counter(r["category"] for r in rows)
    for c, n in by_cat.items():
        print(f"  {c}: {n}")

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[vllm] loading {MODEL_PATH} ...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_num_seqs=16,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"[vllm] loaded {time.time()-t0:.1f}s")

    prompts = []
    for r in rows:
        user = r["prompt"] + BOXED_INSTRUCTION
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
        except Exception:
            text = user
        prompts.append(text)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    lora = LoRARequest("adapter", 1, str(args.adapter_dir))

    print(f"[generate] {len(prompts)} prompts T=0 greedy ...")
    t0 = time.time()
    outs = llm.generate(prompts, sampling, lora_request=lora)
    print(f"[generate] {time.time()-t0:.1f}s ({(time.time()-t0)/len(prompts):.2f}s/prompt)")

    all_rows = []
    hot_rows = []
    per_cat_correct: dict = Counter()
    per_cat_total: dict = Counter()
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        pred = extract_final_answer(text)
        ok = bool(verify(r["gt"], pred))
        rec_all = {"id": r["id"], "category": r["category"], "prompt": r["prompt"],
                   "gt": r["gt"], "predicted": pred, "greedy_correct": ok}
        all_rows.append(rec_all)
        per_cat_total[r["category"]] += 1
        if ok:
            per_cat_correct[r["category"]] += 1
            hot_rows.append({"id": r["id"], "prompt": r["prompt"], "gt": r["gt"],
                             "category": r["category"]})

    write_jsonl(all_path, all_rows)
    write_jsonl(hot_path, hot_rows)

    summary = {"per_category": {}, "total_hot": len(hot_rows), "total_all": len(all_rows)}
    print("\n=== HOT POOL SUMMARY ===")
    for c in sorted(per_cat_total):
        cor, tot = per_cat_correct[c], per_cat_total[c]
        summary["per_category"][c] = {"correct": cor, "total": tot,
                                      "acc": cor / tot if tot else 0.0}
        print(f"  {c:30s} {cor:4d}/{tot:4d}  ({cor/tot:.2%})")
    print(f"\nTier-1 hot pool: {len(hot_rows)} prompts -> {hot_path}")
    print(f"All baseline data: {len(all_rows)} prompts -> {all_path}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
