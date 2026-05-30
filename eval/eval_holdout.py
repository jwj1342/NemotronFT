"""Tiny held-out eval for the canary — generate on N cryptarithm puzzles NOT in
the canary training CSV, compare baseline vs canary adapter, report
per-checkpoint accuracy.

Usage:
  python eval/eval_holdout.py --adapter <path> [--n 30] [--out runs/eval.json]

If `--adapter` is omitted, evaluates the Huikang baseline (control).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
sys.path.insert(0, str(PROJECT / "eval"))
from extract import extract_final_answer, verify, BOXED_INSTRUCTION  # type: ignore
from category import detect_category  # type: ignore

CANARY_CSV = PROJECT / "training" / "opd" / "runs" / "canary" / "train.csv"
PROBLEMS_DIR = PROJECT / "resource" / "winning-solution" / "problems"
PROBLEMS_FILE = PROJECT / "resource" / "winning-solution" / "problems.jsonl"

MODEL_PATH = str(PROJECT / "model")


def load_holdout(n: int, seed: int = 11):
    import random
    canary_ids = set()
    csv.field_size_limit(sys.maxsize)
    with open(CANARY_CSV) as f:
        for r in csv.DictReader(f):
            canary_ids.add(r["id"])
    # collect cryptarithm puzzles outside canary; load full prompt + answer
    candidates = []
    with open(PROBLEMS_FILE) as f:
        for line in f:
            p = json.loads(line)
            if not p.get("category", "").startswith("cryptarithm"):
                continue
            if p["id"] in canary_ids:
                continue
            try:
                with open(PROBLEMS_DIR / f"{p['id']}.jsonl") as pf:
                    d = json.loads(pf.readline())
            except Exception:
                continue
            candidates.append({
                "id": p["id"],
                "prompt": d["prompt"],
                "answer": str(d["answer"]),
                "category": p["category"],
            })
    random.Random(seed).shuffle(candidates)
    return candidates[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None,
                    help="LoRA adapter dir; omit for base-only eval")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=str,
                    default=str(PROJECT / "training" / "opd" / "runs" / "canary" / "eval.jsonl"))
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    rows = load_holdout(args.n, seed=args.seed)
    print(f"holdout puzzles: {len(rows)}")

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"loading model {MODEL_PATH} ...")
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
        max_num_seqs=8,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"loaded in {time.time()-t0:.1f}s")

    prompts = []
    for r in rows:
        user = r["prompt"] + BOXED_INSTRUCTION
        try:
            p = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            p = user
        prompts.append(p)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    lora = LoRARequest("adapter", 1, str(args.adapter)) if args.adapter else None

    print(f"generating {len(prompts)} prompts ...")
    t0 = time.time()
    outs = llm.generate(prompts, sampling, lora_request=lora) if lora else llm.generate(prompts, sampling)
    dt = time.time() - t0
    print(f"done {dt:.1f}s ({dt/len(prompts):.2f}s/sample)")

    correct = 0
    per_cat = {"cryptarithm_deduce": [0, 0], "cryptarithm_guess": [0, 0]}
    with open(args.out, "w") as f:
        for i, o in enumerate(outs):
            r = rows[i]
            text = o.outputs[0].text
            pred = extract_final_answer(text)
            ok = verify(r["answer"], pred)
            correct += int(ok)
            cat = r["category"]
            per_cat.setdefault(cat, [0, 0])
            per_cat[cat][0] += int(ok)
            per_cat[cat][1] += 1
            f.write(json.dumps({"id": r["id"], "category": cat, "answer": r["answer"],
                                "predicted": pred, "correct": ok}) + "\n")
    summary = {
        "adapter": str(args.adapter) if args.adapter else "BASELINE",
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "per_category": {k: {"correct": v[0], "total": v[1],
                             "acc": v[0]/v[1] if v[1] else 0} for k, v in per_cat.items()},
    }
    print(json.dumps(summary, indent=2))
    sum_path = Path(args.out).with_suffix(".summary.json")
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {sum_path}")


if __name__ == "__main__":
    main()
