"""Screen several brevity checkpoints in ONE vLLM session (load 30B base once,
swap LoRA per checkpoint) on shard-0-of-4 of train.csv — the SAME 2375 problems as
the existing baseline shard (output/eval_full_huikang-baseline-converted_shard0of4),
so per-category comparison is apples-to-apples.

Grader-exact: T=0, top_p=1, max_tokens=7680, max_model_len=8192.
Gate focus: did brevity REGRESS any category (esp. bit_manipulation — its 6600-token
search is the regression risk), and did overall hold/improve vs baseline?

Usage: python eval_sweep.py --adapters DIR1 DIR2 ... [--shard_index 0 --shard_total 4]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT / "model")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import extract_final_answer, verify, BOXED_INSTRUCTION
from eval_full_train import detect_category


def score(outputs, df):
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    rows = []
    for i, o in enumerate(outputs):
        r = df.iloc[i]
        txt = o.outputs[0].text
        pred = extract_final_answer(txt)
        ok = verify(str(r["answer"]), pred)
        per_cat[r["category"]]["total"] += 1
        if ok:
            per_cat[r["category"]]["correct"] += 1
        rows.append({"id": r["id"], "category": r["category"], "answer": r["answer"],
                     "predicted": pred, "correct": ok, "num_tokens": len(o.outputs[0].token_ids)})
    return per_cat, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--train_csv", default=str(PROJECT / "resource/winning-solution/train.csv"))
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--shard_total", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv, dtype={"answer": str})
    if args.shard_total > 1:
        df = df.iloc[args.shard_index::args.shard_total].reset_index(drop=True)
    df["category"] = df["prompt"].apply(detect_category)
    print(f"[data] shard {args.shard_index}/{args.shard_total}: {len(df)} problems", flush=True)

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=8192, trust_remote_code=True, enable_lora=True,
              max_lora_rank=32, max_num_seqs=32, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": p + BOXED_INSTRUCTION}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in df["prompt"]]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    results = {}
    for li, adir in enumerate(args.adapters, 1):
        name = Path(adir).name
        t0 = time.time()
        outs = llm.generate(prompts, sp, lora_request=LoRARequest(name, li, adir))
        per_cat, rows = score(outs, df)
        gc = sum(d["correct"] for d in per_cat.values()); gt = sum(d["total"] for d in per_cat.values())
        results[name] = {"overall": gc / gt, "n": gt,
                         "per_cat": {c: per_cat[c]["correct"] / per_cat[c]["total"] for c in per_cat},
                         "per_cat_n": {c: per_cat[c]["total"] for c in per_cat},
                         "bit_tokens_mean": (sum(r["num_tokens"] for r in rows if r["category"]=="bit_manipulation") /
                                             max(1, sum(1 for r in rows if r["category"]=="bit_manipulation")))}
        out_jsonl = PROJECT / "output" / f"sweep_{name}_shard{args.shard_index}of{args.shard_total}.jsonl"
        with open(out_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[{li}/{len(args.adapters)}] {name}: overall={gc/gt*100:.2f}% ({time.time()-t0:.0f}s) -> {out_jsonl.name}", flush=True)

    out = PROJECT / "output" / f"sweep_summary_shard{args.shard_index}of{args.shard_total}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {out}")
    # comparison table
    cats = sorted({c for r in results.values() for c in r["per_cat"]})
    print("\n" + "=" * 110)
    print(f"{'adapter':22s} {'overall':>8s} " + " ".join(f"{c[:9]:>9s}" for c in cats) + f" {'bitTok':>7s}")
    for name, r in results.items():
        print(f"{name:22s} {r['overall']*100:>7.2f}% " +
              " ".join(f"{r['per_cat'].get(c,0)*100:>8.1f}%" for c in cats) + f" {r['bit_tokens_mean']:>7.0f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
