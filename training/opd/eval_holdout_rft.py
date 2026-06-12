"""
eval_holdout_rft.py — T=0 grader-aligned three-metric eval for RFT checkpoints.

For the baseline adapter AND each RFT checkpoint, run a SINGLE greedy (T=0)
completion per holdout/sentinel puzzle (exact grader config), extract+verify.
Then, using the BASELINE's own T=0 results as the reference (NOT the possibly-
polluted `predicted` field in the eval shards), report per checkpoint:

  bit_recovered   : bit holdout puzzles baseline got WRONG that the ckpt gets RIGHT
  bit_regressed   : bit holdout puzzles baseline got RIGHT that the ckpt gets WRONG
  sentinel_regress: non-bit sentinel puzzles (all baseline-correct) the ckpt breaks
  net_gain        : bit_recovered - bit_regressed - sentinel_regress

Pick the checkpoint with the best net_gain (and zero/low regression). This is the
Codex transfer gate: accept only if net_gain is clearly positive.

Runs in venv-vllm, TP=4. Read-only on all adapters.
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
MODEL_PATH = str(PROJECT / "model")
sys.path.insert(0, str(PROJECT / "eval"))
from extract import BOXED_INSTRUCTION, extract_final_answer, verify  # type: ignore


def load_pool(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(PROJECT / "training/opd/runs/rft/r0/eval_pool.jsonl"))
    ap.add_argument("--baseline_adapter", default=str(PROJECT / "output/huikang-baseline-converted"))
    ap.add_argument("--scan_dir", default=str(PROJECT / "output/rft-r0-pilot"),
                    help="dir containing checkpoint-* subdirs to eval")
    ap.add_argument("--checkpoints", action="append", default=[],
                    help="explicit checkpoint dirs (overrides scan_dir if given)")
    ap.add_argument("--out", default=str(PROJECT / "training/opd/runs/rft/r0/holdout_eval.json"))
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    print(f"[eval] {len(pool)} puzzles from {args.pool}")

    # adapters to eval: baseline first (reference), then checkpoints (sorted by step)
    if args.checkpoints:
        ckpts = [Path(c) for c in args.checkpoints]
    else:
        ckpts = sorted(
            [Path(d) for d in glob.glob(str(Path(args.scan_dir) / "checkpoint-*"))
             if (Path(d) / "adapter_config.json").exists()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        # also include final adapter if present
        if (Path(args.scan_dir) / "adapter_config.json").exists():
            ckpts.append(Path(args.scan_dir))
    adapters = [("BASELINE", Path(args.baseline_adapter))] + [(p.name, p) for p in ckpts]
    print(f"[eval] adapters: {[a[0] for a in adapters]}")

    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, dtype="auto",
              max_model_len=args.max_model_len, trust_remote_code=True,
              enable_lora=True, max_lora_rank=args.max_lora_rank,
              max_num_seqs=64, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    print(f"[vllm] loaded in {time.time()-t0:.0f}s")

    prompts = []
    for r in pool:
        uc = r["prompt"] + BOXED_INSTRUCTION
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": uc}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True))

    # grader-aligned greedy
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, n=1)

    # correctness per adapter: {adapter_name: {id: bool}}
    corr = {}
    for idx, (name, path) in enumerate(adapters):
        lora = LoRARequest(name, idx + 1, str(path))
        tg = time.time()
        outs = llm.generate(prompts, sp, lora_request=lora)
        d = {}
        for r, o in zip(pool, outs):
            d[r["id"]] = verify(r["gt"], extract_final_answer(o.outputs[0].text))
        corr[name] = d
        acc = sum(d.values()) / len(d)
        print(f"[eval] {name:18s} acc={acc:.3f} ({time.time()-tg:.0f}s)")

    # three metrics vs BASELINE's own T=0 results
    base = corr["BASELINE"]
    bit_fail = [r["id"] for r in pool if r["category"] == "bit_manipulation" and not r["baseline_correct"]]
    bit_corr = [r["id"] for r in pool if r["category"] == "bit_manipulation" and r["baseline_correct"]]
    sent = [r["id"] for r in pool if r["category"] != "bit_manipulation"]

    results = []
    print("\n" + "=" * 78)
    print(f"{'checkpoint':18s} {'bit_recov':>9s} {'bit_regr':>8s} {'sent_regr':>9s} {'NET':>5s}  {'holdout_acc':>11s}")
    print("-" * 78)
    for name, _ in adapters:
        if name == "BASELINE":
            continue
        c = corr[name]
        # recovered: baseline-failed bit puzzles now correct
        recov = sum(1 for i in bit_fail if c[i] and not base[i])
        # regressed: baseline-correct bit puzzles now wrong
        regr = sum(1 for i in bit_corr if base[i] and not c[i])
        # sentinel regression: non-bit baseline-correct now wrong
        sregr = sum(1 for i in sent if base[i] and not c[i])
        net = recov - regr - sregr
        hold_ids = bit_fail + bit_corr
        hacc = sum(c[i] for i in hold_ids) / len(hold_ids)
        results.append({"checkpoint": name, "bit_recovered": recov,
                        "bit_regressed": regr, "sentinel_regressed": sregr,
                        "net_gain": net, "holdout_acc": hacc})
        print(f"{name:18s} {recov:9d} {regr:8d} {sregr:9d} {net:5d}  {hacc:11.3f}")
    print("=" * 78)

    # baseline holdout acc for reference
    base_hacc = sum(base[i] for i in (bit_fail + bit_corr)) / len(bit_fail + bit_corr)
    print(f"baseline holdout_acc = {base_hacc:.3f}  (bit_fail={len(bit_fail)} bit_corr={len(bit_corr)} sentinel={len(sent)})")
    best = max(results, key=lambda r: r["net_gain"]) if results else None
    if best:
        print(f"\nBEST net_gain: {best['checkpoint']} "
              f"(+{best['bit_recovered']} -{best['bit_regressed']}bit -{best['sentinel_regressed']}sent = {best['net_gain']})")
        print("ACCEPT only if net_gain clearly >0 AND regression tolerable.")

    with open(args.out, "w") as f:
        json.dump({"baseline_holdout_acc": base_hacc, "per_checkpoint": results,
                   "n_bit_fail": len(bit_fail), "n_bit_corr": len(bit_corr),
                   "n_sentinel": len(sent)}, f, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
