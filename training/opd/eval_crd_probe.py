"""
eval_crd_probe.py — T=0 grader-aligned eval for the cryptarithm_deduce
executability probe.

For BASE (no adapter) and each probe checkpoint, run a SINGLE greedy (T=0)
completion per holdout problem (exact grader config: T=0, top_p=1, max_tokens
7680, max_model_len 8192, max_lora_rank 32), extract \\boxed{}, verify vs the
ground-truth answer. Report accuracy overall AND on the well-posed subset.

DECISION GATE (Codex): holdout <25-30% -> the bounded-search CoT is unlearnable,
the whole direction is dead; 35%+ -> continue to Phase 1; 42%+ -> real 0.89 path.

Runs in venv-vllm, TP=4. Read-only on all adapters.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
MODEL_PATH = str(PROJECT / "model")
sys.path.insert(0, str(PROJECT / "eval"))
from extract import BOXED_INSTRUCTION, extract_final_answer, verify  # type: ignore


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(PROJECT / "data/crd_holdout.jsonl"))
    ap.add_argument("--scan_dir", default=str(PROJECT / "output/crd-probe"),
                    help="dir containing checkpoint-* subdirs to eval")
    ap.add_argument("--checkpoints", action="append", default=[])
    ap.add_argument("--eval_base", action="store_true",
                    help="also eval the BASE model with no adapter (reference)")
    ap.add_argument("--out", default=str(PROJECT / "output/crd-probe/holdout_eval.json"))
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    pool = load(args.holdout)
    # normalise fields: need prompt, gt-answer, optional well_posed/id/category
    for i, r in enumerate(pool):
        r.setdefault("id", str(i))
        r["_gt"] = r.get("answer") or r.get("gt")
        r["_wp"] = bool(r.get("well_posed", True))
    print(f"[eval] {len(pool)} holdout problems  ({sum(r['_wp'] for r in pool)} well-posed)")

    if args.checkpoints:
        ckpts = [Path(c) for c in args.checkpoints]
    else:
        ckpts = sorted(
            [Path(d) for d in glob.glob(str(Path(args.scan_dir) / "checkpoint-*"))
             if (Path(d) / "adapter_config.json").exists()],
            key=lambda p: int(p.name.split("-")[-1]))
        if (Path(args.scan_dir) / "adapter_config.json").exists():
            ckpts.append(Path(args.scan_dir))
    print(f"[eval] checkpoints: {[c.name for c in ckpts]}  eval_base={args.eval_base}")

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
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, n=1)

    runs = []
    if args.eval_base:
        runs.append(("BASE", None))
    runs += [(p.name, p) for p in ckpts]

    results = []
    for idx, (name, path) in enumerate(runs):
        tg = time.time()
        if path is None:
            outs = llm.generate(prompts, sp)
        else:
            outs = llm.generate(prompts, sp, lora_request=LoRARequest(name, idx + 1, str(path)))
        ok = {}
        for r, o in zip(pool, outs):
            ok[r["id"]] = verify(r["_gt"], extract_final_answer(o.outputs[0].text))
        acc = sum(ok.values()) / len(pool)
        wp = [r["id"] for r in pool if r["_wp"]]
        wacc = (sum(ok[i] for i in wp) / len(wp)) if wp else 0.0
        results.append({"name": name, "holdout_acc": acc, "wellposed_acc": wacc,
                        "n": len(pool), "n_wellposed": len(wp)})
        print(f"[eval] {name:18s} holdout_acc={acc:.3f}  wellposed_acc={wacc:.3f}  ({time.time()-tg:.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"holdout": args.holdout, "results": results}, open(args.out, "w"), indent=2)
    print("\n=== GATE (Codex): <25-30% DEAD | 35%+ continue | 42%+ 0.89-path ===")
    best = max(results, key=lambda r: r["holdout_acc"])
    print(f"BEST: {best['name']} holdout={best['holdout_acc']:.3f} wellposed={best['wellposed_acc']:.3f}")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
