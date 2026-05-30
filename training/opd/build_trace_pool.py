"""Generate T=0 traces from a LoRA adapter on a set of categories.

For OPD: produces high-quality on-policy CoT traces that become the SFT
target for the unified adapter. Runs in venv-vllm.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "model")
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from extract import extract_final_answer, verify, BOXED_INSTRUCTION  # type: ignore  # noqa: E402
from category import detect_category  # type: ignore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, type=Path)
    ap.add_argument("--categories", required=True, help="comma-separated")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--train_csv",
                    default=str(PROJECT_ROOT / "resource" / "winning-solution" / "train.csv"))
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_num_seqs", type=int, default=32)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--only_correct", action="store_true",
                    help="Filter to traces that verify against ground truth.")
    args = ap.parse_args()

    cats = {c.strip() for c in args.categories.split(",") if c.strip()}

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows: list[dict] = []
    with open(args.train_csv, newline="") as f:
        for r in csv.DictReader(f):
            cat = detect_category(r["prompt"])
            if cat in cats:
                rows.append({
                    "id": r["id"], "prompt": r["prompt"],
                    "answer": str(r["answer"]), "category": cat,
                })
    print(f"[opd-trace] {len(rows)} prompts in {sorted(cats)}")

    if not rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("")
        return

    print(f"[vllm] loading {MODEL_PATH} ...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()
    print(f"[vllm] loaded in {time.time() - t0:.1f}s")

    prompts: list[str] = []
    for r in rows:
        user_content = r["prompt"] + BOXED_INSTRUCTION
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        ))

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    lora = LoRARequest("adapter", 1, str(args.adapter_dir))
    print(f"[opd-trace] generating {len(prompts)} T=0 traces ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora)
    print(f"[opd-trace] done in {time.time() - t0:.1f}s")

    n_correct, n_total, n_kept = 0, 0, 0
    by_cat: dict[str, list[int]] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r, out in zip(rows, outputs):
            text = out.outputs[0].text
            pred = extract_final_answer(text)
            ok = verify(r["answer"], pred)
            n_total += 1
            n_correct += int(ok)
            by_cat.setdefault(r["category"], [0, 0])
            by_cat[r["category"]][0] += int(ok)
            by_cat[r["category"]][1] += 1
            if args.only_correct and not ok:
                continue
            f.write(json.dumps({
                "id": r["id"],
                "category": r["category"],
                "prompt": r["prompt"],
                "answer": r["answer"],
                "text": text,
                "predicted": pred,
                "correct": ok,
                "num_tokens": len(out.outputs[0].token_ids),
            }) + "\n")
            n_kept += 1

    print(f"[opd-trace] wrote {n_kept} traces to {args.out}")
    print(f"[opd-trace] overall correct: {n_correct}/{n_total} = "
          f"{n_correct / max(1, n_total):.3f}")
    for cat in sorted(by_cat):
        c, t = by_cat[cat]
        print(f"  {cat:30s} {c}/{t} = {c / max(1, t):.3f}")


if __name__ == "__main__":
    main()
