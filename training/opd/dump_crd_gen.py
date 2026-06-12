"""Dump checkpoint-125 generations on a sample of holdout problems to diagnose
WHY the probe scored low: trace-format present? anchor right? encoding bug?
undertrained rambling? Prints model output + gt + extracted + the reference cot.
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
PROJECT = Path("/scratch/jwj/NemotronFT")
MODEL_PATH = str(PROJECT / "model")
sys.path.insert(0, str(PROJECT / "eval"))
from extract import BOXED_INSTRUCTION, extract_final_answer, verify  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(PROJECT / "output/crd-probe/checkpoint-125"))
    ap.add_argument("--holdout", default=str(PROJECT / "data/crd_holdout.jsonl"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.holdout) if l.strip()][: args.n]
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=MODEL_PATH, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=8192, trust_remote_code=True,
              enable_lora=True, max_lora_rank=32, max_num_seqs=32, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"] + BOXED_INSTRUCTION}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in pool]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, n=1)
    outs = llm.generate(prompts, sp, lora_request=LoRARequest("c125", 1, args.adapter))

    for r, o in zip(pool, outs):
        txt = o.outputs[0].text
        pred = extract_final_answer(txt)
        ok = verify(r["answer"], pred)
        print("=" * 80)
        print(f"id={r.get('id')}  GT={r['answer']!r}  EXTRACTED={pred!r}  {'CORRECT' if ok else 'WRONG'}  out_tokens={len(o.outputs[0].token_ids)}")
        print(f"PROMPT: {r['prompt'][:240]}")
        print(f"--- MODEL OUTPUT ({len(txt)} chars) ---")
        print(txt[:3500])
        print(f"--- REFERENCE COT (what we trained toward) ---")
        print((r.get('cot') or '')[:1200])
    print("\nNOTE: compare model ANCHOR section vs reference — is the cipher derivation present & right?")


if __name__ == "__main__":
    main()
