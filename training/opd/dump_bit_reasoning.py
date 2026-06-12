"""Dump full bit_manipulation reasoning (correct, long) from the base adapter to
judge COMPRESSIBILITY — the gate for the brevity/length-insurance experiment.

bit-correct answers sit at median 6692 / cap 7680 tokens (1374 of them >6000).
The whole private length-insurance lever rides on whether that 6692-token chain is
mostly HABITUAL NARRATION (compressible without losing the answer) or ESSENTIAL
bit-by-bit COMPUTATION (compressing breaks correctness -> lever dead).

Generates a handful of bit-correct problems at T=0 with the lossy 0.85 base and
prints the full chains so we can read them and decide before any training run.
"""
from __future__ import annotations
import argparse, json, glob, sys, os, csv
from pathlib import Path
PROJECT = Path("/scratch/jwj/NemotronFT")
MODEL_PATH = str(PROJECT / "model")
sys.path.insert(0, str(PROJECT / "eval"))
from extract import BOXED_INSTRUCTION, extract_final_answer, verify  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(PROJECT / "output/huikang-baseline-converted"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out", default=str(PROJECT / "data/bit_reasoning_dump.txt"))
    args = ap.parse_args()

    # pick n bit-correct problems with num_tokens>6000 (representative of the at-risk pool)
    csv.field_size_limit(10**7)
    prompts = {}
    with open(PROJECT / "resource/winning-solution/train.csv") as f:
        for r in csv.DictReader(f):
            prompts[r["id"]] = r["prompt"]
    pool = []
    for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
        for line in open(shard):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["category"] == "bit_manipulation" and r["correct"] and r["num_tokens"] > 6000:
                if r["id"] in prompts:
                    pool.append({"id": r["id"], "prompt": prompts[r["id"]],
                                 "answer": str(r["answer"]), "ref_tok": r["num_tokens"]})
    pool = pool[: args.n]
    print(f"[dump] {len(pool)} bit-correct problems (num_tokens>6000)", flush=True)

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=8192, trust_remote_code=True,
              enable_lora=True, max_lora_rank=32, max_num_seqs=16, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts_fmt = [tok.apply_chat_template([{"role": "user", "content": r["prompt"] + BOXED_INSTRUCTION}],
                   tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in pool]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, n=1)
    outs = llm.generate(prompts_fmt, sp, lora_request=LoRARequest("base", 1, args.adapter))

    with open(args.out, "w") as fout:
        for r, o in zip(pool, outs):
            txt = o.outputs[0].text
            pred = extract_final_answer(txt)
            ok = verify(r["answer"], pred)
            ntok = len(o.outputs[0].token_ids)
            hdr = (f"{'='*90}\nid={r['id']} GT={r['answer']} PRED={pred} {'OK' if ok else 'WRONG'} "
                   f"gen_tokens={ntok} (ref={r['ref_tok']})\nPROMPT:\n{r['prompt']}\n--- FULL REASONING ---\n{txt}\n")
            fout.write(hdr)
            print(f"id={r['id']} {'OK' if ok else 'WRONG'} gen_tokens={ntok}", flush=True)
    print(f"[done] wrote full chains -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
