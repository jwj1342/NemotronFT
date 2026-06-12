"""Standalone KL-reference scorer (separate low-util vLLM process).

Why separate from rollout.py: computing prompt_logprobs over a ~7500-token bit
trace materialises a [seq_len x 131k vocab] logits tensor (~3.6 GB). At the
generation phase's gpu_memory_utilization=0.85 the KV cache leaves no headroom
→ CUDA OOM in _get_prompt_logprobs_dict. Running the ref pass in its OWN engine
at a LOW gpu_memory_utilization (small KV reserve, prefill-only, max_tokens=1)
leaves ~20 GB free for the logits spike, so it fits comfortably.

Reads a rollout JSONL (groups with response_ids), scores each response's tokens
under the FIXED reference adapter (warmstart → KL-to-warmstart), writes ref_logp
back into each group, in place.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "model")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import iter_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True, type=Path, help="rollout JSONL (modified in place)")
    ap.add_argument("--ref_adapter_dir", required=True, type=Path,
                    help="fixed reference adapter (warmstart) — KL-to-warmstart anchor")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.55,
                    help="LOW — leaves headroom for the prompt_logprobs logits spike.")
    ap.add_argument("--max_num_seqs", type=int, default=4)
    args = ap.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = list(iter_jsonl(args.rollout))
    n_resp = sum(len(r["group"]) for r in records)
    print(f"[ref] {len(records)} prompts, {n_resp} responses to score under {args.ref_adapter_dir}")
    if not records:
        print("[ref] empty rollout — nothing to do"); return

    t_load = time.time()
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, dtype="auto",
              max_model_len=args.max_model_len, trust_remote_code=True, enable_lora=True,
              max_lora_rank=args.max_lora_rank, max_num_seqs=args.max_num_seqs,
              enable_prefix_caching=False)
    print(f"[ref] vllm loaded in {time.time() - t_load:.1f}s (util={args.gpu_memory_utilization})")
    ref_lora = LoRARequest("ref", 2, str(args.ref_adapter_dir))

    flat_inputs, flat_meta = [], []
    for rec_idx, rec in enumerate(records):
        prompt_ids = rec["prompt_ids"]
        for g_idx, g in enumerate(rec["group"]):
            full_ids = list(prompt_ids) + list(g["response_ids"])
            flat_inputs.append({"prompt_token_ids": full_ids})
            flat_meta.append((rec_idx, g_idx, len(prompt_ids), len(g["response_ids"])))

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    t_ref = time.time()
    outs = llm.generate(flat_inputs, sp, lora_request=ref_lora)
    n_missing = 0
    for out, (rec_idx, g_idx, n_prompt, n_resp_tok) in zip(outs, flat_meta):
        full_ids_seq = out.prompt_token_ids
        plogs = out.prompt_logprobs or []
        logps = []
        for t in range(n_prompt, n_prompt + n_resp_tok):
            if t >= len(plogs) or t >= len(full_ids_seq):
                logps.append(0.0); n_missing += 1; continue
            lp_dict = plogs[t]; tok = full_ids_seq[t]
            if lp_dict is not None and tok in lp_dict:
                logps.append(float(lp_dict[tok].logprob))
            else:
                logps.append(0.0); n_missing += 1
        records[rec_idx]["group"][g_idx]["ref_logp"] = logps
    print(f"[ref] scored {len(flat_inputs)} seqs in {time.time() - t_ref:.1f}s ({n_missing} missing-token logps)")

    write_jsonl(args.rollout, records)
    print(f"[ref] wrote ref_logp back -> {args.rollout}")


if __name__ == "__main__":
    main()
