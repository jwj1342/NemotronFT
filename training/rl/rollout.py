"""vLLM rollout phase.

Loads the current LoRA adapter, samples G trajectories per prompt at T>0,
computes reward via verify() and group-relative advantages, writes JSONL.

Runs in venv-vllm. One process per node; uses TP=4 within a node.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "model")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import iter_jsonl, write_jsonl  # noqa: E402
from reward import score, group_advantages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--adapter_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--max_num_seqs", type=int, default=32)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                    help="Use 0.85 for smoke / standalone rollout; "
                         "drop to 0.35 only when training will share the same allocation in-process.")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--shard_total", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--compute_ref_logp", action="store_true",
                    help="After generation, do a second vLLM pass to get reference "
                         "logprobs of each response token. Eliminates the need for a "
                         "second forward pass at training time → halves training-phase "
                         "memory. Reference policy = --ref_adapter_dir if given, else base.")
    ap.add_argument("--ref_adapter_dir", type=Path, default=None,
                    help="KL reference adapter for --compute_ref_logp. Set this to the "
                         "FIXED warmstart adapter to get KL-to-warmstart (anti-forgetting "
                         "anchor) instead of KL-to-base. Omit → ref = base model.")
    ap.add_argument("--stats_out", type=Path, default=None,
                    help="Optional JSON sidecar: rollout reward_mean/var, frac mixed/"
                         "degenerate groups (for TensorBoard logging by the driver).")
    args = ap.parse_args()

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    pool = list(iter_jsonl(args.pool))
    if args.shard_total > 1:
        pool = pool[args.shard_index::args.shard_total]
    print(f"[rollout] {len(pool)} prompts (shard {args.shard_index + 1}/{args.shard_total})")
    if not pool:
        print("[rollout] empty shard — nothing to do")
        write_jsonl(args.out, [])
        return

    print(f"[vllm] loading model from {MODEL_PATH} ...")
    t_load = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    print(f"[vllm] loaded in {time.time() - t_load:.1f}s")

    BOXED_INSTRUCTION = (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )
    prompt_texts: list[str] = []
    prompt_ids_list: list[list[int]] = []
    for rec in pool:
        user_content = rec["prompt"] + BOXED_INSTRUCTION
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_texts.append(text)
        prompt_ids_list.append(tokenizer(text, add_special_tokens=False).input_ids)

    # NOTE: do NOT set SamplingParams.seed when n>1.  Empirically (vllm 0.17.1)
    # with a fixed seed, all n completions for a prompt come back nearly identical
    # (131/154 same head 100 chars at T=0.95). Letting vllm use its default RNG
    # gives proper n=8 diversity.
    sampling = SamplingParams(
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    lora = LoRARequest("adapter", 1, str(args.adapter_dir))

    print(f"[generate] {len(prompt_texts)} prompts × G={args.group_size}")
    t_gen = time.time()
    outputs = llm.generate(prompt_texts, sampling, lora_request=lora)
    gen_elapsed = time.time() - t_gen
    print(f"[generate] {gen_elapsed:.1f}s ({gen_elapsed / max(1, len(prompt_texts)):.2f}s/prompt)")

    records = []
    all_rewards: list[float] = []
    by_cat: dict[str, list[float]] = {}
    # collapse/saturation accumulators
    resp_lens: list[int] = []
    n_truncated = 0
    n_boxed = 0
    distinct_ratios: list[float] = []
    for rec, prompt_ids, out in zip(pool, prompt_ids_list, outputs):
        group = []
        rewards = []
        for completion in out.outputs:
            r = score(completion.text, rec["gt"])
            rewards.append(r)
            all_rewards.append(r)
            resp_lens.append(len(completion.token_ids))
            if getattr(completion, "finish_reason", None) == "length":
                n_truncated += 1
            if "\\boxed{" in completion.text:
                n_boxed += 1
            group.append({
                "text": completion.text,
                "response_ids": list(completion.token_ids),
                "reward": r,
            })
        # diversity: distinct completions / group size (1.0 = all unique, →0 = collapse)
        distinct_ratios.append(len(set(g["text"] for g in group)) / max(1, len(group)))
        advs = group_advantages(rewards)
        for g, a in zip(group, advs):
            g["advantage"] = float(a)
        by_cat.setdefault(rec["category"], []).extend(rewards)
        records.append({
            "prompt_id": rec["id"],
            "category": rec["category"],
            "prompt": rec["prompt"],
            "gt": rec["gt"],
            "prompt_ids": list(prompt_ids),
            "group": group,
        })

    # Optional: pre-compute reference-policy logprobs of response tokens to avoid
    # the disable_adapter forward at training time. Reference = --ref_adapter_dir
    # (the FIXED warmstart adapter → KL-to-warmstart anchor) when given, else base.
    if args.compute_ref_logp:
        from vllm import SamplingParams as _SP  # avoid re-import warning
        if args.ref_adapter_dir is not None:
            # distinct int_id from the policy LoRA (id=1) so vLLM caches both
            ref_lora = LoRARequest("ref", 2, str(args.ref_adapter_dir))
            ref_src = f"warmstart adapter {args.ref_adapter_dir}"
        else:
            ref_lora = None
            ref_src = "base model (no LoRA)"
        print(f"[ref-logp] computing reference logprobs of "
              f"{sum(len(r['group']) for r in records)} responses under {ref_src} ...")
        t_ref = time.time()
        flat_inputs = []
        flat_meta = []  # (rec_idx, g_idx, n_prompt, n_resp)
        for rec_idx, (rec, prompt_ids) in enumerate(zip(pool, prompt_ids_list)):
            for g_idx, g in enumerate(records[rec_idx]["group"]):
                full_ids = list(prompt_ids) + list(g["response_ids"])
                flat_inputs.append({"prompt_token_ids": full_ids})
                flat_meta.append((rec_idx, g_idx, len(prompt_ids), len(g["response_ids"])))
        ref_sp = _SP(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        ref_outputs = llm.generate(flat_inputs, ref_sp, lora_request=ref_lora)
        n_missing = 0
        for ref_out, (rec_idx, g_idx, n_prompt, n_resp) in zip(ref_outputs, flat_meta):
            full_ids_seq = ref_out.prompt_token_ids
            plogs = ref_out.prompt_logprobs or []
            logps: list[float] = []
            for t in range(n_prompt, n_prompt + n_resp):
                if t >= len(plogs) or t >= len(full_ids_seq):
                    logps.append(0.0); n_missing += 1; continue
                lp_dict = plogs[t]
                tok = full_ids_seq[t]
                if lp_dict is not None and tok in lp_dict:
                    logps.append(float(lp_dict[tok].logprob))
                else:
                    logps.append(0.0); n_missing += 1
            records[rec_idx]["group"][g_idx]["ref_logp"] = logps
        ref_elapsed = time.time() - t_ref
        print(f"[ref-logp] done in {ref_elapsed:.1f}s "
              f"({len(flat_inputs)} seqs, {n_missing} missing-token logps)")

    n = write_jsonl(args.out, records)
    if all_rewards:
        mean_r = sum(all_rewards) / len(all_rewards)
        var_r = sum((x - mean_r) ** 2 for x in all_rewards) / len(all_rewards)
        # group composition: mixed groups (have both correct & wrong) drive the
        # GRPO gradient; degenerate (all-same) groups contribute zero advantage.
        n_mixed = n_all_right = n_all_wrong = 0
        for rec in records:
            rs = [g["reward"] for g in rec["group"]]
            if all(x == rs[0] for x in rs):
                if rs[0] > 0:
                    n_all_right += 1
                else:
                    n_all_wrong += 1
            else:
                n_mixed += 1
        frac_mixed = n_mixed / max(1, len(records))
        n_resp_total = len(all_rewards)
        resp_len_mean = sum(resp_lens) / max(1, len(resp_lens))
        frac_truncated = n_truncated / max(1, n_resp_total)
        boxed_rate = n_boxed / max(1, n_resp_total)
        distinct_ratio = sum(distinct_ratios) / max(1, len(distinct_ratios))
        print(f"[rollout] wrote {n} prompts  reward_mean={mean_r:.4f}  reward_var={var_r:.4f}")
        print(f"[rollout] groups: mixed={n_mixed} (frac={frac_mixed:.3f}, gradient) "
              f"all_right={n_all_right} all_wrong={n_all_wrong} (zero-advantage)")
        print(f"[rollout] COLLAPSE-WATCH: resp_len_mean={resp_len_mean:.0f} "
              f"frac_truncated={frac_truncated:.3f} boxed_rate={boxed_rate:.3f} "
              f"distinct_ratio={distinct_ratio:.3f}")
        for cat in sorted(by_cat):
            rs = by_cat[cat]
            print(f"  {cat:30s} n={len(rs)}  pass@1={sum(rs)/len(rs):.3f}")
        if args.stats_out is not None:
            import json as _json
            args.stats_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.stats_out, "w") as f:
                _json.dump({"reward_mean": mean_r, "reward_var": var_r,
                            "frac_mixed_groups": frac_mixed, "n_mixed": n_mixed,
                            "n_all_right": n_all_right, "n_all_wrong": n_all_wrong,
                            "n_prompts": len(records),
                            "resp_len_mean": resp_len_mean, "frac_truncated": frac_truncated,
                            "boxed_rate": boxed_rate, "distinct_ratio": distinct_ratio}, f)
            print(f"[rollout] stats -> {args.stats_out}")
    else:
        print(f"[rollout] wrote {n} prompts (no rewards)")


if __name__ == "__main__":
    main()
