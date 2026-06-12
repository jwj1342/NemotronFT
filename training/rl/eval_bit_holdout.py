"""Judge bit GRPO net gain on the DISJOINT holdout (the only valid salvageability test).
Greedy T=0 (grader-exact) for warmstart(=0.85 baseline) and the GRPO-trained adapter on
the same 100 holdout bit problems. Reports per-adapter accuracy split by the baseline's
greedy_correct flag, then recover (failures warmstart got wrong -> trained right) minus
regress (correct warmstart got right -> trained wrong). net>0 -> GRPO helps."""
from __future__ import annotations
import argparse, json
from pathlib import Path
PROJECT = Path("/scratch/jwj/NemotronFT")
MODEL_PATH = str(PROJECT / "model")
import sys
sys.path.insert(0, str(PROJECT / "eval"))
from extract import extract_final_answer, verify, BOXED_INSTRUCTION  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(PROJECT / "training/rl/runs/hotpool/bit_grpo_holdout.jsonl"))
    ap.add_argument("--warm", default=str(PROJECT / "output/huikang-warmstart"))
    ap.add_argument("--trained", default=str(PROJECT / "training/rl/runs/bit_grpo_val/policy_iter1"))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out_json", default=None,
                    help="Optional: dump {recover,regress,net,warm_overall,trained_overall,...} "
                         "as JSON so the iterative driver can log per-iteration TB scalars.")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.holdout) if l.strip()]
    import os; os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=8192, trust_remote_code=True, enable_lora=True,
              max_lora_rank=32, max_num_seqs=32, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"] + BOXED_INSTRUCTION}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in pool]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)

    def run(adir, tag, li):
        outs = llm.generate(prompts, sp, lora_request=LoRARequest(tag, li, adir))
        return [verify(str(r["gt"]), extract_final_answer(o.outputs[0].text)) for r, o in zip(pool, outs)]

    warm = run(args.warm, "warm", 1)
    trained = run(args.trained, "trained", 2)

    fails = [i for i, r in enumerate(pool) if not r["greedy_correct"]]
    corrs = [i for i, r in enumerate(pool) if r["greedy_correct"]]
    def acc(res, idxs): return sum(res[i] for i in idxs) / max(1, len(idxs))
    recover = sum(1 for i in fails if (not warm[i]) and trained[i])
    regress = sum(1 for i in corrs if warm[i] and (not trained[i]))
    print("\n=== BIT GRPO HOLDOUT NET-GAIN (T=0, n=%d: %d fail, %d correct) ===" % (len(pool), len(fails), len(corrs)))
    print(f"  warmstart : overall={sum(warm)/len(pool)*100:5.1f}%  fail-subset={acc(warm,fails)*100:5.1f}%  correct-subset={acc(warm,corrs)*100:5.1f}%")
    print(f"  trained   : overall={sum(trained)/len(pool)*100:5.1f}%  fail-subset={acc(trained,fails)*100:5.1f}%  correct-subset={acc(trained,corrs)*100:5.1f}%")
    print(f"  RECOVER (fail wrong->right): {recover}/{len(fails)}")
    print(f"  REGRESS (correct right->wrong): {regress}/{len(corrs)}")
    net = recover - regress
    print(f"  NET on holdout = recover - regress = {net:+d}  ({net/len(pool)*100:+.1f}pp on this 100)")
    print(f"  VERDICT: {'PROMISING — net>0, extrapolate to full set' if net > 0 else 'WALL HOLDS — net<=0 (prior coupling, as RFT)'}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"recover": recover, "regress": regress, "net": net,
                       "warm_overall": sum(warm) / len(pool), "trained_overall": sum(trained) / len(pool),
                       "warm_fail": acc(warm, fails), "trained_fail": acc(trained, fails),
                       "warm_corr": acc(warm, corrs), "trained_corr": acc(trained, corrs),
                       "n_pool": len(pool), "n_fail": len(fails), "n_corr": len(corrs)}, f)
        print(f"  [eval] metrics -> {args.out_json}")


if __name__ == "__main__":
    main()
