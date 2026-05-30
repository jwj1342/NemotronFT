# training/rl/

Hand-rolled GRPO trainer for the Nemotron LoRA pipeline.

## Modules

| File | Role | Runtime |
|---|---|---|
| `pool.py` | Build per-task prompt pools from `train.csv` (uses `eval/category.py`) | `venv` (no heavy deps) |
| `rollout.py` | vLLM rollout: load adapter, sample G trajectories at T>0, write JSONL | `venv-vllm` |
| `reward.py` | `verify(extract_final_answer(text), gt)` wrapper + group-advantage normalization | either |
| `ref_policy.py` | `disabled_adapter(model)` context manager (PEFT trick for reference logits) | `venv` |
| `grpo_loss.py` | Token-level logprobs + GRPO objective + Schulman k3 KL | `venv` |
| `train_grpo.py` | FSDP+PEFT manual training loop, consumes rollout JSONL, emits new adapter | `venv` |
| `utils.py` | JSONL io + prompt template helpers | either |

## Scripts (sbatch)

- `scripts/smoke.sbatch` — Stage 0 single-node smoke (10 prompts, 25 GRPO steps). Writes `runs/stage0/SMOKE_OK` on success.
- `scripts/smoke_multinode.sbatch` — 2-node NCCL all-reduce sanity. Writes `runs/stage0/MULTINODE_OK`.
- `scripts/iter.sbatch` — outer driver (rollout → train → log). SIGUSR1 chain-resume.
- `scripts/launch_stage1.sh` — convenience launcher for Stage 1 (gated by SMOKE_OK).

## Stage launch examples

```bash
# Smoke (required before anything else)
sbatch training/rl/scripts/smoke.sbatch

# Optional: validate multi-node NCCL (after smoke passes)
sbatch training/rl/scripts/smoke_multinode.sbatch

# Stage 1: RL on equation_numeric_guess
bash training/rl/scripts/launch_stage1.sh

# Stage 3: RL on cryptarithm (combined deduce + guess)
sbatch --export=ALL,\
STAGE_NAME=stage3_cryptarithm,\
STAGE_CATEGORIES=cryptarithm_deduce,cryptarithm_guess,\
N_OUTER=15,N_INNER=50,GROUP_SIZE=8,LR=3e-6,BETA=0.05 \
    training/rl/scripts/iter.sbatch
```

## Output layout

```
runs/<stage>/
  pool.jsonl                # {id, prompt, gt, category}
  policy/iter_K/            # LoRA adapter after K outer iters
  rollouts/iter_K.jsonl     # rewards + token_ids per group
  progress.csv              # iter, n, reward_mean, per-cat pass@1
  COMPLETE                  # written when N_OUTER iters done
  adapter_model.safetensors # final adapter (== last iter's)
```

## Gates

See `ROADMAP.md` §5 (repo root) for the full G0..G4 specifications, and
`ITERATION_REPORT.md` for why the smoke-stage diversity check ultimately
gated the whole RL effort. Mainline checks at G0 (smoke):
- rollout JSONL well-formed (`python training/rl/reward.py --jsonl ...` reproduces stored rewards)
- non-zero reward variance on mixed pool
- `disable_adapter` produces logp_ref distinguishable from logp_new (sanity print at iter 0)
- `adapter_model.safetensors` saves under FSDP

## Known constraints

- Single-node only on initial run (multi-node code paths are written but smoke
  must validate NCCL first; see `smoke_multinode.sbatch`).
- vLLM `gpu_memory_utilization=0.85` is OK because rollout/train are sequential
  in separate processes — vLLM owns the GPU during its phase, FSDP during its.
- LoRA r=32 / alpha=32 / Mamba `in_proj`/`out_proj` targets are pinned —
  changing breaks the Kaggle grader's `max_lora_rank=32` ceiling.
