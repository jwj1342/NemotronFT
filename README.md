# NemotronFT

Fine-tuning **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`** for the [NVIDIA
Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

> **Handoff note (updated 2026-06-02).** The strong baseline is **LB 0.85**.
> Across two iterations, four routes to push past it have been tried and
> falsified: multi-category SFT distillation, a programmatic cryptarithm solver,
> hand-rolled GRPO (all 05-30), and a stronger combo-framework solver (06-02).
> The 06-02 work **quantitatively proved that any hand-written symbolic solver is
> weaker than the model itself on every sub-task**, so "weak solver → generate
> CoT → SFT" is necessarily negative transfer. A literature scan then pointed to
> a route that sidesteps this: **answer-filtered self-training (RFT/STaR)** — the
> model is its own (strongest) generator. The current plan is in
> **[`ROADMAP.md`](ROADMAP.md)**; the empirical record is in
> **[`ITERATION_REPORT.md`](ITERATION_REPORT.md)** (read §7 for 06-02). Read the
> iteration report first.

## Status at a glance

| Adapter | Public LB | Notes |
|---|---|---|
| `output/huikang-baseline-converted/` — Huikang's winning adapter, SVD-converted (rank 32, 75.77% singular mass) | **0.85** | current SOTA in this repo; everything is measured against it |
| `output/opd-v1/` — +1-epoch SFT on 1399 multi-category CoT | 0.85 | no movement |
| `output/opd-v2/` — +115 verified cryptarithm CoT (3× oversample) | 0.85 | no movement |
| `output/self-baseline/` — our own SFT on Huikang's 6355-problem CoT corpus | 0.76 | proves the full recipe works end-to-end on 4×L40s |
| Base model (no adapter) | ~0.45 | |

### Per-category accuracy of the 0.85 baseline (measured, 9500 train set)

Grader-aligned inference (T=0, `max_tokens=7680`, `top_p=1.0`,
`max_model_len=8192`), 4-shard × 4×L40s vLLM. Reproduce with
`eval/eval_full_train.sbatch`.

| Category | Correct / Total | Acc | |
|---|---|---|---|
| numeral | 1576 / 1576 | **100.00%** | maxed |
| unit_conversion | 1591 / 1594 | **99.81%** | near-max |
| gravity | 1591 / 1597 | **99.62%** | near-max |
| cipher | 1560 / 1576 | **98.98%** | near-max |
| equation_numeric_deduce | 542 / 596 | **90.94%** | good |
| bit_manipulation | 1402 / 1602 | **87.52%** | good, capped |
| equation_numeric_guess | 22 / 136 | **16.18%** | ❌ cliff |
| cryptarithm_deduce | 46 / 659 | **6.98%** | ❌ cliff |
| cryptarithm_guess | 10 / 164 | **6.10%** | ❌ cliff |
| **Total** | **8340 / 9500** | **87.79%** | |

All headroom is in three "cliff" categories. The ~3% gap between local 87.79%
and LB 0.85 is the harder private split, **not** a grader-config mismatch.

## What was tried this iteration (and why each stalled)

Full numbers in [`ITERATION_REPORT.md`](ITERATION_REPORT.md); one-line each:

1. **Multi-category SFT distillation** (opd-v1) → LB 0.85. No transfer to cliff
   categories.
2. **Cryptarithm-focused distillation** (opd-v2, only `verify()`-passing CoT) →
   LB 0.85. A holdout *canary* showed **negative transfer** (canary 2/30 vs
   baseline 3/30): the model memorizes each puzzle's random letter→digit map but
   it doesn't generalize. **SFT is bankrupt for cryptarithm.**
3. **Programmatic solver to auto-generate correct CoT** → the dominant failure
   cluster (the "F bucket", 540 / 643 = 84% of unsolved cryptarithm) is **0/100
   solvable** by our 21-op DSL across base-range and bijection variants. Wrong
   solver family, not a missing operator.
4. **Hand-rolled GRPO** → blocked at the source: the Huikang adapter is a
   near-deterministic policy. At T=0.95, G=8 it emits only **1.6–1.8 distinct
   sequences per group**, so intra-group advantage is ~0 and there is no
   gradient. Huikang's SFT compressed policy entropy to align with the T=0
   grader, which is exactly what GRPO needs and doesn't have.

5. **Stronger combo-framework solver** (06-02) → quantitatively falsified the
   whole "solver → CoT → SFT" route. The hand-written solver scores 59.9% (bit) /
   39.8% (numeric_deduce) / 0% (numeric_guess) vs the model's 87.5 / 90.9 / 16.2%;
   a cross-tab shows training on its output would teach +58 but mis-teach −500 on
   bit. Failure is **under-constraint, not a fixable bug**: 77% of bit failures
   differ by only 1–2 bits (ambiguous-bit guesses), truncation is just 3%.

**Takeaway**: the model is already a stronger solver than anything we can quickly
hand-write, and most remaining failures are information-theoretically
under-constrained (the model guesses by a learned frequency prior). The route the
literature supports — and that sidesteps the negative-transfer trap — is
**answer-filtered self-training (RFT/STaR/ReST-EM)**: the model generates its own
CoT, keep only answer-correct + example-consistent chains (the solvers become
*verifiers*, not generators), fine-tune, iterate. See [`ROADMAP.md`](ROADMAP.md)
§3 for the ranked plan and the literature. The single highest-leverage open
question is whether the grader permits any inference-time backprop or
multi-sampling (would unlock TTT / best-of-N).

## Why 0.85 needs SVD conversion

Huikang trained on **Tinker**, whose LoRA layout differs from Kaggle's vLLM
runtime in three places (see his
[writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)):

1. Mamba `in_proj` is trained as separate `gate_proj` + `x_proj` LoRA pairs.
   Submission needs a single `in_proj` LoRA, so we **SVD-merge** the two rank-32
   pairs back to rank 32 (loses ~25% singular mass — his "training-serving
   misalignment"). Adapters trained directly in PEFT here do **not** pay this.
2. MoE experts are stored as fused `experts.w1`/`experts.w2` with one shared
   LoRA; submission needs **per-expert `up_proj`/`down_proj`** LoRAs, so we
   broadcast the shared LoRA into 128 copies (lossless).
3. Adapter keys use prefix `base_model.model.model.*`; submission needs
   `base_model.model.backbone.*` (lossless rename).

All three are in `training/convert_tinker_adapter.py`.

## Repo layout

```
NemotronFT/
├── README.md                     # this file — start here
├── ITERATION_REPORT.md           # per-experiment results (local + Kaggle remote)
├── ROADMAP.md                    # current plan: falsified-solver finding + RFT/STaR routes (RL bet in §8 appendix)
├── RL_ARCHITECTURE.md            # GRPO math + single/multi-node FSDP reference
├── CLAUDE.md                     # operating context for Claude Code
├── training/
│   ├── train.py                  # FSDP LoRA SFT (manual resume via --resume_adapter_dir)
│   ├── train.sbatch              # 12h SLURM job, SIGUSR1 chain-resume
│   ├── dataset.py                # chat-template SFT dataset (enable_thinking=True)
│   ├── build_cot_csv.py          # decode Huikang corpus.jsonl → text CSV
│   ├── convert_tinker_adapter.py # Tinker adapter → Kaggle-format (SVD merge)
│   ├── fsdp_config.yaml          # single-node FULL_SHARD
│   ├── fsdp_multinode_config.yaml# HYBRID_SHARD for N nodes
│   ├── fsdp_offload_config.yaml  # CPU-offload variant
│   ├── rl/                       # hand-rolled GRPO pipeline (see rl/README.md)
│   │   ├── rollout.py reward.py grpo_loss.py train_grpo.py ...
│   │   └── scripts/              # smoke v3–v8, multi-node, hot/hard-pool builders
│   └── opd/                      # distillation + cryptarithm solver diagnostics
│       ├── build_trace_pool.py merge_traces.py prepare_opd_csv.py canary.sbatch
│       └── solvers/              # f_diagnostic, cluster_unsolved, extended_solver, ...
├── eval/
│   ├── extract.py                # extract_final_answer / verify / BOXED_INSTRUCTION
│   ├── eval_vllm.py/.sbatch      # grader-aligned eval (T=0, max_tokens=7680)
│   ├── eval_full_train.py/.sbatch# 9500-sample per-category eval
│   ├── eval_holdout.py/.sbatch   # tiny holdout eval (needs TP=4 on 30B BF16)
│   └── category.py               # detect_category() (importable without vLLM)
├── scripts/                      # build_submission, submit_checkpoint (token via env), ...
├── output/                       # adapters (gitignored)
├── model/  resource/  venv*/  logs/   # all gitignored
```

`training/rl/runs/`, `training/opd/runs/`, and `training/opd/solvers/runs/` hold
heavy artifacts (safetensors, rollout JSONL, SLURM logs) and are gitignored;
their results live in `ITERATION_REPORT.md`.

## Environments

Two manually-built venvs (see `CLAUDE.md` for the full activation incantations):

| Env | Purpose |
|---|---|
| `venv/` | training — `torch 2.5.1`, `transformers 5.3.0`, `peft`, `accelerate`, `mamba_ssm` |
| `venv-vllm/` | inference — `torch 2.10`, `vllm 0.17.1`, `transformers 4.57.6` |

`unset PIP_PREFIX` after sourcing either, or pip breaks inside the venv.

## Quick start

### Submit the strong baseline (~5 min)
```bash
export KAGGLE_API_TOKEN=...   # required: scripts no longer hardcode it
bash scripts/submit_checkpoint.sh \
    output/huikang-baseline-converted \
    "strong baseline: Huikang adapter, SVD-converted"
```

### Convert a fresh Tinker checkpoint
```bash
python training/convert_tinker_adapter.py \
    --in_dir output/<tinker-adapter> --out_dir output/<tinker-adapter>-converted
```

### Grader-aligned eval
```bash
sbatch eval/eval_full_train.sbatch        # full 9500, per-category
CHECKPOINT_DIR=output/huikang-baseline-converted sbatch eval/eval_vllm.sbatch
```

### Train on top of the 0.85 adapter (FSDP LoRA)
```bash
python training/build_cot_csv.py          # one-time, ~3 min
sbatch training/train.sbatch              # 12h, auto-resumes via SIGUSR1
```

## Two facts that are easy to get wrong again

**Grader uses T=0 greedy.** The public `metric.py` shows `temperature=1.0`, but
the actual grader overrides it. Train + eval must align to T=0 or the model
systematically underperforms.
```python
temperature=0.0  top_p=1.0  max_tokens=7680  max_model_len=8192
max_lora_rank=32  gpu_memory_utilization=0.85
```

**Mamba layers must be LoRA targets.** Nemotron-3-Nano is a hybrid
Mamba–Transformer (`nemotron_h`); the Mamba `in_proj`/`out_proj` MUST be in
`target_modules` or the SSM circuits stay frozen (this caused the early
0.27–0.60 runs). `training/train.py` uses:
```python
target_modules = ["q_proj","k_proj","v_proj","o_proj",
                  "in_proj","out_proj",            # Mamba SSM — do not omit
                  "up_proj","down_proj","lm_head"]
```
Other recipe constants (from the published 0.81 Unsloth notebook): LoRA
`r=alpha=32`, `dropout=0`, `lr=2e-4` linear (no warmup), AdamW `beta2=0.95`,
`weight_decay=0`, 1 epoch, effective batch 64, stratified by category.

## Where to look next

- **[`ITERATION_REPORT.md`](ITERATION_REPORT.md)** — what was tried, with exact
  local + Kaggle numbers. The empirical record (read §7 for the 06-02 work).
- **[`ROADMAP.md`](ROADMAP.md)** — the **current** plan: the falsified-solver
  finding, the literature-backed routes (RFT/STaR self-training, program-synthesis
  as offline generator, TTT), ranked by ROL. The old RL strategy is kept as a
  historical appendix (§8).
- **[`RL_ARCHITECTURE.md`](RL_ARCHITECTURE.md)** — GRPO objective, KL estimator,
  single/multi-node FSDP diagrams, the rollout/train decoupling. RL is on hold,
  but this is the reference if RFT reuses the rollout infrastructure.
- **`training/rl/README.md`** — module-level map of the GRPO/rollout code.
