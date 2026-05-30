# CLAUDE.md

Operating context for Claude Code in this repo.

## Project at a glance

This is a LoRA fine-tuning workspace for the **NVIDIA Nemotron Model Reasoning
Challenge** on Kaggle. The target model is
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` — a 32B MoE with **hybrid
Mamba–Transformer** layers (`nemotron_h` architecture).

The strong baseline that everything builds against is
`output/huikang-baseline-converted/` (Huikang's published winning adapter,
SVD-converted, public LB **0.85**). On top of that baseline the repo carries:
the SFT scaffolding that produced our self-trained 0.76 run, grader-aligned
evaluation tooling, a distillation pipeline (`training/opd/`), and a hand-rolled
GRPO pipeline (`training/rl/`).

**Current state (2026-05-30): handed off to an external reviewer.** This
iteration tried three routes past 0.85 — multi-category SFT distillation, a
programmatic cryptarithm solver, and GRPO — and all three stalled at 0.85. The
per-experiment record (local + Kaggle remote) is in `ITERATION_REPORT.md`; the
strategic rationale is in `ROADMAP.md`; the RL engineering reference is in
`RL_ARCHITECTURE.md`. Those three docs + this file + `README.md` live at the
repo root and are the handoff package. **Read `ITERATION_REPORT.md` first.**

## Hardware + environments

We run on **Vulcan (Alliance Canada)**, default account `aip-zhouyang`. Two
Python envs in this repo, both manually set up by the user:

| Env | Purpose | Activation |
|---|---|---|
| `venv/` | Training (`torch 2.5.1`, `transformers 5.3.0`, `peft`, `accelerate`, `mamba_ssm`) | `module load python/3.11.5 && source venv/bin/activate && unset PIP_PREFIX` |
| `venv-vllm/` | Inference (`torch 2.10`, `vllm 0.17.1`, `transformers 4.57.6`) | `module load python/3.11.5 gcc/12.3 opencv/4.11.0 && source venv-vllm/bin/activate && unset PIP_PREFIX` |

`unset PIP_PREFIX` is mandatory after sourcing — `PIP_PREFIX` gets leaked from
the parent shell and silently breaks pip operations inside the venv.

## Hard-won project facts

Things that took experiments to discover and are easy to get wrong again:

- **Grader uses T=0 greedy, NOT T=1.** The public `metric.py` shows
  `temperature=1.0` as a default, but Huikang clarified those defaults are
  overridden by the actual grader. Train + eval must align to T=0 or the model
  systematically underperforms (any single token with logprob < 50% derails
  the chain).
- **Grader budgets**: `max_tokens=7680`, `max_model_len=8192`,
  `max_lora_rank=32`. Filtering training data to `response<=2500` (a previous
  attempt) is counterproductive — the model needs to learn long, complete
  reasoning that ends in `\boxed{}` within the budget.
- **Mamba layers must be in `target_modules`**: `in_proj` and `out_proj`. The
  v3 → v5 runs (scores 0.27–0.60) all missed these and the Mamba reasoning
  circuits stayed frozen. v6 fixed this and jumped to 0.76.
- **HF Trainer 5.3's `resume_from_checkpoint` is broken with PEFT + FSDP** —
  it looks for `pytorch_model.bin.index.json` rather than the PEFT adapter.
  `training/train.py` bypasses it via `--resume_adapter_dir` (manual
  `PeftModel.from_pretrained`) and `training/train.sbatch` rescales `max_steps`
  + starting LR so the linear schedule stays continuous across chained jobs.
- **SLURM SIGUSR1 + bash trap for chain submission**: if you rely on the
  `accelerate launch` exit path to submit the next chained job, SLURM SIGKILLs
  the whole bash on timeout and the chain never fires. The current `.sbatch`
  uses `#SBATCH --signal=B:USR1@600` and pre-submits the successor 10 min
  before timeout.
- **Tinker → Kaggle conversion is lossy by ~25%** (SVD on Mamba
  `gate_proj` + `x_proj` → `in_proj`). Anything trained directly in PEFT
  (i.e. our own training/) does not incur this loss — the recipe's
  theoretical ceiling for us is ~0.81 (Unsloth's measured number) with full
  data, vs 0.85 for the fork-the-winner baseline.

Facts established **this iteration** (full numbers in `ITERATION_REPORT.md`):

- **SFT is bankrupt for cryptarithm.** A holdout *canary* (train on a split,
  eval the rest) showed negative transfer — canary 2/30 vs baseline 3/30 — even
  though train_loss dropped 95→13. Each puzzle is an independent random
  letter→digit cipher, so memorization doesn't generalize. opd-v1 and opd-v2
  both stayed at LB 0.85.
- **The dominant cryptarithm failure cluster is unsolvable by our solver.** Of
  643 unsolved, the "F bucket" (digit cipher, harder op) is 540 (84%). A 21-op
  DSL solver gets 0/100 on it across default/wide base range and `unique=False`.
  Wrong solver family, not a missing operator — auto-generating correct CoT is
  a dead end for 84% of misses.
- **GRPO can't start on the Huikang adapter.** Its SFT collapsed policy entropy
  to align with the T=0 grader; at T=0.95 G=8 it emits only 1.6–1.8 distinct
  sequences/group, so intra-group advantage ≈ 0 → no gradient. The gatekeeper to
  revive RL is the **base** model (not the adapter): G=16, T=1.2, top_k=64, and
  measure rollout diversity *before* training (see `ITERATION_REPORT.md` §5).
- **vLLM 0.17.1 `SamplingParams(seed=N, n>1)` kills diversity** — all n
  completions come back near-identical. `training/rl/rollout.py` deliberately
  omits the seed.
- **30B GRPO OOMs on a single 4×L40s node** (~35 GB/rank static ceiling on 44 GB
  cards); reducing `max_length` doesn't help. Use 2 nodes / 8-way FSDP
  (`fsdp_multinode_config.yaml`, smoke v8).
- **Kaggle token is read from `$KAGGLE_API_TOKEN`** — the submit/poll scripts no
  longer hardcode it. The literal token still exists in pre-`1cbedae` git
  history; the user opted not to scrub it (only requires it be absent from HEAD).

## User feedback rules

- **Kaggle quota discipline**: 5 submissions/day reset at UTC midnight. Always
  `kaggle competitions submissions ...` to check for same-day duplicates
  before submitting. `scripts/submit_checkpoint.sh` enforces this.
- **Minimum 60% epoch before submitting**: don't burn quota on under-trained
  checkpoints. For a 1-epoch run with `effective_batch=64` on 6355 samples
  (~100 steps), that's `step ≥ 60`.
- **Don't auto-submit destructive actions**: confirm with the user before
  cancelling SLURM jobs, deleting checkpoints, or any quota-burning operation.

## How to work in this repo

- All paths in scripts are absolute (`/scratch/jwj/NemotronFT/...`) — they
  assume the project lives at that exact path. Don't rewrite to relative paths.
- Don't use `pip install --user`; use one of the existing venvs.
- For long training jobs, prefer 12h `sbatch` windows. SIGUSR1 chain-resume
  handles the multi-segment case.
- `output/` is gitignored — it's safe to mutate freely, but `huikang-baseline*`,
  `self-baseline*`, and the `opd-v1`/`opd-v2` adapters are valuable artifacts
  that should be preserved.
- The `runs/` dirs under `training/rl/` and `training/opd/` are gitignored (heavy
  safetensors / rollout JSONL / SLURM logs). Their results are recorded in
  `ITERATION_REPORT.md`, so the artifacts are reproducible byproducts.
- `resource/` is also gitignored. It contains Huikang's repo clone, his
  reference notebooks, and the Discord chat dump — large reference material,
  not source-of-truth code. `NeMo-RL` is cloned there for reference only (not
  installed): it's the only first-party toolkit supporting Nemotron LoRA-GRPO,
  the fallback if the hand-rolled GRPO needs replacing.
