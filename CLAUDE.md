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

**Current state (2026-06-04): LB 0.86 achieved (up from 0.85).** The breakthrough
was NOT more training — it was using a **losslessly-converted public adapter** +
**notebook submission**. The 0.86 adapter = `kienngx/nemotron-nano-30b-trained/
triton/tinker-adapter` = Huikang's original adapter converted via asalhi's
`tinker_cookbook.build_lora_adapter` (lossless), vs our `convert_tinker_adapter.py`
which loses ~25% singular mass via hand-rolled SVD → only 0.85. The 3.5GB adapter
exceeds the ~2GB file-upload limit, so it must be submitted via a **notebook**
(`kaggle kernels push` + `competitions submit -k <kernel> -v 1 -f submission.zip`),
not `submit -f`. Full method in memory [[project_086_achieved]]; submitted as
`jwj1342/nemotron-tinker-adapter-submit`.

Earlier this iteration, three "train past 0.85" routes (multi-category SFT
distillation, programmatic cryptarithm solver, GRPO) and a bit-RFT pilot all
netted ≤0 — see `ITERATION_REPORT.md`. The lesson: 0.85→0.86 was a
conversion/packaging fix, not a modeling problem. 0.87 (14 teams) would need
genuinely new modeling (RL / better data / better conversion).
Per-experiment record in `ITERATION_REPORT.md`; strategy in `ROADMAP.md`; RL
reference in `RL_ARCHITECTURE.md`.

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
- **Any hand-written solver is weaker than the model — "solver → CoT → SFT" is
  dead (06-02).** Even with the *correct* combo framework (from Kaggle #684283),
  the solver scores 59.9% (bit) / 39.8% (numeric_deduce) / 0% (numeric_guess) /
  ~4% (cipher-digit) vs the model's 87.5 / 90.9 / 16.2 / 7%. Cross-tab vs the
  baseline eval: training on solver output would teach +58 but mis-teach −500 on
  bit. So the solvers (`training/opd/solvers/*`) are only useful as *verifiers*,
  never as generators. Failure is under-constraint, not a bug: 77% of bit
  failures differ by 1–2 bits, truncation is 3%. See `ITERATION_REPORT.md` §7 and
  the memory files `project_solver_cannot_beat_model` / `project_cryptarithm_sft_ceiling`.
- **RFT/STaR self-training was tried and gives ~0 net gain (06-03 pilot) — 0.85
  is the framework ceiling.** pass@k looked promising (bit pass@16=70%, the model
  self-samples correct CoT), but the bit RFT pilot netted ~0: checkpoint-5 was
  +3 on the holdout subset, but extrapolated to the full set the recovery
  (8.3%×200≈+17) is eaten by regression (1.4%×1402 already-correct≈−20). **Root
  cause = prior coupling**: bit failures and successes share one ambiguous-bit
  frequency prior, so RFT pulling failures regresses the (much larger) correct
  set. Long-CoT SFT overfits fast (sweet spot ~0.4 epoch, then non-bit sentinel
  collapses). All three "salvage the misses" routes (SFT-distill / solver /
  RFT) now measure net ≤0. **+1pp needs a different framework** — a stronger
  base adapter (none available; Huikang 0.85 is public SOTA) or inference-time
  search (grader confirmed CLOSED: submission is a fixed adapter, single greedy
  pass — no multi-sampling/TTT/best-of-N). Full record in `ITERATION_REPORT.md`
  §8, `ROADMAP.md` §0″, memory [[project_recoverable_ceiling]].
- **Methodology rule (hard-won, 3-way swing): salvageability/ceiling can ONLY be
  judged by holdout net-gain extrapolated to the full set.** Static-solver
  estimates (said bit unsalvageable), pass@k upper bounds (said bit +0.2~0.7pp),
  and holdout-subset accuracy ALL systematically mislead. Each was corrected by
  the next-stronger evidence; only the T=0 holdout three-metric eval (recover −
  regress − sentinel-drift) was decisive.
- **RL is technically feasible on bit (unlike cryptarithm), but unlikely to beat
  RFT.** pass@k shows bit has T>0 diversity (pass@1=20%, nonzero reward variance
  → GRPO has gradient), so the §"GRPO can't start" blocker is cryptarithm-only
  (entropy collapse there). But RL faces the SAME prior-coupling limit as RFT;
  its only possible edge is the KL anchor controlling regression more gently than
  MLE-SFT — unverified, high engineering cost, and RFT already mapped the ceiling
  at ~0.
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
