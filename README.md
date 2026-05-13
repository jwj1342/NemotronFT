# NemotronFT

Fine-tuning **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`** for the [NVIDIA
Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

## Strong baseline (current SOTA in this repo)

| Adapter | Public LB |
|---|---|
| `output/huikang-baseline-converted/` — Huikang's published winning adapter, SVD-converted (rank 32, 75.77% singular mass preserved) | **0.85** |
| `output/self-baseline/` — our own SFT on Huikang's CoT corpus (6355 of 9500 problems), 1 epoch via FSDP+PEFT | 0.76 |
| Base model (no adapter) | ~0.45 |

The 0.85 adapter is the reference we plan to *push past*; the self-trained 0.76
shows the full Unsloth recipe works end-to-end on our hardware (4×L40s, FSDP).

## Why 0.85 needs SVD conversion

Huikang trained on **Tinker**, whose LoRA layout differs from Kaggle's vLLM
runtime in three places (see his
[writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)):

1. Mamba `in_proj` is trained as separate `gate_proj` + `x_proj` LoRA pairs.
   Submission needs a single `in_proj` LoRA, so we **SVD-merge** the two rank-32
   pairs back to rank 32 (this loses ~25% of singular mass — Huikang's
   "training-serving misalignment").
2. MoE experts are stored as fused `experts.w1` / `experts.w2` tensors with a
   single shared LoRA; submission needs **per-expert `up_proj` / `down_proj`**
   LoRAs, so we broadcast the shared LoRA into 128 copies (lossless).
3. Adapter keys use the prefix `base_model.model.model.*`; submission needs
   `base_model.model.backbone.*` (lossless rename).

All three steps are implemented in `training/convert_tinker_adapter.py`.

## Repo layout

```
NemotronFT/
├── training/                    # SFT pipeline (mirrors the Unsloth 0.81 recipe)
│   ├── build_cot_csv.py         # Decode Huikang corpus.jsonl → text CSV
│   ├── convert_tinker_adapter.py # Tinker adapter → Kaggle-format adapter (SVD)
│   ├── dataset.py               # Chat-template SFT dataset (enable_thinking=True)
│   ├── train.py                 # FSDP LoRA training script
│   ├── train.sbatch             # 12h SLURM job with SIGUSR1 chain-resume
│   ├── train_with_cot.csv       # 6355 decoded (prompt, answer, cot, type) rows
│   └── fsdp_config.yaml         # FULL_SHARD + TRANSFORMER_BASED_WRAP
├── eval/
│   ├── extract.py               # extract_final_answer / verify / BOXED_INSTRUCTION
│   ├── eval_vllm.py             # vLLM eval matched to grader: T=0, max_tokens=7680
│   └── eval_vllm.sbatch
├── scripts/
│   ├── build_submission.sh      # Pack a checkpoint dir into submission.zip
│   ├── build_all_submissions.sh # Pack every checkpoint-N + final adapter
│   ├── submit_checkpoint.sh     # Build + duplicate-check + Kaggle CLI submit
│   ├── submit.sh                # Bare kaggle competitions submit
│   ├── download_model.sh        # Fetch the base Nemotron weights
│   └── setup_training_env.sh
├── data/by_category/            # Per-category eval splits (gitignored)
├── output/                      # All adapter checkpoints (gitignored)
│   ├── huikang-baseline/                 # Raw Tinker tarball + extracted dir
│   ├── huikang-baseline-converted/       # 0.85 strong baseline ready to submit
│   ├── self-baseline/                    # Our SFT run (0.76)
│   └── self-baseline-submissions/        # Pre-built zips per checkpoint
├── resource/                    # Reference notebooks, Huikang repo (gitignored)
├── model/                       # Base Nemotron weights (gitignored)
├── venv/  venv-vllm/            # Python envs (gitignored)
└── logs/                        # SLURM stdout/stderr (gitignored)
```

## Quick start

### Submit the strong baseline (~5 min)

```bash
bash scripts/submit_checkpoint.sh \
    output/huikang-baseline-converted \
    "strong baseline: Huikang adapter, SVD-converted"
```

### Convert a fresh Tinker checkpoint

```bash
python training/convert_tinker_adapter.py \
    --in_dir  output/<some-tinker-adapter> \
    --out_dir output/<some-tinker-adapter>-converted
```

### Run local vLLM eval (grader-aligned)

```bash
sbatch eval/eval_vllm.sbatch
# Or pass a specific checkpoint:
CHECKPOINT_DIR=output/huikang-baseline-converted sbatch eval/eval_vllm.sbatch
```

### Train from scratch (FSDP LoRA, ~13h)

```bash
# 1. Build the CoT CSV from Huikang's corpus (one-time, ~3 min).
python training/build_cot_csv.py

# 2. Submit the chained training job (12h walltime, auto-resumes via SIGUSR1).
sbatch training/train.sbatch
```

Checkpoints land in `output/sft/`; the final adapter is saved at the same path
root once 1 epoch completes.

## Grader-exact inference config

Confirmed by reading the official metric's `score()` defaults in Huikang's
tinker-submission-notebook:

```python
temperature      = 0.0    # greedy (NOT the 1.0 the public metric.py shows)
top_p            = 1.0
max_tokens       = 7680
max_model_len    = 8192
max_lora_rank    = 32
gpu_memory_utilization = 0.85
```

`eval/eval_vllm.py` mirrors these exactly. Any submission/eval that diverges
will systematically underestimate the model.

## Critical LoRA detail: Mamba layers

Nemotron-3-Nano-30B-A3B is a hybrid Mamba–Transformer (`nemotron_h`). Mamba
layers expose `in_proj` and `out_proj`; **these MUST be in `target_modules`**
or the SSM circuits stay frozen. Our `training/train.py` uses:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "in_proj", "out_proj",                    # Mamba SSM    (do not omit)
    "up_proj", "down_proj",                   # MoE expert MLPs
    "lm_head",
]
```

Other hyperparameters that matter (all from the published 0.81 Unsloth notebook):
LoRA `r=alpha=32`, `dropout=0`, `lr=2e-4` linear (no warmup), AdamW `beta2=0.95`,
`weight_decay=0`, `max_grad_norm=1e9` (effectively off), 1 epoch, effective
batch 64, stratified sampling by category.

## What to try next (from 0.85 → 0.87+)

1. **Get the full 9500-problem CoT data** — our `train_with_cot.csv` only has
   6355 because Huikang's GitHub snapshot omits ~3000 corpus files. The full
   set is on Kaggle as `huikang/huikang-nemotron-repository-snapshot`.
2. **Fine-tune *on top of* the 0.85 adapter** — `training/train.py` already
   accepts `--resume_adapter_dir` for exactly this. Start from
   `output/huikang-baseline-converted` and continue with lr ≤ 1e-5 on fresh
   data to avoid overwriting the conversion.
3. **Rewrite the bit-manipulation CoT** for the ~240 problems Huikang couldn't
   solve (combined SHL+SHR cases that overflow the 7680-token budget). Donald
   Galliano's reverse-engineering posts on Kaggle Discussion document the
   structure.
4. **Rewrite the cryptarithm CoT** — Huikang gets only ~8% here, so this is
   the single biggest absolute headroom (~+0.06 on LB if perfectly solved).
