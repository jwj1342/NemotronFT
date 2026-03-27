#!/bin/bash
#SBATCH --job-name=nemotron-baseline
#SBATCH --account=def-zhouyang
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --exclude=g7,g26,g37
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
VENV_DIR=$PROJECT_DIR/venv

# Uninstall any previously installed torch to avoid ABI mismatch,
# then reinstall with correct version resolved by mamba-ssm.
source $PROJECT_DIR/venv/bin/activate && unset PIP_PREFIX && \
    pip uninstall -y torch mamba-ssm causal-conv1d 2>/dev/null || true

bash $PROJECT_DIR/scripts/setup_training_env.sh

source $VENV_DIR/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1   # model already downloaded, skip network calls
export HF_DATASETS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=0

echo "=== Job info ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Started  : $(date)"
echo "================"

# ── Training ──────────────────────────────────────────────────────────────────
# Stratified sampling: 200 samples × 6 task categories = 1200 total.
# 1 epoch, effective batch = batch_size × grad_acc = 1 × 8 = 8.
# → ~150 optimiser steps; warmup = 5% ≈ 8 steps.
python $PROJECT_DIR/baseline/train.py \
    --output_dir       $PROJECT_DIR/output/baseline-lora \
    --max_steps        -1 \
    --samples_per_class 400 \
    --batch_size       4 \
    --grad_acc         4 \
    --lr               2e-4 \
    --max_length       512 \
    --lora_rank        32 \
    --lora_alpha       16 \
    --neftune_alpha    5.0 \
    --eval_steps       25 \
    --eval_samples_per_class 5 \
    --eval_batch_size  4 \
    --tensorboard_dir  /home/jwj/tensorboard_logs

TRAIN_EXIT=$?
echo ""
echo "Training exit code: $TRAIN_EXIT"

if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Training failed."
    exit $TRAIN_EXIT
fi

# ── Package submission ────────────────────────────────────────────────────────
echo ""
echo "=== Packaging submission.zip ==="
bash $PROJECT_DIR/scripts/build_submission.sh \
    $PROJECT_DIR/output/baseline-lora \
    $PROJECT_DIR/output

echo ""
echo "Done at: $(date)"
echo "Submission zip: $PROJECT_DIR/output/submission.zip"
