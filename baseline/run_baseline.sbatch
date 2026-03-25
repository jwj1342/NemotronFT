#!/bin/bash
#SBATCH --job-name=nemotron-baseline
#SBATCH --account=def-zhouyang
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
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
python $PROJECT_DIR/baseline/train.py \
    --output_dir   $PROJECT_DIR/output/debug-lora \
    --max_steps    10 \
    --batch_size   1 \
    --grad_acc     1 \
    --lr           2e-4 \
    --max_length   512 \
    --lora_rank    32 \
    --lora_alpha   64

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
    $PROJECT_DIR/output/debug-lora \
    $PROJECT_DIR/output

echo ""
echo "Done at: $(date)"
echo "Submission zip: $PROJECT_DIR/output/submission.zip"
