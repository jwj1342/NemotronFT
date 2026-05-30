#!/bin/bash
#SBATCH --job-name=opd-canary
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Canary: can the model be moved at all?
# 100 rows × 5 epoch × grad_acc=8 (eff_batch=8 on 4-GPU FSDP) → ~62 steps.
# LR=5e-5 (5x current). Save every 10 steps for inspection.
# If train_loss drops substantially (e.g. <50 from 187 baseline) → training works,
# we were just under-trained.  If loss stays >120 → fundamental issue with the
# SFT setup, time to escalate (different LR, full re-init, GRPO).

set -u
PROJECT=/scratch/jwj/NemotronFT
CSV=$PROJECT/training/opd/runs/canary/train.csv
OUT=$PROJECT/output/opd-canary
BASE=$PROJECT/output/huikang-baseline-converted

mkdir -p "$OUT" "$PROJECT/logs"
cd "$PROJECT"

echo "============================================================"
echo "Canary  Job:$SLURM_JOB_ID  Node:$(hostname)  $(date)"
echo "  CSV: $CSV    OUT: $OUT"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

module purge && module load python/3.11.5
source venv/bin/activate; unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

accelerate launch \
    --config_file training/fsdp_config.yaml \
    training/train.py \
    --output_dir         "$OUT" \
    --csv_path           "$CSV" \
    --resume_adapter_dir "$BASE" \
    --lora_rank          32 \
    --lora_alpha         32 \
    --lora_dropout       0.0 \
    --max_length         8192 \
    --batch_size         1 \
    --grad_acc           8 \
    --num_epochs         5 \
    --lr                 5e-5 \
    --warmup_steps       0 \
    --save_steps         10 \
    --save_only_model \
    --seed               7

echo ""
echo "=== Canary done at $(date) ==="
