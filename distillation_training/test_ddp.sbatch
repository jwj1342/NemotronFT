#!/bin/bash
#SBATCH --job-name=test-ddp
#SBATCH --account=def-zhouyang
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== DDP Test Job ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "================"

# ── GPU monitor in background (every 10s) ────────────────────────────────────
(
    while true; do
        echo "--- GPU utilization at $(date +%H:%M:%S) ---"
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total \
            --format=csv,noheader
        sleep 10
    done
) &
GPU_MON_PID=$!

# ── Test: 10 steps only, output to a throwaway dir ───────────────────────────
torchrun --nproc_per_node=2 $PROJECT_DIR/distillation_training/train.py \
    --output_dir                $PROJECT_DIR/output/test-ddp \
    --checkpoint_dir            $PROJECT_DIR/output/best-baseline-step100-kaggle064 \
    --regular_samples_per_class 600 \
    --exclude_categories        equation_symbolic \
    --max_length                2048 \
    --boxed_loss_weight         1.0 \
    --reasoning_loss_weight     1.0 \
    --max_steps                 10 \
    --batch_size                2 \
    --grad_acc                  8 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --save_steps                999 \
    --save_total_limit          1 \
    --save_only_model

TRAIN_EXIT=$?
kill $GPU_MON_PID 2>/dev/null

echo ""
echo "=== Train exit code: $TRAIN_EXIT ==="
echo "=== Done at $(date) ==="
