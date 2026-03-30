#!/bin/bash
#SBATCH --job-name=cot-test
#SBATCH --account=def-zhouyang
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --exclude=g7,g26,g37
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
VENV_DIR=$PROJECT_DIR/venv

source $PROJECT_DIR/venv/bin/activate && unset PIP_PREFIX && \
    pip uninstall -y torch mamba-ssm causal-conv1d 2>/dev/null || true

bash $PROJECT_DIR/scripts/setup_training_env.sh

source $VENV_DIR/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TEST RUN: CoT Training ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Started: $(date)"

# Quick test: 5 training steps, 1 eval round at step 5
# batch_size=1 to avoid OOM with 2048-length sequences
python $PROJECT_DIR/cot_training/train.py \
    --output_dir       $PROJECT_DIR/output/cot-test \
    --checkpoint_dir   $PROJECT_DIR/output/baseline-lora/checkpoint-100 \
    --max_steps        5 \
    --batch_size       1 \
    --grad_acc         2 \
    --lr               5e-5 \
    --max_length       2048 \
    --neftune_alpha    5.0 \
    --eval_steps       5 \
    --eval_samples_per_class 2 \
    --eval_batch_size  4 \
    --tensorboard_dir  /home/jwj/tensorboard_logs

echo ""
echo "Test exit code: $?"
echo "Done at: $(date)"
