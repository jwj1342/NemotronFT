#!/bin/bash
#SBATCH --job-name=eval-cot-mixed
#SBATCH --account=def-zhouyang
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Eval: CoT mixed checkpoint-150 vs baseline checkpoint-100 ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Eval script from earlier conversation — aligns with Kaggle (max_new_tokens=7680)
# Evaluate both checkpoints for direct comparison.

# 1. CoT mixed checkpoint-150
echo "=========================================="
echo "=== Evaluating CoT mixed checkpoint-150 ==="
echo "=========================================="
python $PROJECT_DIR/cot_training/eval_cot.py \
    --checkpoint_dir $PROJECT_DIR/output/cot-lora/checkpoint-150 \
    --output_file    $PROJECT_DIR/output/eval_results_cot_mixed_checkpoint-150.jsonl \
    --train_samples_per_class 500 \
    --max_new_tokens 256 \
    --eval_samples_per_class 3 \
    --batch_size     1

echo ""
echo "=========================================="
echo "=== Evaluating baseline checkpoint-100 ==="
echo "=========================================="
python $PROJECT_DIR/cot_training/eval_cot.py \
    --checkpoint_dir $PROJECT_DIR/output/baseline-lora/checkpoint-100 \
    --output_file    $PROJECT_DIR/output/eval_results_baseline_checkpoint-100.jsonl \
    --train_samples_per_class 500 \
    --max_new_tokens 256 \
    --eval_samples_per_class 3 \
    --batch_size     1

echo ""
echo "=== Done at $(date) ==="
