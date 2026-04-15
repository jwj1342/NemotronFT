#!/bin/bash
#SBATCH --job-name=eval-distill
#SBATCH --account=def-zhouyang
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

module purge
module load python/3.11.5 opencv/4.13.0

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv-vllm/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "=== vLLM Evaluation ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Auto-discover and evaluate ALL checkpoints under distill-lora/
# 40 samples per class × 7 classes = 280 samples
# max_tokens=7680 (Kaggle aligned)
python $PROJECT_DIR/distillation_training/eval_vllm.py \
    --scan_dir                  $PROJECT_DIR/output/distill-lora-v2 \
    --max_tokens                7680 \
    --eval_samples_per_class    40 \
    --train_samples_per_class   600

echo ""
echo "=== Done at $(date) ==="
