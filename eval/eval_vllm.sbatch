#!/bin/bash
#SBATCH --job-name=eval-vllm
#SBATCH --account=aip-zhouyang
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# vLLM-based local evaluation. Uses grader-aligned settings:
#   max_tokens=7680, max_model_len=8192, temperature=0.0, top_p=1.0
# Pass --checkpoint_dir (one or more) or --scan_dir to discover checkpoint-* dirs.

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv-vllm/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== vLLM evaluation (4xL40s tensor parallel) ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/output/huikang-baseline-converted}"

python $PROJECT_DIR/eval/eval_vllm.py \
    --checkpoint_dir            "$CHECKPOINT_DIR" \
    --max_tokens                7680 \
    --eval_samples_per_class    40 \
    --train_samples_per_class   600 \
    --tensor_parallel_size      4

echo "=== Done at $(date) ==="
