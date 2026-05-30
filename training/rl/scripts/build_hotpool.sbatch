#!/bin/bash
#SBATCH --job-name=hotpool
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Build hot-pool: baseline Huikang adapter @ T=0 greedy on
# cryptarithm_deduce + cryptarithm_guess + equation_numeric_guess (959 prompts).

set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source venv-vllm/bin/activate; unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== build_hotpool  $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python training/rl/build_hotpool.py --tensor_parallel_size 4

echo "=== done $(date) ==="
