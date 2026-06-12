#!/bin/bash
#SBATCH --job-name=bit-holdout-eval
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error

# T=0 holdout net-gain eval: warmstart vs GRPO-trained on the 100 disjoint bit holdout.
set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"; mkdir -p logs
module purge && module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT/venv-vllm/bin/activate"; unset PIP_PREFIX
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home
echo "=== bit holdout eval === job $SLURM_JOB_ID $(hostname) $(date)"
python3 "$PROJECT/training/rl/eval_bit_holdout.py" --tp 4
echo "=== done $(date) ==="