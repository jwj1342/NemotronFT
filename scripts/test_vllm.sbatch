#!/bin/bash
#SBATCH --job-name=test-vllm
#SBATCH --account=def-zhouyang
#SBATCH --time=00:30:00
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

echo "=== vLLM Quick Test ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'torch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

python $PROJECT_DIR/scripts/test_vllm.py

echo ""
echo "=== Done at $(date) ==="
