#!/bin/bash
#SBATCH --job-name=crd-dump
#SBATCH --account=aip-zhouyang
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/crd-dump-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/crd-dump-%j.err
set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"
module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source $PROJECT/venv-vllm/bin/activate
unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
python3 $PROJECT/training/opd/dump_crd_gen.py --n 12
