#!/bin/bash
#SBATCH --job-name=dumpcand
#SBATCH --account=aip-zhouyang
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

set -u
cd /scratch/jwj/NemotronFT
module purge && module load python/3.11.5
source venv/bin/activate; unset PIP_PREFIX
export PYTHONUNBUFFERED=1
python training/opd/solvers/dump_candidates.py
