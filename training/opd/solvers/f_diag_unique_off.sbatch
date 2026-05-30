#!/bin/bash
#SBATCH --job-name=fduo
#SBATCH --account=aip-zhouyang
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"
module purge && module load python/3.11.5
source venv/bin/activate; unset PIP_PREFIX
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
python training/opd/solvers/f_diag_unique_off.py
