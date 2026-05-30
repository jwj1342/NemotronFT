#!/bin/bash
#SBATCH --account=aip-zhouyang
#SBATCH --job-name=cryptarithm-probe
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/scratch/jwj/NemotronFT/training/opd/solvers/runs/probe_%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/training/opd/solvers/runs/probe_%j.err

set -euo pipefail

cd /scratch/jwj/NemotronFT

module purge
module load python/3.11.5
source /scratch/jwj/NemotronFT/venv/bin/activate
unset PIP_PREFIX

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1   # let multiprocessing.Pool own all parallelism

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
echo "started: $(date)"

python3 training/opd/solvers/probe_v2.py

echo "finished: $(date)"
