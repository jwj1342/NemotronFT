#!/bin/bash
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=gen_cot_data
#SBATCH --output=/scratch/jwj/NemotronFT/training/opd/runs/gen_cot_%j.log

set -euo pipefail
cd /scratch/jwj/NemotronFT

module purge
module load python/3.11.5
source venv/bin/activate && unset PIP_PREFIX

echo "Starting CoT data generation at $(date)"
python training/opd/gen_cipher_cot.py
echo "Done at $(date)"
