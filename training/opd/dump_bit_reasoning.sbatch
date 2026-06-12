#!/bin/bash
#SBATCH --job-name=bit-dump
#SBATCH --account=aip-zhouyang
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/bit-dump-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/bit-dump-%j.err

# Cheap gate for the brevity/length-insurance experiment: dump full bit-correct
# reasoning chains (median 6692/7680 tok) to judge if they are compressible
# (habitual narration) or essential bit-by-bit computation. Decides whether to
# invest in the full brevity training run.
set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"; mkdir -p logs

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT/venv-vllm/bin/activate"
unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home

echo "=== bit reasoning dump === job $SLURM_JOB_ID $(hostname) $(date)"
python3 "$PROJECT/training/opd/dump_bit_reasoning.py" --n 8 --tp 4
echo "=== done $(date) ==="
