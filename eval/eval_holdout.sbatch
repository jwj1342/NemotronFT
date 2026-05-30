#!/bin/bash
#SBATCH --job-name=eval-holdout
#SBATCH --account=aip-zhouyang
#SBATCH --time=00:45:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Inputs (set via env or default):
#   EVAL_ADAPTER  — adapter dir to eval (omit for baseline)
#   EVAL_N        — number of holdout puzzles (default 30)
#   EVAL_OUT      — output jsonl path

set -u
PROJECT=/scratch/jwj/NemotronFT
EVAL_N=${EVAL_N:-30}
EVAL_OUT=${EVAL_OUT:-$PROJECT/training/opd/runs/canary/eval_${EVAL_ADAPTER##*/}.jsonl}
ADAPTER_ARG=""
if [[ -n "${EVAL_ADAPTER:-}" ]]; then
    ADAPTER_ARG="--adapter $EVAL_ADAPTER"
fi

cd "$PROJECT"
module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source venv-vllm/bin/activate; unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== eval_holdout  adapter=${EVAL_ADAPTER:-BASELINE}  n=$EVAL_N  out=$EVAL_OUT  $(date) ==="
python eval/eval_holdout.py $ADAPTER_ARG --n "$EVAL_N" --out "$EVAL_OUT" --tensor_parallel_size 4
echo "=== done $(date) ==="
