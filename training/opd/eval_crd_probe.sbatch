#!/bin/bash
#SBATCH --job-name=crd-probe-eval
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/crd-probe-eval-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/crd-probe-eval-%j.err

# T=0 holdout eval for the crd probe (gate #21). venv-vllm, TP=4.
# Evals BASE + every checkpoint in $CRD_OUT on data/crd_holdout.jsonl.
# Decision: <25-30% DEAD | 35%+ continue | 42%+ 0.89-path.

set -u
PROJECT=/scratch/jwj/NemotronFT
OUT=${CRD_OUT:-$PROJECT/output/crd-probe}
HOLDOUT=${CRD_HOLDOUT:-$PROJECT/data/crd_holdout.jsonl}
cd "$PROJECT"

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source $PROJECT/venv-vllm/bin/activate
unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1

echo "=== CRD probe eval ==="
echo "Job $SLURM_JOB_ID $(hostname) $(date) out=$OUT holdout=$HOLDOUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 $PROJECT/training/opd/eval_crd_probe.py \
    --holdout "$HOLDOUT" \
    --scan_dir "$OUT" \
    --eval_base \
    --tensor_parallel_size 4
echo "eval exit=$? $(date)"
