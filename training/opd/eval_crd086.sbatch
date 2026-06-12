#!/bin/bash
#SBATCH --job-name=crd086-eval
#SBATCH --account=aip-zhouyang
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error
set -u
P=/scratch/jwj/NemotronFT
cd "$P"
module purge; module load python/3.11.5 gcc/12.3 opencv/4.11.0
source $P/venv-vllm/bin/activate; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
echo "=== crd-086 crypt-recovery eval (well-posed holdout, T=0) $(date) ==="
# 0.86 base (crypt baseline) + 3 crd checkpoints; --eval_base = raw BASE (the 6% probe ref)
python3 $P/training/opd/eval_crd_probe.py \
    --holdout $P/data/crd_holdout.jsonl \
    --checkpoints $P/output/kienngx-086-renamed \
    --checkpoints $P/output/crd-086/clean-40 \
    --checkpoints $P/output/crd-086/clean-80 \
    --checkpoints $P/output/crd-086/clean-120 \
    --eval_base \
    --out $P/output/crd-086/holdout_eval.json \
    --tensor_parallel_size 4
echo "exit=$? $(date)"
