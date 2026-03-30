#!/bin/bash
#SBATCH --job-name=eval-cot-7680
#SBATCH --account=def-zhouyang
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/eval-cot-7680-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/eval-cot-7680-%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Started  : $(date)"
echo "================"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT=/scratch/jwj/NemotronFT

# ── Setup env (reuse project venv, same as training) ─────────────────────────
module purge
module load python/3.11.5
source "$PROJECT/venv/bin/activate"
unset PIP_PREFIX

echo ""
echo "=== Evaluating CoT checkpoint-120 (max_new_tokens=7680, Kaggle-aligned) ==="
echo ""

python "$PROJECT/cot_training/eval_cot.py" \
    --checkpoint_dir "$PROJECT/output/cot-lora/checkpoint-120" \
    --max_new_tokens 7680 \
    --eval_samples_per_class 5 \
    --batch_size 1 \
    --output_file "$PROJECT/output/eval_results_cot_checkpoint-120_7680.jsonl"

echo ""
echo "============================================================"
echo "=== Evaluating baseline checkpoint-100 (same token budget) ==="
echo "============================================================"
echo ""

python "$PROJECT/cot_training/eval_cot.py" \
    --checkpoint_dir "$PROJECT/output/baseline-lora/checkpoint-100" \
    --max_new_tokens 7680 \
    --eval_samples_per_class 5 \
    --batch_size 1 \
    --output_file "$PROJECT/output/eval_results_baseline_checkpoint-100_7680.jsonl"

echo ""
echo "=== All evaluations complete ==="
echo "Finished: $(date)"
