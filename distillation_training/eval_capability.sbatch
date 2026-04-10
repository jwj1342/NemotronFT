#!/bin/bash
#SBATCH --job-name=cap-audit
#SBATCH --account=def-zhouyang
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Usage:
#   sbatch --export=CATEGORIES="bit_ops cipher" eval_capability.sbatch
#   CATEGORIES env var: space-separated list of categories to evaluate

module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Capability Audit ==="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Categories : ${CATEGORIES:-ALL}"
echo "Started    : $(date)"
echo "========================"

CATS_ARG=""
if [[ -n "$CATEGORIES" ]]; then
    CATS_ARG="--categories $CATEGORIES"
    SUFFIX=$(echo "$CATEGORIES" | tr ' ' '_')
else
    SUFFIX="all"
fi

python $PROJECT_DIR/distillation_training/eval_capability.py \
    --checkpoint_dir $PROJECT_DIR/output/best-distill-v1-step160-kaggle070 \
    --samples_per_class 0 \
    --max_tokens 7680 \
    --gpu_memory_utilization 0.90 \
    $CATS_ARG \
    --output_file $PROJECT_DIR/output/capability_audit_${SUFFIX}.jsonl

echo ""
echo "Done at: $(date)"
