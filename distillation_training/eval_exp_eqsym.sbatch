#!/bin/bash
#SBATCH --job-name=eval-eqsym
#SBATCH --account=def-zhouyang
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Evaluate ALL exp-eqsym checkpoints on equation_symbolic (dual-cohort).

module purge
module load python/3.11.5 opencv/4.13.0

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv-vllm/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Eval exp-eqsym checkpoints ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo ""

CKPT_DIR=$PROJECT_DIR/output/exp-eqsym
OUT_DIR=$PROJECT_DIR/output/eval-exp-eqsym
mkdir -p $OUT_DIR

# Only evaluate the final checkpoint (epoch 3)
for ckpt in $CKPT_DIR/checkpoint-102; do
    [ -d "$ckpt" ] || continue
    step=$(basename $ckpt | sed 's/checkpoint-//')
    echo ""
    echo "=================================================="
    echo "Evaluating checkpoint-${step} ..."
    echo "=================================================="
    python $PROJECT_DIR/distillation_training/eval_capability.py \
        --checkpoint_dir $ckpt \
        --samples_per_class 0 \
        --max_tokens 7680 \
        --gpu_memory_utilization 0.90 \
        --categories equation_symbolic \
        --output_file $OUT_DIR/step_${step}.jsonl
done

echo ""
echo "=== All checkpoints evaluated. Done at $(date) ==="
