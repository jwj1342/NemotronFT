#!/bin/bash
#SBATCH --job-name=exp-eqsym
#SBATCH --account=def-zhouyang
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# ── Diagnostic experiment: can the model learn equation_symbolic at all? ─────
# Starting from baseline (not v1-step160) to avoid the "0% local minimum".
# Uses only 532 Flash-COT-Filtered samples for equation_symbolic, excludes all
# other categories. If unseen accuracy > 5% after this, the data is learnable.

module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Equation Symbolic Diagnostic ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "===================================="

# Single-process with model parallelism (balanced across 2 GPUs)
# Exclude all categories except equation_symbolic
# regular_samples_per_class=0 → pure CoT training on 532 filtered samples
python $PROJECT_DIR/distillation_training/train.py \
    --output_dir                $PROJECT_DIR/output/exp-eqsym \
    --checkpoint_dir            $PROJECT_DIR/output/best-baseline-step100-kaggle064 \
    --cot_data_dir              $PROJECT_DIR/data/Flash-COT-Filtered \
    --regular_samples_per_class 0 \
    --exclude_categories        bit_ops,cipher,equation_numeric,gravity,numeral,unit_conv \
    --max_length                2048 \
    --boxed_loss_weight         1.0 \
    --reasoning_loss_weight     1.0 \
    --num_epochs                3 \
    --batch_size                8 \
    --grad_acc                  2 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --save_steps                10 \
    --save_total_limit          -1 \
    --save_only_model

TRAIN_EXIT=$?
echo ""
echo "Training exit code: $TRAIN_EXIT"

if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Training failed."
    exit $TRAIN_EXIT
fi

echo ""
echo "Checkpoints saved to: $PROJECT_DIR/output/exp-eqsym/"
ls -d $PROJECT_DIR/output/exp-eqsym/checkpoint-* 2>/dev/null
echo ""
echo "Done at: $(date)"
