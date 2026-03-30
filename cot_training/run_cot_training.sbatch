#!/bin/bash
#SBATCH --job-name=nemotron-cot-mixed
#SBATCH --account=def-zhouyang
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
VENV_DIR=$PROJECT_DIR/venv

source $VENV_DIR/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job info ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Started  : $(date)"
echo "================"

# ── Auto-link TensorBoard logs ───────────────────────────────────────────────
# Trainer writes events to {output_dir}/runs/{timestamp}/.
# Create symlinks in ~/tensorboard_logs/ so TensorBoard can discover new runs
# immediately.  A background loop checks for new run dirs every 30s.
TB_LINK_DIR=/home/jwj/tensorboard_logs
TB_RUN_DIR=$PROJECT_DIR/output/cot-lora/runs
mkdir -p "$TB_LINK_DIR" "$TB_RUN_DIR"
(
    while true; do
        for d in "$TB_RUN_DIR"/*/; do
            [ -d "$d" ] || continue
            name="$(basename "$d")"
            [ -L "$TB_LINK_DIR/$name" ] || ln -sf "$d" "$TB_LINK_DIR/$name"
        done
        sleep 30
    done
) &
TB_LINK_PID=$!

# ── Mixed CoT Training ──────────────────────────────────────────────────────
# Data: 7 categories × 500 regular + ~1500 CoT (bit_ops, equation_symbolic, cipher)
# Total: ~5000 samples, shuffled.
# Effective batch = 1 × 16 = 16 → ~313 steps/epoch × 2 = ~625 steps total.
# Loss: boxed tokens 5× weight vs reasoning tokens 1×.
python $PROJECT_DIR/cot_training/train.py \
    --output_dir                $PROJECT_DIR/output/cot-lora \
    --checkpoint_dir            $PROJECT_DIR/output/baseline-lora/checkpoint-100 \
    --regular_samples_per_class 500 \
    --eval_samples_per_class    5 \
    --max_length                2048 \
    --boxed_loss_weight         5.0 \
    --reasoning_loss_weight     1.0 \
    --num_epochs                2 \
    --batch_size                1 \
    --grad_acc                  16 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --eval_steps                200 \
    --eval_max_new_tokens       7680 \
    --eval_batch_size           2 \
    --save_steps                50 \
    --tensorboard_dir           /home/jwj/tensorboard_logs

TRAIN_EXIT=$?
kill $TB_LINK_PID 2>/dev/null  # stop symlink watcher
echo ""
echo "Training exit code: $TRAIN_EXIT"

if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Training failed."
    exit $TRAIN_EXIT
fi

# ── Package submission ────────────────────────────────────────────────────────
echo ""
echo "=== Packaging submission.zip ==="
bash $PROJECT_DIR/scripts/build_submission.sh \
    $PROJECT_DIR/output/cot-lora \
    $PROJECT_DIR/output

echo ""
echo "Done at: $(date)"
echo "Submission zip: $PROJECT_DIR/output/submission.zip"
