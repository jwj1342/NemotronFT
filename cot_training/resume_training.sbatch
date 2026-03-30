#!/bin/bash
#SBATCH --job-name=cot-resume
#SBATCH --account=def-zhouyang
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

module purge
module load python/3.11.5

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Resume CoT mixed training from checkpoint-150 ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ── Auto-link TensorBoard logs ───────────────────────────────────────────────
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

# ── Resume training ─────────────────────────────────────────────────────────
# Resume from step 150. Original config: 2 epochs = ~626 steps.
# Changing to 1 epoch = ~313 steps. Trainer resumes from step 150,
# continues to step 313 (remaining ~163 steps).
# Optimizer, scheduler, and step count are restored from checkpoint.
python $PROJECT_DIR/cot_training/train.py \
    --output_dir                $PROJECT_DIR/output/cot-lora \
    --checkpoint_dir            $PROJECT_DIR/output/cot-lora/checkpoint-150 \
    --resume_from_checkpoint    $PROJECT_DIR/output/cot-lora/checkpoint-150 \
    --regular_samples_per_class 500 \
    --eval_samples_per_class    5 \
    --max_length                2048 \
    --boxed_loss_weight         5.0 \
    --reasoning_loss_weight     1.0 \
    --num_epochs                1 \
    --batch_size                1 \
    --grad_acc                  16 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --eval_steps                200 \
    --eval_max_new_tokens       256 \
    --eval_batch_size           1 \
    --save_steps                50 \
    --tensorboard_dir           /home/jwj/tensorboard_logs

TRAIN_EXIT=$?
kill $TB_LINK_PID 2>/dev/null
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
