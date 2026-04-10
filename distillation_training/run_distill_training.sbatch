#!/bin/bash
#SBATCH --job-name=distill-v2
#SBATCH --account=def-zhouyang
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# ── Environment ───────────────────────────────────────────────────────────────
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

echo "=== Job info ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Started  : $(date)"
echo "================"

# ── Auto-link TensorBoard logs ───────────────────────────────────────────────
TB_LINK_DIR=/home/jwj/tensorboard_logs
TB_RUN_DIR=$PROJECT_DIR/output/distill-lora-v2/runs
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

# ── Distill-v2: priority data loading + exclude equation_symbolic ───────────
# Continue from baseline-lora/checkpoint-100 (Kaggle 0.64)
# Changes vs v1:
#   - Distillation prompts prioritized into train split (100% utilization)
#   - equation_symbolic excluded (0% across all v1 checkpoints)
#   - ~6600 samples = 3000 regular + 809 upgraded + 3000 CoT (−1600 vs v1)
torchrun --nproc_per_node=2 $PROJECT_DIR/distillation_training/train.py \
    --output_dir                $PROJECT_DIR/output/distill-lora-v2 \
    --checkpoint_dir            $PROJECT_DIR/output/best-baseline-step100-kaggle064 \
    --regular_samples_per_class 600 \
    --exclude_categories        equation_symbolic \
    --max_length                2048 \
    --boxed_loss_weight         1.0 \
    --reasoning_loss_weight     1.0 \
    --num_epochs                2 \
    --batch_size                2 \
    --grad_acc                  8 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --save_steps                20 \
    --save_total_limit          -1 \
    --save_only_model \
    --tensorboard_dir           /home/jwj/tensorboard_logs

TRAIN_EXIT=$?
kill $TB_LINK_PID 2>/dev/null
echo ""
echo "Training exit code: $TRAIN_EXIT"

if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Training failed."
    exit $TRAIN_EXIT
fi

echo ""
echo "Checkpoints saved to: $PROJECT_DIR/output/distill-lora-v2/"
ls -d $PROJECT_DIR/output/distill-lora-v2/checkpoint-* 2>/dev/null

# ── Auto-submit vLLM eval ────────────────────────────────────────────────────
echo ""
echo "=== Auto-submitting vLLM eval job ==="
EVAL_JOB=$(sbatch --parsable $PROJECT_DIR/distillation_training/eval_vllm.sbatch)
echo "Eval job submitted: $EVAL_JOB"
echo ""
echo "Done at: $(date)"
