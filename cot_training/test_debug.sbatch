#!/bin/bash
#SBATCH --job-name=cot-debug
#SBATCH --account=def-zhouyang
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
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
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Debug run: testing training loop, eval, TensorBoard ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo ""

# ── Small-scale test ─────────────────────────────────────────────────────────
# Only 10 samples/class (regular) + full CoT, 5 training steps, eval at step 3.
# This tests: dataset loading, weighted loss, eval callback, TensorBoard write.
# Should complete in ~15 minutes.
python $PROJECT_DIR/cot_training/train.py \
    --output_dir                $PROJECT_DIR/output/cot-debug \
    --checkpoint_dir            $PROJECT_DIR/output/baseline-lora/checkpoint-100 \
    --regular_samples_per_class 10 \
    --eval_samples_per_class    2 \
    --max_length                1024 \
    --boxed_loss_weight         5.0 \
    --reasoning_loss_weight     1.0 \
    --max_steps                 5 \
    --batch_size                1 \
    --grad_acc                  2 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --eval_steps                3 \
    --eval_max_new_tokens       256 \
    --eval_batch_size           1 \
    --save_steps                5 \
    --tensorboard_dir           /home/jwj/tensorboard_logs

echo ""
echo "=== Debug run finished at $(date) ==="
echo "Exit code: $?"
