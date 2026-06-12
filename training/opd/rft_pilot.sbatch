#!/bin/bash
#SBATCH --job-name=rft-pilot
#SBATCH --account=aip-zhouyang
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --exclude=rack04-03
#SBATCH --output=/scratch/jwj/NemotronFT/logs/rft-pilot-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/rft-pilot-%j.err

# RFT PILOT (bit round-0): low-dose, controlled SFT to test if answer-filtered
# self-training recovers bit failures WITHOUT regressing the 1402 already-correct.
# Continues from Huikang baseline (output unchanged -> rollback = discard rft-r0-pilot).
# 825 rows (262 bit-recover + 263 bit-anchor + 300 non-bit-anchor), 3 epochs,
# LOW lr 5e-5 (Codex: regression is the main risk), save every 5 steps so we can
# pick the checkpoint by T=0 holdout net-gain, NOT by train loss.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
OUTPUT_DIR=$PROJECT_DIR/output/rft-r0-pilot
BASE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
RFT_CSV=$PROJECT_DIR/training/opd/runs/rft/r0/train_cot_rft.csv
LR=5e-5
EPOCHS=3
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

module purge
module load python/3.11.5
source $PROJECT_DIR/venv/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== RFT PILOT (bit round-0) ==="
echo "Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "base=$BASE_ADAPTER  csv=$RFT_CSV  out=$OUTPUT_DIR  lr=$LR  epochs=$EPOCHS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

accelerate launch \
    --config_file $PROJECT_DIR/training/fsdp_config.yaml \
    $PROJECT_DIR/training/train.py \
    --output_dir            $OUTPUT_DIR \
    --csv_path              $RFT_CSV \
    --resume_adapter_dir    $BASE_ADAPTER \
    --lora_rank             32 \
    --lora_alpha            32 \
    --lora_dropout          0.0 \
    --max_length            8192 \
    --batch_size            1 \
    --grad_acc              16 \
    --num_epochs            $EPOCHS \
    --lr                    $LR \
    --warmup_steps          3 \
    --save_steps            5 \
    --save_total_limit      -1 \
    --save_only_model \
    --seed                  456
RC=$?
echo "Pilot train exit=$RC  $(date)"
echo "Checkpoints in $OUTPUT_DIR (pick by T=0 holdout net-gain, not loss)"
exit $RC
