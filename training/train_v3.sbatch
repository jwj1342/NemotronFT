#!/bin/bash
#SBATCH --job-name=sft-v3
#SBATCH --account=aip-zhouyang
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --exclude=rack04-03
#SBATCH --output=/scratch/jwj/NemotronFT/logs/sft-v3-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/sft-v3-%j.err

# SFT v3: cipher-digit CoT augmentation
# Base adapter: Huikang baseline (0.85 LB)
# New data: 319 cipher-digit + 237 numeric + balanced existing = 5253 total
# Goal: push cryptarithm_deduce from 7% to ~30%+

PROJECT_DIR=/scratch/jwj/NemotronFT
OUTPUT_DIR=$PROJECT_DIR/output/sft-v3
BASE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
# 1 epoch over 5253 rows with effective batch 64: ceil(5253/64) = 83 steps
TOTAL_STEPS=83
BASE_LR=1e-4   # lower LR for fine-tuning on top of Huikang baseline
MAX_CHAIN_DEPTH=10
CHAIN_DEPTH=${CHAIN_DEPTH:-1}

if [[ -f "$OUTPUT_DIR/adapter_model.safetensors" ]]; then
    echo "[chain] Final adapter already saved — exiting."
    exit 0
fi

LATEST_CKPT=""
if compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
    LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
fi

RESUME_FLAG="--resume_adapter_dir $BASE_ADAPTER"
EFFECTIVE_LR=$BASE_LR
EFFECTIVE_MAX_STEPS=$TOTAL_STEPS
EFFECTIVE_SEED=$((456 + CHAIN_DEPTH * 1000))

if [[ -n "$LATEST_CKPT" ]]; then
    module load python/3.11.5 >/dev/null 2>&1
    source $PROJECT_DIR/venv/bin/activate >/dev/null 2>&1
    unset PIP_PREFIX
    GS=$(python3 -c "import json; print(json.load(open('$LATEST_CKPT/trainer_state.json'))['global_step'])")
    REMAINING=$((TOTAL_STEPS - GS))
    if [[ "$REMAINING" -le 0 ]]; then
        echo "[chain] Step $GS/$TOTAL_STEPS — copying final adapter."
        cp $LATEST_CKPT/adapter_config.json $OUTPUT_DIR/
        cp $LATEST_CKPT/adapter_model.safetensors $OUTPUT_DIR/
        exit 0
    fi
    EFFECTIVE_LR=$(python3 -c "print(${BASE_LR} * (1 - $GS / $TOTAL_STEPS))")
    EFFECTIVE_MAX_STEPS=$REMAINING
    RESUME_FLAG="--resume_adapter_dir $LATEST_CKPT --max_steps $EFFECTIVE_MAX_STEPS --lr $EFFECTIVE_LR"
    echo "[chain] depth=$CHAIN_DEPTH resume from $LATEST_CKPT (gs=$GS)"
fi

submit_next_chain() {
    if [[ $CHAIN_DEPTH -ge $MAX_CHAIN_DEPTH ]]; then return; fi
    if [[ ! -f "$OUTPUT_DIR/adapter_model.safetensors" ]]; then
        if compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
            NEXT_DEPTH=$((CHAIN_DEPTH + 1))
            sbatch --export=ALL,CHAIN_DEPTH=$NEXT_DEPTH $PROJECT_DIR/training/train_v3.sbatch
        fi
    fi
}
trap submit_next_chain SIGUSR1

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

mkdir -p "$OUTPUT_DIR" /scratch/jwj/NemotronFT/logs

echo "=== SFT v3: cipher-digit CoT augmentation ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

accelerate launch \
    --config_file $PROJECT_DIR/training/fsdp_config.yaml \
    $PROJECT_DIR/training/train.py \
    --output_dir            $OUTPUT_DIR \
    --csv_path              $PROJECT_DIR/training/train_cot_v3.csv \
    --lora_rank             32 \
    --lora_alpha            32 \
    --lora_dropout          0.0 \
    --max_length            8192 \
    --batch_size            1 \
    --grad_acc              16 \
    --num_epochs            1 \
    --warmup_steps          5 \
    --save_steps            5 \
    --save_total_limit      -1 \
    --save_only_model \
    --seed                  $EFFECTIVE_SEED \
    $RESUME_FLAG &
LAUNCH_PID=$!

while kill -0 $LAUNCH_PID 2>/dev/null; do
    wait $LAUNCH_PID
done
EXIT=$?
echo "Train exit: $EXIT  $(date)"
submit_next_chain
echo "Done: $(date)"
