#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --account=aip-zhouyang
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --exclude=rack04-03
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# SFT training: 12h per job, manually resumes the LoRA adapter and scales the
# remaining LR schedule. HF Trainer 5.3's built-in resume cannot detect PEFT
# after FSDP wrap so we bypass it entirely. SIGUSR1 fires 10 min before the
# wall-time limit to submit the next chain proactively (SLURM will SIGKILL
# bash directly otherwise).

PROJECT_DIR=/scratch/jwj/NemotronFT
OUTPUT_DIR=$PROJECT_DIR/output/sft
TOTAL_STEPS=100        # ceil(6355 / 64) -- one epoch
BASE_LR=2e-4
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

RESUME_FLAG=""
EFFECTIVE_LR=$BASE_LR
EFFECTIVE_MAX_STEPS=-1
EFFECTIVE_SEED=$((123 + CHAIN_DEPTH * 1000))
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
        bash $PROJECT_DIR/scripts/build_all_submissions.sh \
            $OUTPUT_DIR $PROJECT_DIR/output/sft-submissions
        exit 0
    fi
    EFFECTIVE_LR=$(python3 -c "print(${BASE_LR} * (1 - $GS / $TOTAL_STEPS))")
    EFFECTIVE_MAX_STEPS=$REMAINING
    RESUME_FLAG="--resume_adapter_dir $LATEST_CKPT --max_steps $EFFECTIVE_MAX_STEPS --lr $EFFECTIVE_LR"
    echo "[chain] depth=$CHAIN_DEPTH resume from $LATEST_CKPT (gs=$GS)"
    echo "        remaining=$REMAINING steps, lr=$EFFECTIVE_LR (linear -> 0 over remaining)"
    echo "        seed=$EFFECTIVE_SEED"
else
    echo "[chain] depth=$CHAIN_DEPTH fresh start, lr=$BASE_LR, seed=$EFFECTIVE_SEED"
fi

submit_next_chain() {
    if [[ -f "$OUTPUT_DIR/adapter_model.safetensors" ]]; then
        echo "[chain] Training complete — building zips."
        bash $PROJECT_DIR/scripts/build_all_submissions.sh \
            $OUTPUT_DIR $PROJECT_DIR/output/sft-submissions
        return 0
    fi
    NEW_CKPT=""
    if compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
        NEW_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    fi
    if [[ -z "$NEW_CKPT" || "$NEW_CKPT" == "$LATEST_CKPT" ]]; then
        echo "[chain] No new checkpoint — NOT chaining (crash-loop guard)."
        return 1
    fi
    if [[ "$CHAIN_DEPTH" -lt "$MAX_CHAIN_DEPTH" ]]; then
        NEXT_DEPTH=$((CHAIN_DEPTH + 1))
        echo "[chain] Submitting next job (depth $NEXT_DEPTH/$MAX_CHAIN_DEPTH)."
        cd $PROJECT_DIR
        sbatch --export=ALL,CHAIN_DEPTH=$NEXT_DEPTH $PROJECT_DIR/training/train.sbatch
    else
        echo "[chain] Hit MAX_CHAIN_DEPTH=$MAX_CHAIN_DEPTH."
    fi
}

on_pre_timeout() {
    echo ""
    echo "[signal] SIGUSR1 received at $(date) — 10 min until timeout."
    submit_next_chain
    echo "[signal] Chain submission attempted; allowing remaining time for save."
}
trap on_pre_timeout USR1

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

echo "=== SFT training (Unsloth 0.81 recipe + Mamba LoRA + FSDP) ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

TB_LINK_DIR=/home/jwj/tensorboard_logs
TB_RUN_DIR=$OUTPUT_DIR/runs
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

accelerate launch \
    --config_file $PROJECT_DIR/training/fsdp_config.yaml \
    $PROJECT_DIR/training/train.py \
    --output_dir            $OUTPUT_DIR \
    --csv_path              $PROJECT_DIR/training/train_with_cot.csv \
    --lora_rank             32 \
    --lora_alpha            32 \
    --lora_dropout          0.0 \
    --max_length            8192 \
    --batch_size            1 \
    --grad_acc              16 \
    --num_epochs            1 \
    --warmup_steps          0 \
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
kill $TB_LINK_PID 2>/dev/null
echo ""
echo "Train exit: $EXIT  $(date)"

submit_next_chain

echo ""
echo "Done: $(date)"
