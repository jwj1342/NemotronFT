#!/bin/bash
#SBATCH --job-name=opd-train
#SBATCH --account=aip-zhouyang
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# OPD unified SFT: from Huikang baseline, 1 epoch on merged trace CSV.
# Reuses the existing training/train.py FSDP+PEFT trainer (low risk).
# Env vars expected:
#   OPD_CSV       (required, path to merged CSV from merge_traces.py)
#   OPD_OUT_DIR   (default: output/opd-final)
#   BASELINE      (default: output/huikang-baseline-converted)
#   LR            (default: 1e-5)

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
OPD_CSV=${OPD_CSV:?OPD_CSV required}
OPD_OUT_DIR=${OPD_OUT_DIR:-$PROJECT_DIR/output/opd-final}
BASELINE=${BASELINE:-$PROJECT_DIR/output/huikang-baseline-converted}
LR=${LR:-1e-5}
MAX_LEN=${MAX_LEN:-8192}
SAVE_STEPS=${SAVE_STEPS:-25}
SEED=${SEED:-4242}

mkdir -p "$OPD_OUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "OPD unified SFT"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "  CSV:  $OPD_CSV"
echo "  out:  $OPD_OUT_DIR"
echo "  init: $BASELINE"
echo "  lr:   $LR    max_len: $MAX_LEN    save_steps: $SAVE_STEPS"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

module purge && module load python/3.11.5
source venv/bin/activate; unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

accelerate launch \
    --config_file training/fsdp_config.yaml \
    training/train.py \
    --output_dir         "$OPD_OUT_DIR" \
    --csv_path           "$OPD_CSV" \
    --resume_adapter_dir "$BASELINE" \
    --lora_rank          32 \
    --lora_alpha         32 \
    --lora_dropout       0.0 \
    --max_length         "$MAX_LEN" \
    --batch_size         1 \
    --grad_acc           16 \
    --num_epochs         1 \
    --lr                 "$LR" \
    --warmup_steps       0 \
    --save_steps         "$SAVE_STEPS" \
    --save_only_model \
    --seed               "$SEED"

if [[ -f "$OPD_OUT_DIR/adapter_model.safetensors" ]]; then
    echo "✅ OPD final adapter: $OPD_OUT_DIR/adapter_model.safetensors"
else
    LATEST=$(ls -d "$OPD_OUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [[ -n "$LATEST" ]]; then
        echo "[opd] copying latest ckpt to final: $LATEST"
        cp "$LATEST/adapter_config.json"     "$OPD_OUT_DIR/adapter_config.json"
        cp "$LATEST/adapter_model.safetensors" "$OPD_OUT_DIR/adapter_model.safetensors"
    fi
fi

echo ""
echo "=== Done at $(date) ==="
