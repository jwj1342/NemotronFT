#!/bin/bash
#SBATCH --job-name=rl-smoke7
#SBATCH --account=aip-zhouyang
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Smoke v7 — reuse v5 rollout JSONL, retrain only.
# v5 OOMed at max_length=4096 (actual max seq is 2180). Drop to 2200.
# Same 8 retained items, β=0, dynamic filter, 25 steps.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
SRC_ROLLOUT=$PROJECT_DIR/training/rl/runs/stage0_v5/rollouts/iter_0.jsonl
STAGE_DIR=$PROJECT_DIR/training/rl/runs/stage0_v7
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

echo "=== Smoke v7 (retrain only, max_length=2200) $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

module purge
module load python/3.11.5
source venv/bin/activate; unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True
export PYTHONUNBUFFERED=1

accelerate launch \
    --config_file "$PROJECT_DIR/training/fsdp_config.yaml" \
    "$PROJECT_DIR/training/rl/train_grpo.py" \
    --rollout_jsonl "$SRC_ROLLOUT" \
    --adapter_dir   "$BASELINE_ADAPTER" \
    --output_dir    "$STAGE_DIR/policy/iter_1" \
    --lr            5e-6 \
    --beta          0.0 \
    --batch_size    1 \
    --grad_acc      4 \
    --max_steps     25 \
    --max_length    1200 \
    --seed          1234

if [[ -f "$STAGE_DIR/policy/iter_1/adapter_model.safetensors" ]]; then
    touch "$STAGE_DIR/SMOKE_OK"
    echo "✅ SMOKE V7 PASSED"
    ls -la "$STAGE_DIR/policy/iter_1/"
else
    echo "❌ Smoke v7 failed"
    exit 1
fi
echo "=== Done at $(date) ==="
