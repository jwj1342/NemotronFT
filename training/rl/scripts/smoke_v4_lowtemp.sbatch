#!/bin/bash
#SBATCH --job-name=rl-smoke4
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:30:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Smoke v4 (T=0.5) — Codex hot-pool prescription.
# - eq_num_guess only, tier-1 hot pool (22 prompts where baseline T=0 was correct)
# - G=8, T=0.95, top_p=0.95
# - β=0, dynamic_sampling drops zero-adv groups at train time
# - 25 optimizer steps
# Success criteria:
#   - dynamic filter retention > 30% (Codex)
#   - reward signal trends up across 25 steps, no NaN

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/stage0_v4
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
POOL_JSONL=$PROJECT_DIR/training/rl/runs/hotpool/tier1_hot_eq_num_guess.jsonl
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Smoke v4 (T=0.5) (hot-pool, 22 prompts)"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "Pool: $POOL_JSONL"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# === 1/4: Rollout ===
echo ""
echo "[1/4] Rollout @ T=0.95 G=8 ..."
module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"
unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

python "$PROJECT_DIR/training/rl/rollout.py" \
    --pool "$POOL_JSONL" \
    --adapter_dir "$BASELINE_ADAPTER" \
    --out "$STAGE_DIR/rollouts/iter_0.jsonl" \
    --group_size 8 \
    --temperature 0.5 \
    --top_p 0.95 \
    --max_tokens 2048 \
    --max_model_len 4096 \
    --max_num_seqs 16 \
    --gpu_memory_utilization 0.85 \
    --tensor_parallel_size 4 \
    --seed 1234
RC=$?
deactivate
[[ $RC -ne 0 ]] && { echo "rollout failed"; exit 1; }

# === 2/4: Reward sanity ===
echo ""
echo "[2/4] Reward sanity ..."
module purge
module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/reward.py" --jsonl "$STAGE_DIR/rollouts/iter_0.jsonl"
[[ $? -ne 0 ]] && { echo "reward sanity failed"; exit 1; }

# === 3/4: Wait for VRAM drain ===
echo ""
echo "[3/4] Waiting for VRAM to drain ..."
for i in $(seq 1 20); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    echo "  attempt $i: total VRAM used = ${USED} MiB"
    [[ ${USED:-99999} -lt 4000 ]] && break
    sleep 6
done

# === 4/4: GRPO 25 steps ===
echo ""
echo "[4/4] GRPO 25 steps β=0 + dynamic_sampling filter ..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6

accelerate launch \
    --config_file "$PROJECT_DIR/training/fsdp_config.yaml" \
    "$PROJECT_DIR/training/rl/train_grpo.py" \
    --rollout_jsonl "$STAGE_DIR/rollouts/iter_0.jsonl" \
    --adapter_dir   "$BASELINE_ADAPTER" \
    --output_dir    "$STAGE_DIR/policy/iter_1" \
    --lr            5e-6 \
    --beta          0.0 \
    --batch_size    1 \
    --grad_acc      8 \
    --max_steps     25 \
    --max_length    4096 \
    --seed          1234
RC=$?

# === Sentinel ===
if [[ -f "$STAGE_DIR/policy/iter_1/adapter_model.safetensors" ]]; then
    touch "$STAGE_DIR/SMOKE_OK"
    echo ""
    echo "============================================================"
    echo "✅ SMOKE V4 PASSED. Sentinel: $STAGE_DIR/SMOKE_OK"
    ls -la "$STAGE_DIR/policy/iter_1/"
    echo "============================================================"
else
    echo "❌ Smoke v4 (T=0.5) failed: adapter not saved"
    exit 1
fi
echo "=== Done at $(date) ==="
