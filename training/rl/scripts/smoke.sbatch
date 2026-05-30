#!/bin/bash
#SBATCH --job-name=rl-smoke
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Stage 0 smoke (revised 2026-05-25 per Codex):
#   - eq_num_guess only (baseline ~16% pass — enough reward density)
#   - 128 prompts × G=8 (low chance of all-fail/all-pass groups)
#   - β=0 (skip KL term, skip ref policy forward)
#   - dynamic sampling filter drops zero-advantage rollouts at train time
#   - 25 optimizer steps
# Success criteria (Codex):
#   - retain rate > 30% after dynamic filter
#   - reward mean monotone or near-monotone up over 25 steps
#   - no NaN, adapter saves
# Writes SMOKE_OK sentinel.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/stage0_v2
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

echo "============================================================"
echo "Stage 0 smoke (single node)"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# === 1/5: Build pool ===
echo ""
echo "[1/5] Building 64-prompt pool (eq_num_guess only, Codex bring-up) ..."
module purge
module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"
unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/pool.py" \
    --train_csv "$PROJECT_DIR/resource/winning-solution/train.csv" \
    --categories equation_numeric_guess \
    --cap_per_cat 64 \
    --out "$STAGE_DIR/pool.jsonl"
RC=$?
deactivate
if [[ $RC -ne 0 ]]; then echo "❌ pool build failed"; exit 1; fi

# === 2/5: Rollout (venv-vllm) ===
echo ""
echo "[2/5] Rollout vs baseline @ T=0.95 G=8 (max_tokens=4096) ..."
module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"
unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

python "$PROJECT_DIR/training/rl/rollout.py" \
    --pool "$STAGE_DIR/pool.jsonl" \
    --adapter_dir "$BASELINE_ADAPTER" \
    --out "$STAGE_DIR/rollouts/iter_0.jsonl" \
    --group_size 8 \
    --temperature 0.95 \
    --top_p 0.95 \
    --max_tokens 2048 \
    --max_model_len 4096 \
    --max_num_seqs 16 \
    --gpu_memory_utilization 0.85 \
    --tensor_parallel_size 4 \
    --seed 1234
RC=$?
deactivate
if [[ $RC -ne 0 ]]; then echo "❌ rollout failed"; exit 1; fi

# === 3/5: Reward sanity ===
echo ""
echo "[3/5] Reward sanity (re-score, should match) ..."
module purge
module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"
unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/reward.py" \
    --jsonl "$STAGE_DIR/rollouts/iter_0.jsonl"
RC=$?
if [[ $RC -ne 0 ]]; then echo "❌ reward sanity failed"; exit 1; fi

# === 4/5: Wait for VRAM to drain ===
echo ""
echo "[4/5] Waiting for VRAM to drain ..."
for i in $(seq 1 20); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    echo "  attempt $i: total VRAM used = ${USED} MiB"
    if [[ ${USED:-99999} -lt 4000 ]]; then break; fi
    sleep 6
done

# === 5/5: GRPO training (25 steps) ===
echo ""
echo "[5/5] GRPO 25 steps β=0 + dynamic_sampling filter ..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
export PYTHONUNBUFFERED=1

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
if [[ $RC -ne 0 ]]; then echo "❌ GRPO train failed"; exit 1; fi

# === Sentinel ===
if [[ -f "$STAGE_DIR/policy/iter_1/adapter_model.safetensors" ]]; then
    touch "$STAGE_DIR/SMOKE_OK"
    echo ""
    echo "============================================================"
    echo "✅ SMOKE PASSED. Sentinel: $STAGE_DIR/SMOKE_OK"
    ls -la "$STAGE_DIR/policy/iter_1/"
    echo "============================================================"
else
    echo "❌ Smoke failed: adapter not saved at $STAGE_DIR/policy/iter_1/"
    exit 1
fi

echo ""
echo "=== Done at $(date) ==="
