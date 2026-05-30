#!/bin/bash
#SBATCH --job-name=rl-smoke-4n
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# 4-node smoke: more headroom (32B / 16 ranks = 2 GB per rank) + dual NIC NCCL
# Config under test:
#   - 4 nodes × 4 L40s = 16 ranks FSDP FULL_SHARD
#   - max_length 4096 (vs 2048 in prior smoke)
#   - G=16 (vs 4)
#   - max_steps 5 (vs 10)
#   - NCCL using both bnxt_re NICs (rail-aggregation)

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/stage0_4n
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Stage 0 smoke 4 NODE: $SLURM_NNODES × 4 L40s = $((SLURM_NNODES * 4)) ranks"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "============================================================"

# === 1: pool ===
echo ""
echo "[1/5] Building pool ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/pool.py" \
    --train_csv "$PROJECT_DIR/resource/winning-solution/train.csv" \
    --categories equation_numeric_guess,cryptarithm_deduce \
    --cap_per_cat 5 \
    --out "$STAGE_DIR/pool.jsonl"
deactivate

# === 2: rollout on rank-0 host only ===
echo ""
echo "[2/5] Rollout @ G=16 max_tokens=3072 ..."
module purge && module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
python "$PROJECT_DIR/training/rl/rollout.py" \
    --pool "$STAGE_DIR/pool.jsonl" \
    --adapter_dir "$BASELINE_ADAPTER" \
    --out "$STAGE_DIR/rollouts/iter_0.jsonl" \
    --group_size 16 \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 3072 \
    --max_model_len 4096 \
    --max_num_seqs 32 \
    --gpu_memory_utilization 0.85 \
    --tensor_parallel_size 4 \
    --seed 1234 \
    --compute_ref_logp
RC=$?
deactivate
if [[ $RC -ne 0 ]]; then echo "❌ rollout failed"; exit 1; fi

# === 3: reward sanity ===
echo ""
echo "[3/5] Reward sanity ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/reward.py" --jsonl "$STAGE_DIR/rollouts/iter_0.jsonl" | head -20
RC=$?
if [[ $RC -ne 0 ]]; then echo "❌ reward sanity failed"; exit 1; fi

# === 4: drain VRAM ===
echo ""
echo "[4/5] Waiting for VRAM to drain ..."
sleep 20

# === 5: GRPO training across 4 nodes ===
echo ""
echo "[5/5] GRPO 5 steps, FSDP FULL_SHARD on 4 × 4 = 16 GPUs, max_len=4096 ..."
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
export PYTHONUNBUFFERED=1
# Dual-NIC RoCE: tell NCCL to use both Broadcom HCAs (rail aggregation, ~2x bandwidth)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=bnxt_re0,bnxt_re1
export NCCL_DEBUG=WARN

echo "[grpo-4n] MASTER=$MASTER_ADDR  NNODES=$SLURM_NNODES  procs=$((SLURM_NNODES * 4))"
echo "[nccl] NCCL_IB_HCA=$NCCL_IB_HCA  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    /bin/bash -lc "
        module purge
        module load python/3.11.5
        source $PROJECT_DIR/venv/bin/activate
        unset PIP_PREFIX
        export OMP_NUM_THREADS=8
        export TOKENIZERS_PARALLELISM=false
        export TRANSFORMERS_OFFLINE=1
        export HF_DATASETS_OFFLINE=1
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
        export PYTHONUNBUFFERED=1
        export NCCL_IB_DISABLE=0
        export NCCL_IB_HCA=bnxt_re0,bnxt_re1
        export NCCL_DEBUG=WARN
        accelerate launch \
            --config_file $PROJECT_DIR/training/fsdp_multinode_config.yaml \
            --num_machines $SLURM_NNODES \
            --num_processes \$((SLURM_NNODES * 4)) \
            --machine_rank \$SLURM_NODEID \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MASTER_PORT \
            $PROJECT_DIR/training/rl/train_grpo.py \
                --rollout_jsonl $STAGE_DIR/rollouts/iter_0.jsonl \
                --adapter_dir $BASELINE_ADAPTER \
                --output_dir $STAGE_DIR/policy/iter_1 \
                --lr 5e-6 \
                --beta 0.05 \
                --batch_size 1 \
                --grad_acc 16 \
                --max_steps 5 \
                --max_length 4096 \
                --seed 1234
    "
RC=$?
if [[ $RC -ne 0 ]]; then echo "❌ GRPO train failed"; exit 1; fi

if [[ -f "$STAGE_DIR/policy/iter_1/adapter_model.safetensors" ]]; then
    touch "$STAGE_DIR/SMOKE_4N_OK"
    SIZE_MB=$(stat -c "%s" "$STAGE_DIR/policy/iter_1/adapter_model.safetensors" | awk '{printf "%.1f", $1/1024/1024}')
    BASE_SIZE_MB=$(stat -c "%s" "$BASELINE_ADAPTER/adapter_model.safetensors" | awk '{printf "%.1f", $1/1024/1024}')
    echo ""
    echo "============================================================"
    echo "✅ 4-NODE SMOKE PASSED. Sentinel: $STAGE_DIR/SMOKE_4N_OK"
    echo "  adapter size: ${SIZE_MB} MB (baseline: ${BASE_SIZE_MB} MB)"
    ls -la "$STAGE_DIR/policy/iter_1/"
    echo "============================================================"
else
    echo "❌ Smoke failed: adapter not saved"
    exit 1
fi

echo ""
echo "=== Done at $(date) ==="
