#!/bin/bash
#SBATCH --job-name=rl-smoke-mnf
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Multi-node full GRPO smoke: same shape as smoke.sbatch but across 2 nodes
# with FULL_SHARD spreading the 32B base over 8 GPUs instead of 4.
# Expected: no OOM, first step prints loss, sentinel written.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/stage0_mn
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Stage 0 smoke MULTI-NODE: $SLURM_NNODES × 4 L40s"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "============================================================"

# === 1/5: Build pool (rank 0 host only) ===
echo ""
echo "[1/5] Building 10-prompt pool ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/pool.py" \
    --train_csv "$PROJECT_DIR/resource/winning-solution/train.csv" \
    --categories equation_numeric_guess,cryptarithm_deduce \
    --cap_per_cat 5 \
    --out "$STAGE_DIR/pool.jsonl"
deactivate

# === 2/5: Rollout on rank-0 host (local — no srun needed) ===
echo ""
echo "[2/5] Rollout on rank-0 host's 4 L40s (TP=4) ..."
module purge && module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
python "$PROJECT_DIR/training/rl/rollout.py" \
    --pool "$STAGE_DIR/pool.jsonl" \
    --adapter_dir "$BASELINE_ADAPTER" \
    --out "$STAGE_DIR/rollouts/iter_0.jsonl" \
    --group_size 8 \
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

# === 3/5: Reward sanity ===
echo ""
echo "[3/5] Reward sanity ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/reward.py" --jsonl "$STAGE_DIR/rollouts/iter_0.jsonl"
RC=$?
if [[ $RC -ne 0 ]]; then echo "❌ reward sanity failed"; exit 1; fi

# === 4/5: VRAM drain ===
echo ""
echo "[4/5] Waiting for VRAM to drain (rank-0 host) ..."
sleep 15

# === 5/5: GRPO training across 2 nodes ===
echo ""
echo "[5/5] GRPO 10 steps, FSDP FULL_SHARD on 2 × 4 L40s ..."
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
export PYTHONUNBUFFERED=1
# NCCL: dual-NIC RoCE (both Broadcom HCAs, rail aggregation)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=bnxt_re0,bnxt_re1
export NCCL_DEBUG=WARN

echo "[grpo-mn] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$SLURM_NNODES"

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
    touch "$STAGE_DIR/SMOKE_MN_OK"
    echo ""
    echo "============================================================"
    echo "✅ MULTI-NODE SMOKE PASSED. Sentinel: $STAGE_DIR/SMOKE_MN_OK"
    ls -la "$STAGE_DIR/policy/iter_1/"
    echo "============================================================"
else
    echo "❌ Smoke failed: adapter not saved"
    exit 1
fi

echo ""
echo "=== Done at $(date) ==="
