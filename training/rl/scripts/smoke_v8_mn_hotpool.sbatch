#!/bin/bash
#SBATCH --job-name=rl-smoke8
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Smoke v8: multi-node (2 × 4 L40s = 8-way FSDP) on hot-pool eq_num_guess.
# 1-node 4-GPU FSDP OOMs (smoke v5/v6/v7 all hit static 35GB/rank ceiling).
# May 15 multi-node smoke succeeded at max_length=4096; reuse that resource shape.
# Algorithm: β=0, dynamic_sampling filter, seed fix.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/stage0_v8
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
POOL_JSONL=$PROJECT_DIR/training/rl/runs/hotpool/tier1_hot_eq_num_guess.jsonl
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Smoke v8 (2 nodes × 4 L40s, hot-pool, β=0)"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "============================================================"

# === 1/4: Rollout on rank-0 host (vLLM TP=4 local) ===
echo ""
echo "[1/4] Rollout T=0.95 G=8 (seed bug fixed) ..."
module purge && module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
python "$PROJECT_DIR/training/rl/rollout.py" \
    --pool "$POOL_JSONL" \
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
[[ $RC -ne 0 ]] && { echo "❌ rollout failed"; exit 1; }

# === 2/4: Reward sanity ===
echo ""
echo "[2/4] Reward sanity ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/reward.py" --jsonl "$STAGE_DIR/rollouts/iter_0.jsonl"
[[ $? -ne 0 ]] && { echo "❌ reward sanity failed"; exit 1; }
deactivate

# === 3/4: VRAM drain on rank-0 ===
echo ""
echo "[3/4] Waiting for VRAM to drain ..."
sleep 20

# === 4/4: 2-node GRPO ===
echo ""
echo "[4/4] GRPO 25 steps β=0 across 2 × 4 L40s ..."
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
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
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True
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
                --beta 0.0 \
                --batch_size 1 \
                --grad_acc 8 \
                --max_steps 25 \
                --max_length 2200 \
                --seed 1234
    "
RC=$?

if [[ -f "$STAGE_DIR/policy/iter_1/adapter_model.safetensors" ]]; then
    touch "$STAGE_DIR/SMOKE_OK"
    echo "✅ SMOKE V8 PASSED"
    ls -la "$STAGE_DIR/policy/iter_1/"
else
    echo "❌ Smoke v8 failed"
    exit 1
fi
echo "=== Done at $(date) ==="
