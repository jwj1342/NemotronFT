#!/bin/bash
#SBATCH --job-name=bit-grpo-smoke
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error

# bit GRPO feasibility smoke. Diversity gate PASSED (68% non-degenerate groups,
# 73% of failures recoverable — strong gradient; the "distinct=1.1" verdict was a
# prefix-collision metric bug). The REAL risk is long-sequence training OOM: bit
# responses are ~6600 tok, so max_length must be ~8192 vs the working eq smoke's 2200.
# Reuses the diversity rollout directly (already has response_ids + advantages).
# Tests: does 30B GRPO TRAIN at max_length 8192 on 2 nodes (8-way FSDP + grad ckpt)
# without OOM/NaN? beta=0 (no KL/ref-logp needed for the feasibility smoke).
set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/bit_grpo_smoke
ROLLOUT=$PROJECT_DIR/training/rl/runs/bit_diversity/rollout.jsonl
WARM=$PROJECT_DIR/output/huikang-warmstart
MAXLEN=${BIT_MAXLEN:-8192}
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "bit GRPO smoke (2 nodes x 4 L40s, max_length=$MAXLEN, beta=0)"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "rollout: $ROLLOUT"
echo "============================================================"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29501
echo "[grpo-mn] MASTER_ADDR=$MASTER_ADDR NNODES=$SLURM_NNODES"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    /bin/bash -lc "
        module purge
        module load python/3.11.5
        source $PROJECT_DIR/venv/bin/activate
        unset PIP_PREFIX
        export OMP_NUM_THREADS=8
        export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
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
                --rollout_jsonl $ROLLOUT \
                --adapter_dir $WARM \
                --output_dir $STAGE_DIR/policy_iter1 \
                --lr 5e-6 \
                --beta 0.0 \
                --batch_size 1 \
                --grad_acc 8 \
                --max_steps 25 \
                --max_length $MAXLEN \
                --seed 1234
    "
RC=$?
echo "grpo train exit=$RC $(date)"
if [[ -f "$STAGE_DIR/policy_iter1/adapter_model.safetensors" ]]; then
    echo "BIT GRPO SMOKE: TRAIN COMPLETED (no OOM) — feasibility OK"
    ls -la "$STAGE_DIR/policy_iter1/"
else
    echo "BIT GRPO SMOKE: FAILED (likely OOM at max_length=$MAXLEN — fall back to fsdp_offload_config or 4 nodes)"
fi
echo "=== done $(date) ==="