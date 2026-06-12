#!/bin/bash
#SBATCH --job-name=bit-grpo-val
#SBATCH --account=aip-zhouyang
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error

# bit GRPO small-scale VALIDATION (1 iteration). Feasibility already proven (smoke
# trained at maxlen 8192 on 2 nodes, no OOM). This tests LEARNING: does GRPO recover
# bit failures without regressing the correct set? beta=0 (GRPO group-relative +/-
# advantage; differs from RFT which lacks the negative gradient). Net gain judged on
# a DISJOINT holdout by a separate T=0 eval. If net+ -> add KL-to-warmstart for full
# run; if net~0 -> prior-coupling wall holds.
set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/bit_grpo_val
POOL=$PROJECT_DIR/training/rl/runs/hotpool/bit_grpo_train.jsonl
WARM=$PROJECT_DIR/output/huikang-warmstart
mkdir -p "$STAGE_DIR/rollouts" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "bit GRPO validation (2 nodes, rollout G8 + train 60 steps beta=0)"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "============================================================"

# === 1/3: Rollout (rank-0 host, vLLM TP=4 local) ===
echo "[1/3] Rollout G=8 T=1.0 on bit train pool ($(wc -l < "$POOL") prompts) ..."
module purge && module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT_DIR/.hf_home
export CUDA_VISIBLE_DEVICES=0,1,2,3
python "$PROJECT_DIR/training/rl/rollout.py" \
    --pool "$POOL" --adapter_dir "$WARM" --out "$STAGE_DIR/rollouts/iter0.jsonl" \
    --group_size 8 --temperature 1.0 --top_p 1.0 \
    --max_tokens 7680 --max_model_len 8192 --max_num_seqs 16 \
    --gpu_memory_utilization 0.85 --tensor_parallel_size 4
RC=$?; deactivate
[[ $RC -ne 0 ]] && { echo "rollout failed rc=$RC"; exit 1; }

# === 2/3: reward sanity + VRAM drain ===
echo "[2/3] reward sanity ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/reward.py" --jsonl "$STAGE_DIR/rollouts/iter0.jsonl" || true
deactivate
sleep 20

# === 3/3: 2-node GRPO train (beta=0) ===
echo "[3/3] GRPO 60 steps beta=0 across 2x4 L40s ..."
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29502
echo "[grpo-mn] MASTER_ADDR=$MASTER_ADDR NNODES=$SLURM_NNODES"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    /bin/bash -lc "
        module purge; module load python/3.11.5
        source $PROJECT_DIR/venv/bin/activate; unset PIP_PREFIX
        export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True
        export PYTHONUNBUFFERED=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=bnxt_re0,bnxt_re1 NCCL_DEBUG=WARN
        accelerate launch \
            --config_file $PROJECT_DIR/training/fsdp_multinode_config.yaml \
            --num_machines $SLURM_NNODES --num_processes \$((SLURM_NNODES * 4)) \
            --machine_rank \$SLURM_NODEID \
            --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
            $PROJECT_DIR/training/rl/train_grpo.py \
                --rollout_jsonl $STAGE_DIR/rollouts/iter0.jsonl \
                --adapter_dir $WARM \
                --output_dir $STAGE_DIR/policy_iter1 \
                --lr 5e-6 --beta 0.0 --batch_size 1 --grad_acc 8 \
                --max_steps 60 --max_length 8192 --seed 1234
    "
RC=$?
echo "grpo train exit=$RC $(date)"
if [[ -f "$STAGE_DIR/policy_iter1/adapter_model.safetensors" ]]; then
    echo "BIT GRPO VAL: TRAIN OK -> $STAGE_DIR/policy_iter1"; ls -la "$STAGE_DIR/policy_iter1/"
else
    echo "BIT GRPO VAL: train produced no adapter (check log)"
fi
echo "=== done $(date) ==="