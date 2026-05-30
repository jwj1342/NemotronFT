#!/bin/bash
#SBATCH --job-name=rl-iter-mn
#SBATCH --account=aip-zhouyang
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Multi-node outer driver: rollout (rank-0 host only via vLLM TP=4) →
# train (FSDP FULL_SHARD across all 8 GPUs via accelerate launch) → repeat.
# Env vars: STAGE_NAME, STAGE_CATEGORIES, BASELINE_ADAPTER, N_OUTER, N_INNER,
# GROUP_SIZE, LR, BETA, ROLLOUT_MAX_TOKENS, TRAIN_MAX_LEN, CAP_PER_CAT,
# CHAIN_DEPTH (auto), MAX_CHAIN_DEPTH (default 10).

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_NAME=${STAGE_NAME:?STAGE_NAME required}
STAGE_CATEGORIES=${STAGE_CATEGORIES:?STAGE_CATEGORIES required}
STAGE_DIR=$PROJECT_DIR/training/rl/runs/$STAGE_NAME
mkdir -p "$STAGE_DIR"/rollouts "$STAGE_DIR"/policy "$STAGE_DIR"/eval "$PROJECT_DIR/logs"

BASELINE_ADAPTER=${BASELINE_ADAPTER:-$PROJECT_DIR/output/huikang-baseline-converted}
N_OUTER=${N_OUTER:-12}
N_INNER=${N_INNER:-50}
GROUP_SIZE=${GROUP_SIZE:-8}
LR=${LR:-5e-6}
BETA=${BETA:-0.05}
CAP_PER_CAT=${CAP_PER_CAT:-0}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-2048}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((ROLLOUT_MAX_TOKENS + 1536))}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-$ROLLOUT_MAX_MODEL_LEN}
CHAIN_DEPTH=${CHAIN_DEPTH:-1}
MAX_CHAIN_DEPTH=${MAX_CHAIN_DEPTH:-10}

cd "$PROJECT_DIR"

echo "============================================================"
echo "RL multi-node iter driver: $STAGE_NAME"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "  cats=$STAGE_CATEGORIES  cap=$CAP_PER_CAT"
echo "  N_OUTER=$N_OUTER  N_INNER=$N_INNER  G=$GROUP_SIZE  LR=$LR  BETA=$BETA"
echo "  rollout max_tok=$ROLLOUT_MAX_TOKENS  rollout model_len=$ROLLOUT_MAX_MODEL_LEN  train max_len=$TRAIN_MAX_LEN"
echo "  CHAIN_DEPTH=$CHAIN_DEPTH/$MAX_CHAIN_DEPTH"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

submit_next_chain() {
    if [[ -f "$STAGE_DIR/COMPLETE" ]]; then echo "[chain] complete — no chain."; return 0; fi
    if [[ "$CHAIN_DEPTH" -ge "$MAX_CHAIN_DEPTH" ]]; then echo "[chain] MAX_CHAIN_DEPTH hit"; return 1; fi
    LATEST_K=$(ls -d "$STAGE_DIR/policy"/iter_* 2>/dev/null | sort -V | tail -1 | sed 's|.*iter_||')
    if [[ -z "$LATEST_K" ]] || [[ "$LATEST_K" == "0" ]]; then echo "[chain] no progress — guarding"; return 1; fi
    echo "[chain] submitting next (depth $((CHAIN_DEPTH+1))/$MAX_CHAIN_DEPTH)"
    sbatch --export=ALL,CHAIN_DEPTH=$((CHAIN_DEPTH+1)) "$0"
}
trap 'echo "[signal] SIGUSR1 at $(date) — pre-submitting chain."; submit_next_chain' USR1

# Build pool (on rank-0 host, once)
if [[ ! -f "$STAGE_DIR/pool.jsonl" ]]; then
    echo ""
    echo "[setup] Building stage pool ..."
    module purge && module load python/3.11.5
    source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
    CAP_FLAG=""
    if [[ "$CAP_PER_CAT" -gt 0 ]]; then CAP_FLAG="--cap_per_cat $CAP_PER_CAT"; fi
    python "$PROJECT_DIR/training/rl/pool.py" \
        --categories "$STAGE_CATEGORIES" \
        $CAP_FLAG \
        --out "$STAGE_DIR/pool.jsonl"
    deactivate
fi
POOL_N=$(wc -l < "$STAGE_DIR/pool.jsonl")
echo "[setup] pool: $POOL_N prompts"

# Determine start iter.
# iter_0 is the BASELINE seed (a symlink), NOT a training output.
# Outer iter K reads policy/iter_K and writes policy/iter_{K+1}.
# So if iter_{LAST_TRAINED} exists as a real training output, resume at K=LAST_TRAINED.
LAST_TRAINED=-1
for d in "$STAGE_DIR/policy"/iter_*; do
    [[ -d "$d" ]] || continue
    K=${d##*/iter_}
    [[ "$K" =~ ^[0-9]+$ ]] || continue
    [[ "$K" == "0" ]] && continue                                    # seed, skip
    [[ -f "$d/adapter_model.safetensors" ]] || continue
    if [[ "$K" -gt "$LAST_TRAINED" ]]; then LAST_TRAINED="$K"; fi
done
if [[ "$LAST_TRAINED" -lt 0 ]]; then
    START_K=0
    if [[ ! -e "$STAGE_DIR/policy/iter_0/adapter_model.safetensors" ]]; then
        mkdir -p "$STAGE_DIR/policy/iter_0"
        ln -sf "$BASELINE_ADAPTER/adapter_config.json"     "$STAGE_DIR/policy/iter_0/adapter_config.json"
        ln -sf "$BASELINE_ADAPTER/adapter_model.safetensors" "$STAGE_DIR/policy/iter_0/adapter_model.safetensors"
    fi
else
    START_K=$LAST_TRAINED
fi
echo "[loop] resume from iter $START_K / $N_OUTER  (LAST_TRAINED=$LAST_TRAINED)"

# Multi-node training setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
echo "[nccl] MASTER=$MASTER_ADDR:$MASTER_PORT  NNODES=$SLURM_NNODES  procs_total=$((SLURM_NNODES * 4))"

for K in $(seq "$START_K" $((N_OUTER - 1))); do
    if [[ -f "$STAGE_DIR/ABORT" ]]; then echo "[loop] ABORT — stop."; break; fi
    if [[ -f "$STAGE_DIR/policy/iter_$((K+1))/adapter_model.safetensors" ]]; then
        echo "[loop] iter_$((K+1)) already complete — skip"; continue
    fi

    echo ""
    echo "─── outer iter $K → $((K+1))  $(date) ───"
    CUR_POLICY="$STAGE_DIR/policy/iter_$K"
    ROLL_OUT="$STAGE_DIR/rollouts/iter_${K}.jsonl"

    # Rollout phase: each node runs its own vLLM on its shard of the pool.
    # `rollout.py --shard_index/--shard_total` already splits the pool.
    # Each node writes rollouts/iter_K_nodeN.jsonl; training reads all.
    SHARDS_OK=1
    for i in $(seq 0 $((SLURM_NNODES - 1))); do
        if [[ ! -s "$STAGE_DIR/rollouts/iter_${K}_node${i}.jsonl" ]]; then
            SHARDS_OK=0; break
        fi
    done
    if [[ $SHARDS_OK -eq 0 ]]; then
        echo "[rollout] iter $K  parallel across $SLURM_NNODES nodes  policy=$CUR_POLICY"
        srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
            /bin/bash -lc "
                module purge
                module load python/3.11.5 gcc/12.3 opencv/4.11.0
                source $PROJECT_DIR/venv-vllm/bin/activate
                unset PIP_PREFIX
                export TOKENIZERS_PARALLELISM=false
                export TRANSFORMERS_OFFLINE=1
                export CUDA_VISIBLE_DEVICES=0,1,2,3
                python $PROJECT_DIR/training/rl/rollout.py \
                    --pool $STAGE_DIR/pool.jsonl \
                    --adapter_dir $CUR_POLICY \
                    --out $STAGE_DIR/rollouts/iter_${K}_node\${SLURM_NODEID}.jsonl \
                    --group_size $GROUP_SIZE \
                    --temperature $ROLLOUT_TEMP \
                    --top_p 0.95 \
                    --max_tokens $ROLLOUT_MAX_TOKENS \
                    --max_model_len $ROLLOUT_MAX_MODEL_LEN \
                    --max_num_seqs 32 \
                    --gpu_memory_utilization 0.85 \
                    --tensor_parallel_size 4 \
                    --shard_index \$SLURM_NODEID \
                    --shard_total $SLURM_NNODES \
                    --seed $((1234 + K * SLURM_NNODES)) \
                    --compute_ref_logp
            "
        RC=$?
        if [[ $RC -ne 0 ]]; then echo "❌ rollout failed at iter $K"; exit 1; fi
        sync; sleep 5
    fi

    # Wait for VRAM drain on rank-0 host
    for i in $(seq 1 30); do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
        if [[ ${USED:-99999} -lt 4000 ]]; then break; fi
        sleep 5
    done

    # Training phase (multi-node FSDP FULL_SHARD)
    echo "[train] iter $K → iter $((K+1))  (FSDP across $((SLURM_NNODES * 4)) GPUs)"
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
            export NCCL_IB_DISABLE=$NCCL_IB_DISABLE
            export NCCL_DEBUG=$NCCL_DEBUG
            accelerate launch \
                --config_file $PROJECT_DIR/training/fsdp_multinode_config.yaml \
                --num_machines $SLURM_NNODES \
                --num_processes \$((SLURM_NNODES * 4)) \
                --machine_rank \$SLURM_NODEID \
                --main_process_ip $MASTER_ADDR \
                --main_process_port $MASTER_PORT \
                $PROJECT_DIR/training/rl/train_grpo.py \
                    --rollout_jsonl $STAGE_DIR/rollouts/iter_${K}_node*.jsonl \
                    --adapter_dir $CUR_POLICY \
                    --output_dir $STAGE_DIR/policy/iter_$((K+1)) \
                    --lr $LR \
                    --beta $BETA \
                    --batch_size 1 \
                    --grad_acc 16 \
                    --max_steps $N_INNER \
                    --max_length $TRAIN_MAX_LEN \
                    --seed $((42 + K))
        "
    RC=$?
    if [[ $RC -ne 0 ]]; then echo "❌ train failed at iter $K"; exit 1; fi
    sync; sleep 5

    # Progress log
    module purge && module load python/3.11.5
    source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
    python3 - <<PYEOF
import csv, json
from pathlib import Path
stage_dir = Path("$STAGE_DIR"); K = $K
total_r, n = 0, 0; by_cat = {}
for p in sorted(stage_dir.glob(f"rollouts/iter_{K}_node*.jsonl")):
  with open(p) as f:
    for line in f:
        rec = json.loads(line)
        cat = rec["category"]
        for g in rec["group"]:
            r = float(g["reward"])
            total_r += r; n += 1
            by_cat.setdefault(cat, []).append(r)
mean_r = total_r / n if n else 0
prog = stage_dir / "progress.csv"
first = not prog.exists()
with open(prog, "a") as f:
    w = csv.writer(f)
    cats_sorted = sorted(by_cat)
    if first:
        w.writerow(["iter","n","reward_mean"] + cats_sorted)
    means = [sum(by_cat[c])/len(by_cat[c]) for c in cats_sorted]
    w.writerow([K, n, f"{mean_r:.4f}"] + [f"{m:.4f}" for m in means])
print(f"[progress] iter {K}: n={n} reward_mean={mean_r:.4f}")
for c in cats_sorted:
    rs = by_cat[c]
    print(f"  {c:30s} pass@1={sum(rs)/len(rs):.3f}")
PYEOF
    deactivate
done

FINAL_ITER="$STAGE_DIR/policy/iter_${N_OUTER}"
if [[ -f "$FINAL_ITER/adapter_model.safetensors" ]]; then
    touch "$STAGE_DIR/COMPLETE"
    cp "$FINAL_ITER/adapter_config.json"     "$STAGE_DIR/adapter_config.json"
    cp "$FINAL_ITER/adapter_model.safetensors" "$STAGE_DIR/adapter_model.safetensors"
    echo ""
    echo "✅ Stage $STAGE_NAME complete: $STAGE_DIR/COMPLETE"
else
    submit_next_chain
fi

echo ""
echo "=== Done at $(date) ==="
