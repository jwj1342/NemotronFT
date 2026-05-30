#!/bin/bash
#SBATCH --job-name=rl-iter
#SBATCH --account=aip-zhouyang
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Outer driver: rollout → train → repeat. Single node, 4 L40s.
# Env vars expected:
#   STAGE_NAME            (e.g. stage1_equation_numeric_guess)
#   STAGE_CATEGORIES      (e.g. equation_numeric_guess)
#   BASELINE_ADAPTER      (default: output/huikang-baseline-converted)
#   N_OUTER               default 12
#   N_INNER               default 50  GRPO steps per outer iter
#   GROUP_SIZE            default 8
#   LR                    default 5e-6
#   BETA                  default 0.05
#   CAP_PER_CAT           default 0 (no cap)
#   ROLLOUT_MAX_TOKENS    default 7680
#   TRAIN_MAX_LEN         default 8192
#   CHAIN_DEPTH           set by chain submission (default 1)
#   MAX_CHAIN_DEPTH       default 10

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
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-7680}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-8192}
CHAIN_DEPTH=${CHAIN_DEPTH:-1}
MAX_CHAIN_DEPTH=${MAX_CHAIN_DEPTH:-10}

cd "$PROJECT_DIR"

echo "============================================================"
echo "RL outer driver: $STAGE_NAME"
echo "Job: $SLURM_JOB_ID   Node: $(hostname)   $(date)"
echo "  cats=$STAGE_CATEGORIES  cap=$CAP_PER_CAT"
echo "  N_OUTER=$N_OUTER  N_INNER=$N_INNER  G=$GROUP_SIZE  LR=$LR  BETA=$BETA"
echo "  rollout: T=$ROLLOUT_TEMP  max_tok=$ROLLOUT_MAX_TOKENS"
echo "  CHAIN_DEPTH=$CHAIN_DEPTH/$MAX_CHAIN_DEPTH"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── chain helper ──
submit_next_chain() {
    if [[ -f "$STAGE_DIR/COMPLETE" ]]; then
        echo "[chain] Stage complete — no chain."
        return 0
    fi
    if [[ "$CHAIN_DEPTH" -ge "$MAX_CHAIN_DEPTH" ]]; then
        echo "[chain] Hit MAX_CHAIN_DEPTH=$MAX_CHAIN_DEPTH"
        return 1
    fi
    LATEST_K=$(ls -d "$STAGE_DIR/policy"/iter_* 2>/dev/null | sort -V | tail -1 | sed 's|.*iter_||')
    if [[ -z "$LATEST_K" ]] || [[ "$LATEST_K" == "0" ]]; then
        echo "[chain] No new iter completed — NOT chaining (crash-loop guard)."
        return 1
    fi
    echo "[chain] Submitting next (depth $((CHAIN_DEPTH+1))/$MAX_CHAIN_DEPTH)."
    sbatch --export=ALL,CHAIN_DEPTH=$((CHAIN_DEPTH+1)) "$0"
}

on_pre_timeout() {
    echo ""
    echo "[signal] SIGUSR1 at $(date) — pre-submitting chain."
    submit_next_chain
}
trap on_pre_timeout USR1

# ── 1. Build pool if not exists ──
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

# ── 2. Determine start iter ──
LAST_K=-1
for d in "$STAGE_DIR/policy"/iter_*; do
    [[ -d "$d" ]] || continue
    if [[ ! -f "$d/adapter_model.safetensors" ]]; then continue; fi
    K=${d##*/iter_}
    if [[ "$K" =~ ^[0-9]+$ ]] && [[ "$K" -gt "$LAST_K" ]]; then LAST_K="$K"; fi
done
START_K=$((LAST_K + 1))
if [[ "$START_K" -eq 0 ]]; then
    # iter_0 = symlinked baseline (no training yet)
    mkdir -p "$STAGE_DIR/policy/iter_0"
    ln -sf "$BASELINE_ADAPTER/adapter_config.json"     "$STAGE_DIR/policy/iter_0/adapter_config.json"
    ln -sf "$BASELINE_ADAPTER/adapter_model.safetensors" "$STAGE_DIR/policy/iter_0/adapter_model.safetensors"
    START_K=0
fi

echo "[loop] resume from iter $START_K / $N_OUTER"

# ── 3. Outer loop ──
for K in $(seq "$START_K" $((N_OUTER - 1))); do
    if [[ -f "$STAGE_DIR/ABORT" ]]; then echo "[loop] ABORT sentinel — stop."; break; fi
    if [[ -f "$STAGE_DIR/policy/iter_$((K+1))/adapter_model.safetensors" ]]; then
        echo "[loop] iter_$((K+1)) already complete — skip"; continue
    fi

    echo ""
    echo "─── outer iter $K → $((K+1))  $(date) ───"
    CUR_POLICY="$STAGE_DIR/policy/iter_$K"
    ROLL_OUT="$STAGE_DIR/rollouts/iter_${K}.jsonl"

    # Rollout phase
    if [[ ! -s "$ROLL_OUT" ]]; then
        echo "[rollout] iter $K  policy=$CUR_POLICY"
        module purge && module load python/3.11.5 gcc/12.3 opencv/4.11.0
        source "$PROJECT_DIR/venv-vllm/bin/activate"; unset PIP_PREFIX
        export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        python "$PROJECT_DIR/training/rl/rollout.py" \
            --pool "$STAGE_DIR/pool.jsonl" \
            --adapter_dir "$CUR_POLICY" \
            --out "$ROLL_OUT" \
            --group_size "$GROUP_SIZE" \
            --temperature "$ROLLOUT_TEMP" \
            --top_p 0.95 \
            --max_tokens "$ROLLOUT_MAX_TOKENS" \
            --max_model_len 8192 \
            --max_num_seqs 16 \
            --gpu_memory_utilization 0.85 \
            --tensor_parallel_size 4 \
            --seed $((1234 + K)) \
            --compute_ref_logp
        RC=$?
        deactivate
        if [[ $RC -ne 0 ]]; then echo "❌ rollout failed at iter $K"; exit 1; fi
        sync; sleep 3
    else
        echo "[rollout] iter $K already done"
    fi

    # Wait for VRAM
    for i in $(seq 1 30); do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
        if [[ ${USED:-99999} -lt 4000 ]]; then break; fi
        sleep 5
    done

    # Training phase
    echo "[train] iter $K → iter $((K+1))"
    module purge && module load python/3.11.5
    source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    accelerate launch \
        --config_file "$PROJECT_DIR/training/fsdp_config.yaml" \
        "$PROJECT_DIR/training/rl/train_grpo.py" \
        --rollout_jsonl "$ROLL_OUT" \
        --adapter_dir   "$CUR_POLICY" \
        --output_dir    "$STAGE_DIR/policy/iter_$((K+1))" \
        --lr            "$LR" \
        --beta          "$BETA" \
        --batch_size    1 \
        --grad_acc      16 \
        --max_steps     "$N_INNER" \
        --max_length    "$TRAIN_MAX_LEN" \
        --seed          $((42 + K)) &
    LAUNCH_PID=$!
    while kill -0 $LAUNCH_PID 2>/dev/null; do
        wait $LAUNCH_PID
    done
    RC=$?
    deactivate
    if [[ $RC -ne 0 ]]; then echo "❌ train failed at iter $K"; exit 1; fi
    sync; sleep 3

    # Progress log
    python3 - <<PYEOF
import csv, json
from pathlib import Path
stage_dir = Path("$STAGE_DIR"); K = $K
total_r, n = 0, 0; by_cat = {}
with open(stage_dir / f"rollouts/iter_{K}.jsonl") as f:
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
done

# ── 4. Stage completion ──
FINAL_ITER="$STAGE_DIR/policy/iter_${N_OUTER}"
if [[ -f "$FINAL_ITER/adapter_model.safetensors" ]]; then
    touch "$STAGE_DIR/COMPLETE"
    cp "$FINAL_ITER/adapter_config.json"     "$STAGE_DIR/adapter_config.json"
    cp "$FINAL_ITER/adapter_model.safetensors" "$STAGE_DIR/adapter_model.safetensors"
    echo ""
    echo "✅ Stage $STAGE_NAME complete: $STAGE_DIR/COMPLETE"
else
    # In-progress: try to chain
    submit_next_chain
fi

echo ""
echo "=== Done at $(date) ==="
