#!/bin/bash
#SBATCH --job-name=rl-hardpool
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Scan all bit_manipulation prompts at T=0 with the baseline adapter,
# then keep only the ones the baseline gets wrong (hard prompts).
# These become the focused pool for Stage 1 v3 RL.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
HARD_DIR=$PROJECT_DIR/training/rl/runs/hardpool_bitmanip
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
mkdir -p "$HARD_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Hard-pool scan: bit_manipulation @ T=0 on $SLURM_NNODES nodes"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "============================================================"

# === 1: build full pool ===
echo ""
echo "[1/3] Building full bit_manipulation pool ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python "$PROJECT_DIR/training/rl/pool.py" \
    --categories bit_manipulation \
    --cap_per_cat 0 \
    --out "$HARD_DIR/full_pool.jsonl"
deactivate

# === 2: parallel T=0 G=1 scan ===
echo ""
echo "[2/3] T=0 G=1 scan across $SLURM_NNODES nodes ..."
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
            --pool $HARD_DIR/full_pool.jsonl \
            --adapter_dir $BASELINE_ADAPTER \
            --out $HARD_DIR/scan_T0_node\${SLURM_NODEID}.jsonl \
            --group_size 1 \
            --temperature 0.0 \
            --top_p 1.0 \
            --max_tokens 3072 \
            --max_model_len 4096 \
            --max_num_seqs 32 \
            --gpu_memory_utilization 0.85 \
            --tensor_parallel_size 4 \
            --shard_index \$SLURM_NODEID \
            --shard_total $SLURM_NNODES \
            --seed 0
    "
RC=$?
if [[ $RC -ne 0 ]]; then echo "❌ scan failed"; exit 1; fi

# === 3: filter to hard_pool.jsonl ===
echo ""
echo "[3/3] Filtering hard prompts (reward < 0.5) ..."
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python3 - <<'PYEOF'
import json
from pathlib import Path
hd = Path("/scratch/jwj/NemotronFT/training/rl/runs/hardpool_bitmanip")

correct_ids = set()
n_scanned = 0
for p in sorted(hd.glob("scan_T0_node*.jsonl")):
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            n_scanned += 1
            if float(rec["group"][0]["reward"]) >= 0.5:
                correct_ids.add(str(rec["prompt_id"]))

hard = []
with open(hd / "full_pool.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if str(rec["id"]) not in correct_ids:
            hard.append(rec)

with open(hd / "hard_pool.jsonl", "w") as f:
    for h in hard:
        f.write(json.dumps(h) + "\n")

print(f"[hardpool] scanned={n_scanned} correct={len(correct_ids)} wrong={len(hard)} pass@1={len(correct_ids)/max(1,n_scanned):.3f}")
print(f"[hardpool] wrote {len(hard)} hard prompts to {hd}/hard_pool.jsonl")
PYEOF
RC=$?
if [[ $RC -ne 0 ]]; then echo "❌ filter failed"; exit 1; fi

touch "$HARD_DIR/COMPLETE"
echo ""
echo "============================================================"
echo "✅ HARD POOL COMPLETE: $HARD_DIR/hard_pool.jsonl"
ls -la "$HARD_DIR"
echo "============================================================"
echo "=== Done at $(date) ==="
