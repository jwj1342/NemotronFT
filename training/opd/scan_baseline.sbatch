#!/bin/bash
#SBATCH --job-name=opd-scan
#SBATCH --account=aip-zhouyang
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Scan all 6682 Flash-COT prompts with the Huikang baseline @ T=0, max_tokens=7680
# (grader-aligned). Output: per-node JSONL with {prompt_id, reward, ...} for each
# prompt. Used to build OPD diff-set (baseline-wrong ∩ Flash-COT-correct) +
# anchors (baseline-correct).

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
SCAN_DIR=$PROJECT_DIR/training/opd/runs/baseline_scan
BASELINE_ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
mkdir -p "$SCAN_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "OPD baseline scan @ T=0 max_tokens=7680  on $SLURM_NNODES nodes"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_JOB_NODELIST  $(date)"
echo "============================================================"

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
            --pool $SCAN_DIR/flashcot_pool.jsonl \
            --adapter_dir $BASELINE_ADAPTER \
            --out $SCAN_DIR/scan_node\${SLURM_NODEID}.jsonl \
            --group_size 1 \
            --temperature 0.0 \
            --top_p 1.0 \
            --max_tokens 7680 \
            --max_model_len 8192 \
            --max_num_seqs 16 \
            --gpu_memory_utilization 0.85 \
            --tensor_parallel_size 4 \
            --shard_index \$SLURM_NODEID \
            --shard_total $SLURM_NNODES \
            --seed 0
    "
RC=$?
if [[ $RC -ne 0 ]]; then echo "❌ scan failed"; exit 1; fi

# Compute per-category pass rate
module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX
python3 - <<'PYEOF'
import json
from pathlib import Path
from collections import defaultdict

sd = Path("/scratch/jwj/NemotronFT/training/opd/runs/baseline_scan")
n_by_cat = defaultdict(int)
ok_by_cat = defaultdict(int)
for p in sorted(sd.glob("scan_node*.jsonl")):
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            cat = rec["category"]
            n_by_cat[cat] += 1
            if float(rec["group"][0]["reward"]) >= 0.5:
                ok_by_cat[cat] += 1

print("[scan] per-category pass@1 (T=0 max_tokens=7680):")
total_n, total_ok = 0, 0
for cat in sorted(n_by_cat):
    n, ok = n_by_cat[cat], ok_by_cat[cat]
    print(f"  {cat:30s}  {ok:5d} / {n:5d}  = {ok/n:.3f}")
    total_n += n; total_ok += ok
print(f"  {'TOTAL':30s}  {total_ok:5d} / {total_n:5d}  = {total_ok/total_n:.3f}")
PYEOF

touch "$SCAN_DIR/COMPLETE"
echo ""
echo "============================================================"
echo "✅ BASELINE SCAN COMPLETE"
ls -la "$SCAN_DIR"
echo "============================================================"
echo "=== Done at $(date) ==="
