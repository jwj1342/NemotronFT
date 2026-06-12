#!/bin/bash
#SBATCH --job-name=rft-probe
#SBATCH --account=aip-zhouyang
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# RFT PROBE phase — sample G CoT per bit train-split FAILED puzzle at T>0 via the
# validated rollout.py. Output (rollout JSONL: {prompt_id,category,prompt,gt,
# group:[{text}]}) feeds rft_pipeline.py FILTER+BUILD. Read-only on adapter.
#
# Parameterized via env: PROBE_POOL (required), PROBE_GROUP (default 16),
# PROBE_OUT (required). Pass with: sbatch --export=ALL,PROBE_POOL=...,...

set -u
PROJECT=/scratch/jwj/NemotronFT
POOL=${PROBE_POOL:?set PROBE_POOL}
GROUP=${PROBE_GROUP:-16}
OUT=${PROBE_OUT:?set PROBE_OUT}
ADAPTER=$PROJECT/output/huikang-baseline-converted
mkdir -p "$(dirname "$OUT")" "$PROJECT/logs"
cd "$PROJECT"

echo "============================================================"
echo "RFT PROBE  Job:$SLURM_JOB_ID  Node:$SLURM_JOB_NODELIST  $(date)"
echo "pool=$POOL  group=$GROUP  out=$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT/venv-vllm/bin/activate"
unset PIP_PREFIX

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT/.hf_home"
export XDG_CACHE_HOME="$PROJECT/.cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python "$PROJECT/training/rl/rollout.py" \
    --pool "$POOL" \
    --adapter_dir "$ADAPTER" \
    --out "$OUT" \
    --group_size "$GROUP" \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 7680 \
    --max_model_len 8192 \
    --tensor_parallel_size 4 \
    --max_num_seqs 64 \
    --gpu_memory_utilization 0.85
RC=$?
echo ""
echo "PROBE end $(date)  exit=$RC  -> $OUT"
exit $RC
