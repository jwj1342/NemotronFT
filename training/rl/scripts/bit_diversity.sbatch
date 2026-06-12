#!/bin/bash
#SBATCH --job-name=bit-div
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/bit-div-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/bit-div-%j.err

# GRPO gatekeeper: does bit have intra-group reward variance on our 0.85 adapter at
# the diversity-max config (G=16, T=1.2, top_p=1.0)? If WRONG-prompt groups have >=1
# correct sample and groups aren't entropy-collapsed -> GRPO has gradient on bit.
set -u
PROJECT=/scratch/jwj/NemotronFT
POOL=$PROJECT/training/rl/runs/hotpool/bit_diversity_pool.jsonl
OUT=$PROJECT/training/rl/runs/bit_diversity/rollout.jsonl
mkdir -p "$(dirname "$OUT")" "$PROJECT/logs"
cd "$PROJECT"

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT/venv-vllm/bin/activate"
unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home

echo "=== bit diversity probe === job $SLURM_JOB_ID $(hostname) $(date)"
python3 "$PROJECT/training/rl/rollout.py" \
    --pool "$POOL" \
    --adapter_dir "$PROJECT/output/huikang-warmstart" \
    --out "$OUT" \
    --group_size 16 \
    --temperature 1.2 \
    --top_p 1.0 \
    --max_tokens 7680 \
    --tensor_parallel_size 4
echo "--- analysis ---"
python3 "$PROJECT/training/rl/diversity_probe.py" --rollout "$OUT" --pool "$POOL"
echo "=== done $(date) ==="
