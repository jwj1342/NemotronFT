#!/bin/bash
#SBATCH --job-name=gen-traces
#SBATCH --account=aip-zhouyang
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/gen-traces-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/gen-traces-%j.err

# Harvest the 0.85 model's OWN correct reasoning traces (grader-exact T=0) on all
# 9500 train problems, WITH full text. These are the clean reasoning-preserving
# ANCHORS for the recipe-E mix (hard categories keep full traces -> no collapse) and
# are reusable for GRPO. Also yields a fresh baseline per-category number.
set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"; mkdir -p logs

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT/venv-vllm/bin/activate"
unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home

echo "=== gen traces === job $SLURM_JOB_ID $(hostname) $(date)"
python3 "$PROJECT/eval/eval_full_train.py" \
    --adapter_dir "$PROJECT/output/huikang-warmstart" \
    --save_text \
    --tensor_parallel_size 4 \
    --output_jsonl "$PROJECT/output/traces_huikang-warmstart.jsonl" \
    --output_summary "$PROJECT/output/traces_huikang-warmstart_summary.json"
echo "=== done $(date) ==="
wc -l "$PROJECT/output/traces_huikang-warmstart.jsonl" 2>/dev/null