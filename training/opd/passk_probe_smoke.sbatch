#!/bin/bash
#SBATCH --job-name=passk-smoke
#SBATCH --account=aip-zhouyang
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# SMOKE TEST of passk_probe.py — tiny params to validate the end-to-end pipeline
# (model+adapter load, sampling, grader verify, solver column, output writing)
# before committing the full 4h run. per_cat=3, n=4, all 4 categories.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
OUT_DIR=$PROJECT_DIR/training/opd/runs/passk
mkdir -p "$OUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "pass@k SMOKE   Job: $SLURM_JOB_ID   Node: $SLURM_JOB_NODELIST"
echo "Start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"
unset PIP_PREFIX

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT_DIR/.hf_home"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python "$PROJECT_DIR/training/opd/passk_probe.py" \
    --adapter_dir "$ADAPTER" \
    --out_dir "$OUT_DIR" \
    --categories "bit_manipulation,equation_numeric_guess,cryptarithm_deduce,cryptarithm_guess" \
    --per_cat 3 \
    --n 4 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k -1 \
    --max_tokens 7680 \
    --max_model_len 8192 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --max_num_seqs 64 \
    --sample_seed 42 \
    --tag smoke
RC=$?

echo ""
echo "============================================================"
echo "SMOKE End: $(date)   exit=$RC"
[ $RC -eq 0 ] && echo "SMOKE_OK" && touch "$OUT_DIR/SMOKE_OK"
echo "============================================================"
exit $RC
