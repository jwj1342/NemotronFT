#!/bin/bash
#SBATCH --job-name=eval-full
#SBATCH --account=aip-zhouyang
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Per-category evaluation on the full 9500-problem train.csv.
# Default: evaluate the 0.85 strong baseline (huikang-baseline-converted).
# Override via:  ADAPTER_DIR=output/<dir> sbatch eval/eval_full_train.sbatch

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0

PROJECT_DIR=/scratch/jwj/NemotronFT
source $PROJECT_DIR/venv-vllm/bin/activate
unset PIP_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

ADAPTER_DIR="${ADAPTER_DIR:-$PROJECT_DIR/output/huikang-baseline-converted}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "=== Per-category eval on full train.csv ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "Adapter: $ADAPTER_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

EXTRA_ARGS=""
if [[ -n "$MAX_SAMPLES" ]]; then
    EXTRA_ARGS="--max_samples $MAX_SAMPLES"
fi

python $PROJECT_DIR/eval/eval_full_train.py \
    --adapter_dir            "$ADAPTER_DIR" \
    --tensor_parallel_size   4 \
    --max_num_seqs           32 \
    $EXTRA_ARGS

echo "=== Done at $(date) ==="
