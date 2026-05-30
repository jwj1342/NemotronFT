#!/bin/bash
#SBATCH --job-name=eval-shard
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Sharded full train.csv eval. Submit N copies with SHARD_INDEX = 0..N-1.
# Each shard handles df[SHARD_INDEX::SHARD_TOTAL] independently.
# Env vars: ADAPTER_DIR, SHARD_INDEX, SHARD_TOTAL

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
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_TOTAL="${SHARD_TOTAL:-4}"

echo "=== Shard $SHARD_INDEX/$SHARD_TOTAL on adapter $(basename $ADAPTER_DIR) ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python $PROJECT_DIR/eval/eval_full_train.py \
    --adapter_dir            "$ADAPTER_DIR" \
    --tensor_parallel_size   4 \
    --max_num_seqs           32 \
    --shard_index            "$SHARD_INDEX" \
    --shard_total            "$SHARD_TOTAL"

echo "=== Done at $(date) ==="
