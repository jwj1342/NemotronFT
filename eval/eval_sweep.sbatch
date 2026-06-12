#!/bin/bash
#SBATCH --job-name=brev-sweep
#SBATCH --account=aip-zhouyang
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/brev-sweep-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/brev-sweep-%j.err

# Screen brevity checkpoints vs the 0.85 warmstart reference on shard-0-of-8 (~1188
# problems, bit ~200 — enough to detect regression). One vLLM session, LoRA swap per
# checkpoint. Gate: find the sweet spot that holds/improves overall WITHOUT regressing
# bit (its 6600-tok search) or other hard categories. Override $ADAPTERS / $SHARD_TOTAL.
set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"; mkdir -p logs
SHARD_TOTAL=${SHARD_TOTAL:-8}
# default: warmstart reference + every brevity checkpoint that exists
ADAPTERS=${ADAPTERS:-"$PROJECT/output/huikang-warmstart $(ls -d $PROJECT/output/brevity-r1/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tr '\n' ' ')"}

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT/venv-vllm/bin/activate"
unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home

echo "=== brevity sweep === job $SLURM_JOB_ID $(hostname) $(date)"
echo "shard_total=$SHARD_TOTAL"
echo "adapters: $ADAPTERS"
python3 "$PROJECT/eval/eval_sweep.py" --adapters $ADAPTERS --shard_index 0 --shard_total $SHARD_TOTAL --tp 4
echo "=== done $(date) ==="
