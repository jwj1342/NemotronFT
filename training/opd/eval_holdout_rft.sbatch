#!/bin/bash
#SBATCH --job-name=rft-eval
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/rft-eval-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/rft-eval-%j.err

# T=0 grader-aligned three-metric eval of all RFT pilot checkpoints on the
# holdout + sentinel pool. Picks the checkpoint by net_gain (recover - regress).

set -u
PROJECT=/scratch/jwj/NemotronFT
cd "$PROJECT"
echo "RFT EVAL  Job:$SLURM_JOB_ID  Node:$SLURM_JOB_NODELIST  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT/venv-vllm/bin/activate"
unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT/.hf_home" XDG_CACHE_HOME="$PROJECT/.cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python "$PROJECT/training/opd/eval_holdout_rft.py" \
    --pool "$PROJECT/training/opd/runs/rft/r0/eval_pool.jsonl" \
    --baseline_adapter "$PROJECT/output/huikang-baseline-converted" \
    --scan_dir "$PROJECT/output/rft-r0-pilot" \
    --out "$PROJECT/training/opd/runs/rft/r0/holdout_eval.json" \
    --tensor_parallel_size 4
echo "RFT EVAL end $(date) exit=$?"
