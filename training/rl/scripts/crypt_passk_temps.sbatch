#!/bin/bash
#SBATCH --job-name=crypt-temps
#SBATCH --account=aip-zhouyang
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error
set -u
P=/scratch/jwj/NemotronFT; cd "$P"
module purge; module load python/3.11.5 gcc/12.3 opencv/4.11.0
source $P/venv-vllm/bin/activate; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$P/.hf_home CUDA_VISIBLE_DEVICES=0,1,2,3
POOL=$P/training/rl/runs/hotpool/crypt_wellposed_probe.jsonl
# explore at higher temperatures: does pass@k climb? (top_p=1.0 for max diversity)
for T in 1.3 1.6; do
  echo "=== well-posed crypt pass@16 @ T=$T $(date) ==="
  python $P/training/rl/rollout.py \
    --pool $POOL --adapter_dir $P/output/kienngx-086-renamed \
    --out $P/training/rl/runs/hotpool/crypt_probe_T${T}.jsonl \
    --stats_out $P/training/rl/runs/hotpool/crypt_probe_T${T}.stats.json \
    --group_size 16 --temperature $T --top_p 1.0 \
    --max_tokens 7680 --max_model_len 8192 --max_num_seqs 48 \
    --gpu_memory_utilization 0.85 --tensor_parallel_size 4
done
echo "exit=$? $(date)"
