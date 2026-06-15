#!/bin/bash
#SBATCH --job-name=star-mc-kl
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error
# NEW config: multi-category STaR + KL-to-0.86 anchor. The plain STaR (beta=0) regressed
# (0.84-0.85) because nothing pinned the drift; the KL anchor is what held 0.86 in GRPO.
# This combines self-correct-CoT SFT (advantage=1) WITH a beta=0.02 KL-to-warmstart anchor
# (ref_logp pre-scored under the 0.86 adapter) — the one config we never tried. Gamble for 0.87.
set -u
P=/scratch/jwj/NemotronFT; cd "$P"
SRC=$P/training/rl/runs/star_multicat/sft_rollout.jsonl
ROLL=$P/training/rl/runs/star_multicat/sft_rollout_kl.jsonl
cp "$SRC" "$ROLL"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1); MASTER_PORT=29506

# ---- phase 1: score ref_logp under the 0.86 adapter (node-0 only, vllm env) ----
echo "=== [1/2] scoring ref_logp under kienngx-086-renamed $(date) ==="
module purge; module load python/3.11.5 gcc/12.3 opencv/4.11.0
source $P/venv-vllm/bin/activate; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$P/.hf_home VLLM_CACHE_ROOT=$P/.vllm_cache CUDA_VISIBLE_DEVICES=0,1,2,3
python $P/training/rl/score_ref_logp.py --rollout "$ROLL" --ref_adapter_dir $P/output/kienngx-086-renamed \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.55 --max_num_seqs 16 || { echo "ref scoring failed"; exit 1; }
echo "=== ref_logp scored $(date) ==="

# ---- phase 2: 2-node STaR-SFT with KL anchor (training env) ----
echo "=== [2/2] STaR+KL train (beta=0.02) across 2x4 L40s $(date) ==="
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 /bin/bash -lc "
    module purge; module load python/3.11.5
    source $P/venv/bin/activate; unset PIP_PREFIX
    export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True
    export NCCL_IB_DISABLE=0 NCCL_IB_HCA=bnxt_re0,bnxt_re1 NCCL_DEBUG=WARN
    accelerate launch --config_file $P/training/fsdp_multinode_config.yaml \
        --num_machines $SLURM_NNODES --num_processes \$((SLURM_NNODES*4)) --machine_rank \$SLURM_NODEID \
        --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
        $P/training/rl/train_grpo.py \
            --rollout_jsonl $ROLL \
            --adapter_dir $P/output/kienngx-086-renamed --output_dir $P/output/star-multicat-kl \
            --beta 0.02 --no_drop_zero_std_groups --lr 1e-5 --batch_size 1 --grad_acc 8 \
            --max_steps 16 --max_length 8192 --seed 1234 --tb_dir \$HOME/tensorboard_logs/star_multicat_kl
"
echo "exit=$? $(date)"; ls -la $P/output/star-multicat-kl/adapter_model.safetensors 2>/dev/null
