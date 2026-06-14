#!/bin/bash
#SBATCH --job-name=star-sft
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error
set -u
P=/scratch/jwj/NemotronFT; cd "$P"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1); MASTER_PORT=29504
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
            --rollout_jsonl $P/training/rl/runs/star_uc/sft_rollout.jsonl \
            --adapter_dir $P/output/kienngx-086-renamed --output_dir $P/output/star-uc-sft \
            --beta 0.0 --no_drop_zero_std_groups --lr 1e-5 --batch_size 1 --grad_acc 8 \
            --max_steps 15 --max_length 8192 --seed 1234 --tb_dir $HOME/tensorboard_logs/star_uc_sft
"
echo "exit=$? $(date)"; ls -la $P/output/star-uc-sft/adapter_model.safetensors 2>/dev/null
