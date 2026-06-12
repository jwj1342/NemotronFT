#!/bin/bash
#SBATCH --job-name=repro-kienngx
#SBATCH --account=aip-zhouyang
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --exclude=rack04-03
#SBATCH --output=/scratch/jwj/NemotronFT/logs/repro-kienngx-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/repro-kienngx-%j.err

# Reproduce the public 0.86 recipe (kienngx/nvidia-nemotron-training-copy-run-instantly):
# plain-answer SFT (NO CoT) + target_modules=all-linear, on the per-expert base
# model so the LoRA is Kaggle-format directly (no lossy Tinker->Kaggle SVD).
# kienngx hparams: 1200 samples, max_seq 2048, 2 epochs, lr 1e-4, rank 32,
# alpha 32, dropout 0.05, effective batch 4 (1 x grad_acc1 x 4 GPUs). Trains
# from BASE (no --resume_adapter_dir).
#
# Params via env: REPRO_OUT (out dir), REPRO_MAX_STEPS (default -1 = full epochs).

set -u
PROJECT=/scratch/jwj/NemotronFT
OUT=${REPRO_OUT:-$PROJECT/output/repro-kienngx}
MAX_STEPS=${REPRO_MAX_STEPS:--1}
mkdir -p "$OUT" "$PROJECT/logs"
cd "$PROJECT"

module purge
module load python/3.11.5
source $PROJECT/venv/bin/activate
unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

echo "=== REPRO kienngx 0.86 recipe ==="
echo "Job $SLURM_JOB_ID  $(hostname)  $(date)  out=$OUT max_steps=$MAX_STEPS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MS_FLAG=""
[ "$MAX_STEPS" -gt 0 ] 2>/dev/null && MS_FLAG="--max_steps $MAX_STEPS"

accelerate launch --config_file $PROJECT/training/fsdp_config.yaml \
    $PROJECT/training/train.py \
    --output_dir         $OUT \
    --csv_path           $PROJECT/training/repro_1200.csv \
    --plain_answer \
    --all_linear \
    --max_length         2048 \
    --num_epochs         2 \
    --lr                 1e-4 \
    --batch_size         1 \
    --grad_acc           1 \
    --lora_rank          32 \
    --lora_alpha         32 \
    --lora_dropout       0.05 \
    --warmup_steps       10 \
    --save_steps         150 \
    --save_total_limit   -1 \
    --save_only_model \
    --seed               42 \
    $MS_FLAG
RC=$?
echo "repro train exit=$RC  $(date)"
echo "=== adapter 格式检查 (必须 per-expert Kaggle 格式) ==="
python3 -c "
import json,glob
cfg=glob.glob('$OUT/adapter_config.json')+glob.glob('$OUT/checkpoint-*/adapter_config.json')
if cfg:
    c=json.load(open(sorted(cfg)[-1]))
    tm=c.get('target_modules')
    print('target_modules:', tm if isinstance(tm,str) else f'{len(tm)} mods')
" 2>/dev/null
exit $RC
