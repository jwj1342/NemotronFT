#!/bin/bash
#SBATCH --job-name=fingerprint
#SBATCH --account=aip-zhouyang
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/fingerprint-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/fingerprint-%j.err

# Unified operator-fingerprinting SFT (expert's feature-elimination method, executed
# to the end). Warmstart 0.85, teach the model to READ disguised operators from the
# examples instead of assuming glyph conventions. Traces are CONTENTFUL (op-testing
# reasoning) -> lower collapse risk than the brevity boilerplate. Frequent checkpoints
# + lower lr (1e-4) for safety; eval sweep picks the non-regressing sweet spot.
set -u
PROJECT=/scratch/jwj/NemotronFT
OUT=${FP_OUT:-$PROJECT/output/fingerprint-r1}
CSV=${FP_CSV:-$PROJECT/data/fingerprint_sft.csv}
WARM=${FP_WARM:-$PROJECT/output/huikang-warmstart}
MAX_STEPS=${FP_STEPS:-150}
LR=${FP_LR:-1e-4}
MAXLEN=${FP_MAXLEN:-1280}
SAVE_STEPS=${FP_SAVE:-20}
mkdir -p "$OUT" "$PROJECT/logs"; cd "$PROJECT"

module purge; module load python/3.11.5
source $PROJECT/venv/bin/activate; unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

echo "=== fingerprint SFT === job $SLURM_JOB_ID $(hostname) $(date)"
echo "warm=$WARM out=$OUT steps=$MAX_STEPS lr=$LR maxlen=$MAXLEN csv=$CSV"
wc -l "$CSV"
accelerate launch --config_file $PROJECT/training/fsdp_config.yaml \
    $PROJECT/training/train.py \
    --output_dir $OUT --csv_path $CSV --resume_adapter_dir $WARM \
    --max_length $MAXLEN --max_steps $MAX_STEPS --lr $LR \
    --batch_size 2 --grad_acc 2 --warmup_steps 5 \
    --save_steps $SAVE_STEPS --save_total_limit -1 --save_only_model --seed 42
RC=$?
echo "fingerprint train exit=$RC $(date)"; ls -d $OUT/checkpoint-* 2>/dev/null
exit $RC
