#!/bin/bash
#SBATCH --job-name=brevity
#SBATCH --account=aip-zhouyang
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/brevity-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/brevity-%j.err

# 0.87-recipe reproduction (secure-0.87 path). Warmstart from the corrected lossy-0.85
# adapter (huikang-warmstart) and do a SHORT-trace SFT on easy-category correct
# examples. Faithful to the public cluster (lossy base + short-trace recipe -> 0.87).
# Active ingredient = the short-trace/decisiveness format; data = real answers (no
# poison). Frequent checkpoints so eval can find the sweet spot before brevity bleeds
# into (and regresses) the hard categories. Decision/submit gate run separately.

set -u
PROJECT=/scratch/jwj/NemotronFT
OUT=${BREV_OUT:-$PROJECT/output/brevity-r1}
TRAIN_CSV=${BREV_CSV:-$PROJECT/data/crd_brevity_sft.csv}
WARM=${BREV_WARM:-$PROJECT/output/huikang-warmstart}
MAX_STEPS=${BREV_STEPS:-240}
LR=${BREV_LR:-2e-4}
MAXLEN=${BREV_MAXLEN:-1536}
SAVE_STEPS=${BREV_SAVE:-20}
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

echo "=== brevity recipe train === job $SLURM_JOB_ID $(hostname) $(date)"
echo "warm=$WARM out=$OUT steps=$MAX_STEPS lr=$LR maxlen=$MAXLEN csv=$TRAIN_CSV"
[ -s "$TRAIN_CSV" ] || { echo "ERROR: missing $TRAIN_CSV"; exit 1; }
[ -s "$WARM/adapter_model.safetensors" ] || { echo "ERROR: missing warmstart $WARM"; exit 1; }
wc -l "$TRAIN_CSV"

# eff-batch = 2 * 2 * 4 = 16 (recipe). Warmstart inherits target_modules/rank from
# huikang-warmstart adapter_config (all attn+mamba+MoE+lm_head, rank 32).
accelerate launch --config_file $PROJECT/training/fsdp_config.yaml \
    $PROJECT/training/train.py \
    --output_dir        $OUT \
    --csv_path          $TRAIN_CSV \
    --resume_adapter_dir $WARM \
    --max_length        $MAXLEN \
    --max_steps         $MAX_STEPS \
    --lr                $LR \
    --batch_size        2 \
    --grad_acc          2 \
    --warmup_steps      5 \
    --save_steps        $SAVE_STEPS \
    --save_total_limit  -1 \
    --save_only_model \
    --seed              42
RC=$?
echo "brevity train exit=$RC $(date)"
echo "checkpoints:"; ls -d $OUT/checkpoint-* 2>/dev/null
exit $RC
