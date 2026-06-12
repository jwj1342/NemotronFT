#!/bin/bash
#SBATCH --job-name=crd-warm86
#SBATCH --account=aip-zhouyang
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/crd-warm86-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/crd-warm86-%j.err

# Cryptarithm-cipher attack, take 2: WARMSTART FROM 0.86 (not BASE like the 6% probe).
# Same 4226 bounded-search cipher-deduce traces (median 629 tok = genuinely long, vs
# fingerprint's 118 that collapsed). LOW lr 1e-5 (fingerprint died at 1e-4) + frequent
# checkpoints so eval_sweep can find the pre-collapse sweet spot. crypt-only first;
# if other categories collapse too fast, retrain with rehearsal.
set -u
PROJECT=/scratch/jwj/NemotronFT
OUT=${CRD_OUT:-$PROJECT/output/crd-warm86}
WARM=${CRD_WARM:-$PROJECT/output/kienngx-086-bf16}
TRAIN_JSONL=${CRD_TRAIN:-$PROJECT/data/crd_sft_train.jsonl}
LR=${CRD_LR:-1e-5}
MAXLEN=${CRD_MAXLEN:-1536}
MAX_STEPS=${CRD_STEPS:-240}
SAVE_STEPS=${CRD_SAVE_STEPS:-20}
mkdir -p "$OUT" "$PROJECT/logs"; cd "$PROJECT"

module purge; module load python/3.11.5
source $PROJECT/venv/bin/activate; unset PIP_PREFIX
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

echo "=== CRD cipher attack from 0.86 === job $SLURM_JOB_ID $(hostname) $(date)"
echo "warm=$WARM out=$OUT lr=$LR maxlen=$MAXLEN max_steps=$MAX_STEPS save=$SAVE_STEPS"

# JSONL -> CSV (dataset.py CoT cols: id,prompt,answer,generated_cot,type)
CSV=$OUT/crd_train.csv
python3 - "$TRAIN_JSONL" "$CSV" <<'PY'
import csv, json, sys
src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src) if l.strip()]
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id","prompt","answer","generated_cot","type"])
    w.writeheader()
    for i, r in enumerate(rows):
        cot = r.get("cot") or r.get("generated_cot") or ""
        w.writerow({"id": r.get("id", i), "prompt": r["prompt"], "answer": r["answer"],
                    "generated_cot": cot, "type": r.get("type", "cryptarithm_deduce")})
print(f"[convert] {len(rows)} rows -> {dst}", flush=True)
PY
[ -s "$CSV" ] || { echo "ERROR: empty CSV"; exit 1; }

accelerate launch --config_file $PROJECT/training/fsdp_config.yaml \
    $PROJECT/training/train.py \
    --output_dir $OUT --csv_path $CSV --resume_adapter_dir $WARM \
    --max_length $MAXLEN --max_steps $MAX_STEPS --lr $LR \
    --batch_size 2 --grad_acc 4 --warmup_steps 5 \
    --save_steps $SAVE_STEPS --save_total_limit -1 --save_only_model --seed 42
RC=$?
echo "crd-warm86 train exit=$RC $(date)"; ls -d $OUT/checkpoint-* 2>/dev/null
exit $RC
