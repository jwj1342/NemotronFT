#!/bin/bash
#SBATCH --job-name=crd-probe
#SBATCH --account=aip-zhouyang
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/crd-probe-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/crd-probe-%j.err

# Cryptarithm_deduce EXECUTABILITY PROBE (gate #21).
# Warmstart from BASE (fresh max-capacity rank-32 LoRA), train crypt-ONLY on the
# bounded-search CoT traces (data/crd_sft_train.jsonl), checkpoint sweep so eval
# can find the overfit sweet-spot. NO KL / NO replay (Phase 1 handles preservation).
# Decision gate is run separately by eval_crd_probe.py.

set -u
PROJECT=/scratch/jwj/NemotronFT
OUT=${CRD_OUT:-$PROJECT/output/crd-probe}
TRAIN_JSONL=${CRD_TRAIN:-$PROJECT/data/crd_sft_train.jsonl}
# traces are short (max ~1180 tok cot + short prompt) -> 2048 covers all, big speedup
EPOCHS=${CRD_EPOCHS:-3}
LR=${CRD_LR:-2e-4}
MAXLEN=${CRD_MAXLEN:-2048}
SAVE_STEPS=${CRD_SAVE_STEPS:-25}
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

echo "=== CRD executability probe ==="
echo "Job $SLURM_JOB_ID $(hostname) $(date) out=$OUT epochs=$EPOCHS lr=$LR maxlen=$MAXLEN"

# --- JSONL -> CSV for dataset.py CoT mode (cols: id,prompt,answer,generated_cot,type) ---
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
        w.writerow({"id": r.get("id", i), "prompt": r["prompt"],
                    "answer": r["answer"], "generated_cot": cot,
                    "type": r.get("type", "cryptarithm_deduce")})
print(f"[convert] {len(rows)} rows -> {dst}", flush=True)
PY
[ -s "$CSV" ] || { echo "ERROR: empty CSV (train jsonl missing?)"; exit 1; }

# Fresh max-capacity LoRA from BASE (no --resume_adapter_dir): default target set
# = q/k/v/o + Mamba in_proj/out_proj + per-expert up/down (+ shared-expert via
# suffix) + lm_head. rank 32, alpha 32. crypt-only, checkpoint sweep.
accelerate launch --config_file $PROJECT/training/fsdp_config.yaml \
    $PROJECT/training/train.py \
    --output_dir       $OUT \
    --csv_path         $CSV \
    --max_length       $MAXLEN \
    --num_epochs       $EPOCHS \
    --lr               $LR \
    --batch_size       2 \
    --grad_acc         4 \
    --lora_rank        32 \
    --lora_alpha       32 \
    --lora_dropout     0.0 \
    --warmup_steps     5 \
    --save_steps       $SAVE_STEPS \
    --save_total_limit -1 \
    --save_only_model \
    --seed             42
RC=$?
echo "probe train exit=$RC $(date)"
echo "checkpoints:"; ls -d $OUT/checkpoint-* 2>/dev/null
exit $RC
