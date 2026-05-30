#!/bin/bash
# submit_and_wait.sh — After training completes, submit to Kaggle and poll LB.
set -uo pipefail

PROJECT=/scratch/jwj/NemotronFT
CKPT="$PROJECT/output/opd-v2"
MSG="opd-v2: 115 verified extended-DSL cryptarithm CoTs (3x oversample) on top of huikang baseline"
LOG_FILE=/scratch/jwj/NemotronFT/training/opd/solvers/runs/submit_log.txt

exec >> "$LOG_FILE" 2>&1
echo "============================================================"
echo "submit_and_wait.sh starting at $(date)"
echo "============================================================"

# Wait for training to produce adapter
while true; do
  if [ -f "$CKPT/adapter_model.safetensors" ] && [ -f "$CKPT/adapter_config.json" ]; then
    echo "Adapter ready at $(date)"
    break
  fi
  # Check if training is still running
  if ! squeue -h -j 5049623 -o "%T" 2>/dev/null | grep -q .; then
    echo "Training job 5049623 finished but adapter not at $CKPT/adapter_model.safetensors"
    # Try to find latest checkpoint
    LATEST=$(ls -d "$CKPT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST" ]; then
      echo "Found latest checkpoint: $LATEST. Copying to $CKPT/"
      cp "$LATEST/adapter_config.json" "$CKPT/adapter_config.json"
      cp "$LATEST/adapter_model.safetensors" "$CKPT/adapter_model.safetensors"
      break
    else
      echo "No checkpoint found. Aborting submission."
      exit 1
    fi
  fi
  sleep 60
done

# Build + submit
echo "Building submission and submitting at $(date)"
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN in the environment before running}"
cd "$PROJECT"
bash scripts/submit_checkpoint.sh "$CKPT" "$MSG"
echo "Submission complete at $(date)"

# Poll for LB result
echo "Polling Kaggle for score..."
module purge
module load python/3.11.5
source "$PROJECT/venv/bin/activate"
unset PIP_PREFIX
for i in $(seq 1 60); do
  sleep 60
  RESULT=$(kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge 2>/dev/null | head -5)
  SCORE=$(echo "$RESULT" | grep -F "$MSG" | head -1 | awk '{for(j=NF-1;j<=NF;j++) print $j}' || true)
  if [ -n "$SCORE" ] && [ "$SCORE" != "complete" ] && [ "$SCORE" != "pending" ]; then
    echo "[$(date)] Score returned: $SCORE"
    echo "FINAL: $RESULT" | head -3
    break
  fi
  echo "[$(date) poll $i/60] still scoring: $(echo "$RESULT" | head -2 | tail -1)"
done
echo "============================================================"
echo "submit_and_wait.sh done at $(date)"
