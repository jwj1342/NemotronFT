#!/bin/bash
set -uo pipefail
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN in the environment before running}"
LOG=/scratch/jwj/NemotronFT/training/opd/solvers/runs/kaggle_poll.log
MSG_KEY="opd-v2: huikang baseline + 115"

exec >> "$LOG" 2>&1
echo "=== Poll start $(date) ==="
module purge >/dev/null 2>&1
module load python/3.11.5 >/dev/null 2>&1
source /scratch/jwj/NemotronFT/venv/bin/activate
unset PIP_PREFIX

for i in $(seq 1 90); do
  OUT=$(kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge 2>/dev/null | grep -F "$MSG_KEY" | head -1)
  echo "[$(date) iter $i] $OUT"
  if echo "$OUT" | grep -q "COMPLETE"; then
    SCORE=$(echo "$OUT" | awk '{print $(NF-1)}')
    echo "=== FINAL SCORE: $SCORE ==="
    break
  fi
  if echo "$OUT" | grep -qi "ERROR\|FAIL"; then
    echo "=== SUBMISSION ERRORED ==="
    break
  fi
  sleep 60
done
echo "=== Poll end $(date) ==="
