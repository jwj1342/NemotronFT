#!/usr/bin/env bash
# scripts/submit_checkpoint.sh — Build + submit a checkpoint to Kaggle.
# Checks for same-day duplicate messages (preserves the 5/day quota).
#
# Usage:
#   bash scripts/submit_checkpoint.sh <checkpoint_dir> "<message>"
# Example:
#   bash scripts/submit_checkpoint.sh output/huikang-baseline-converted \
#        "strong baseline: Huikang adapter SVD-converted, 0.85 reference"

set -euo pipefail

CKPT="${1:-}"
MSG="${2:-submission $(date '+%Y-%m-%d %H:%M')}"

if [[ -z "$CKPT" ]]; then
    echo "Usage: $0 <checkpoint_dir> [message]"
    exit 1
fi
if [[ ! -f "$CKPT/adapter_config.json" ]]; then
    echo "ERROR: $CKPT/adapter_config.json not found."
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_DIR="$(mktemp -d)"

echo "→ Building submission zip from $CKPT ..."
bash "$PROJECT_DIR/scripts/build_submission.sh" "$CKPT" "$ZIP_DIR"
ZIP_PATH="$ZIP_DIR/submission.zip"

echo "→ Checking today's Kaggle submissions for duplicates ..."
module purge
module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"
unset PIP_PREFIX
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-KGAT_04bcfc5348e1637ed220119fafd3904b}"
TODAY=$(date '+%Y-%m-%d')
DUPE=$(kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge 2>/dev/null \
       | grep "$TODAY" | grep -F "$MSG" || true)
if [[ -n "$DUPE" ]]; then
    echo "WARNING: A submission with the same message already exists today:"
    echo "$DUPE"
    echo "Aborting to preserve daily quota."
    exit 1
fi

echo "→ Submitting ..."
bash "$PROJECT_DIR/scripts/submit.sh" "$ZIP_PATH" "$MSG"

rm -rf "$ZIP_DIR"
echo "→ Done."
