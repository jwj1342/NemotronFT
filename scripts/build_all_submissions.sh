#!/usr/bin/env bash
# scripts/build_all_submissions.sh — Build submission.zip for every checkpoint
# in a training run. Drops zips at $OUT_DIR/$ckpt_name/submission.zip.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${1:-$PROJECT_DIR/output/sft}"
OUT_DIR="${2:-$PROJECT_DIR/output/sft-submissions}"

mkdir -p "$OUT_DIR"

# Collect: every checkpoint-N + final adapter (if it exists at root)
CKPTS=()
if [[ -f "$TRAIN_DIR/adapter_config.json" ]]; then
    CKPTS+=("$TRAIN_DIR")
fi
while IFS= read -r d; do
    CKPTS+=("$d")
done < <(find "$TRAIN_DIR" -maxdepth 1 -name "checkpoint-*" -type d | sort -V)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "ERROR: no checkpoints found in $TRAIN_DIR"
    exit 1
fi

echo "Found ${#CKPTS[@]} adapter dirs to package."
for ckpt in "${CKPTS[@]}"; do
    name=$(basename "$ckpt")
    # If ckpt is the train_dir root, label as "final"
    if [[ "$ckpt" == "$TRAIN_DIR" ]]; then
        name="final"
    fi
    zip_dir="$OUT_DIR/$name"
    mkdir -p "$zip_dir"
    echo ""
    echo "→ $ckpt"
    bash "$PROJECT_DIR/scripts/build_submission.sh" "$ckpt" "$zip_dir"
done

echo ""
echo "All submission zips:"
ls -lh "$OUT_DIR"/*/submission.zip 2>/dev/null
