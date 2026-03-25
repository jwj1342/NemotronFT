#!/usr/bin/env bash
# build_submission.sh — Package a trained LoRA adapter into submission.zip
#
# Usage:
#   bash build_submission.sh <lora_adapter_dir> [output_dir]
#
# Arguments:
#   lora_adapter_dir  Path to the directory containing adapter_config.json
#                     and adapter_model.safetensors (or adapter_model.bin)
#   output_dir        Where to write submission.zip (default: current directory)
#
# Example:
#   bash submission-pipeline/build_submission.sh output/lora-r32-v1 .

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
LORA_DIR="${1:-}"
OUTPUT_DIR="${2:-.}"

if [[ -z "$LORA_DIR" ]]; then
    echo "ERROR: No LoRA adapter directory specified."
    echo "Usage: $0 <lora_adapter_dir> [output_dir]"
    exit 1
fi

if [[ ! -d "$LORA_DIR" ]]; then
    echo "ERROR: Directory not found: $LORA_DIR"
    exit 1
fi

# ── Validation ─────────────────────────────────────────────────────────────
ADAPTER_CONFIG="$LORA_DIR/adapter_config.json"
if [[ ! -f "$ADAPTER_CONFIG" ]]; then
    echo "ERROR: adapter_config.json not found in $LORA_DIR"
    exit 1
fi

# Check LoRA rank <= 32
RANK=$(python3 -c "
import json, sys
with open('$ADAPTER_CONFIG') as f:
    cfg = json.load(f)
r = cfg.get('r', cfg.get('rank', None))
if r is None:
    print('UNKNOWN')
else:
    print(r)
")

if [[ "$RANK" == "UNKNOWN" ]]; then
    echo "WARNING: Could not determine LoRA rank from adapter_config.json"
elif [[ "$RANK" -gt 32 ]]; then
    echo "ERROR: LoRA rank is $RANK, but the competition requires rank <= 32."
    exit 1
else
    echo "LoRA rank: $RANK (OK)"
fi

# Check that adapter weights exist
WEIGHTS_FOUND=false
for f in "$LORA_DIR"/adapter_model.safetensors "$LORA_DIR"/adapter_model.bin; do
    if [[ -f "$f" ]]; then
        WEIGHTS_FOUND=true
        break
    fi
done
# Also accept sharded safetensors
if [[ "$WEIGHTS_FOUND" == false ]]; then
    if ls "$LORA_DIR"/adapter_model-*.safetensors 1>/dev/null 2>&1; then
        WEIGHTS_FOUND=true
    fi
fi

if [[ "$WEIGHTS_FOUND" == false ]]; then
    echo "ERROR: No adapter weights found in $LORA_DIR"
    echo "Expected: adapter_model.safetensors or adapter_model.bin"
    exit 1
fi

# ── Build ZIP ──────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
# Resolve to absolute path to avoid double-path issue when cd'ing into LORA_DIR
SUBMISSION_ZIP="$(realpath "$OUTPUT_DIR")/submission.zip"

# Remove old zip if present
rm -f "$SUBMISSION_ZIP"

echo "Packaging files from $LORA_DIR ..."

# Zip only the adapter files (not the full directory tree)
# The metric searches recursively for adapter_config.json, so flat layout is fine.
(
    cd "$LORA_DIR"
    zip -j "$SUBMISSION_ZIP" \
        adapter_config.json \
        $(ls adapter_model*.safetensors adapter_model*.bin 2>/dev/null || true) \
        $(ls tokenizer*.json tokenizer*.model special_tokens_map.json 2>/dev/null || true)
)

echo ""
echo "submission.zip contents:"
unzip -l "$SUBMISSION_ZIP"

echo ""
echo "Done. Submission saved to: $(realpath "$SUBMISSION_ZIP")"
