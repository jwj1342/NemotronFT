#!/usr/bin/env bash
# submit.sh — Submit submission.zip to the Kaggle competition
#
# Usage:
#   export KAGGLE_API_TOKEN=<your_token>
#   bash scripts/submit.sh [submission_zip] [message]
#
# Arguments:
#   submission_zip  Path to zip file (default: output/submission.zip)
#   message         Submission message (default: "Submission <timestamp>")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

SUBMISSION_ZIP="${1:-$PROJECT_DIR/output/submission.zip}"
MESSAGE="${2:-Submission $(date '+%Y-%m-%d %H:%M')}"
COMPETITION="nvidia-nemotron-model-reasoning-challenge"

# ── Checks ────────────────────────────────────────────────────────────────────
if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "ERROR: KAGGLE_API_TOKEN is not set."
    echo "Run: export KAGGLE_API_TOKEN=<your_token>"
    exit 1
fi

if [[ ! -f "$SUBMISSION_ZIP" ]]; then
    echo "ERROR: Submission file not found: $SUBMISSION_ZIP"
    exit 1
fi

echo "Competition : $COMPETITION"
echo "File        : $SUBMISSION_ZIP ($(du -sh "$SUBMISSION_ZIP" | cut -f1))"
echo "Message     : $MESSAGE"
echo ""

# ── Activate venv ─────────────────────────────────────────────────────────────
module purge
module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"
unset PIP_PREFIX

# Ensure kaggle >= 2.0.0 is installed (required for KGAT_ token support)
python -c "import kaggle; from packaging.version import Version; assert Version(kaggle.__version__) >= Version('2.0.0')" 2>/dev/null || \
    pip install --quiet kaggle==2.0.0

# ── Submit ────────────────────────────────────────────────────────────────────
kaggle competitions submit \
    -c "$COMPETITION" \
    -f "$SUBMISSION_ZIP" \
    -m "$MESSAGE"

echo ""
echo "=== Recent submissions ==="
kaggle competitions submissions -c "$COMPETITION" 2>/dev/null | head -10
