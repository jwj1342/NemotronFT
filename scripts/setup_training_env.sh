#!/usr/bin/env bash
# setup_training_env.sh — Install training dependencies into the project venv
#
# Usage: source scripts/setup_training_env.sh
#   (or)  bash scripts/setup_training_env.sh
#
# Must be run AFTER loading the python module:
#   module load python/3.11.5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creating virtualenv at $VENV_DIR ..."
    virtualenv --no-download "$VENV_DIR"
fi

echo "[setup] Activating venv ..."
source "$VENV_DIR/bin/activate"
unset PIP_PREFIX

echo "[setup] Installing training packages (cluster wheels only) ..."
pip install --no-index --upgrade pip --quiet

# Step 1: install mamba-ssm first so pip resolves torch to the version
# the CUDA extensions were compiled against (avoids ABI mismatch).
pip install --no-index mamba-ssm causal-conv1d

# Step 2: install the rest against the already-pinned torch version.
pip install --no-index transformers peft accelerate einops

echo "[setup] Done. Installed packages:"
pip list | grep -E "torch|mamba|causal|transform|peft|acceler|einops"
