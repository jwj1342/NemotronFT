#!/bin/bash
# Download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 from HuggingFace
# Usage: bash download_model.sh [HF_TOKEN]

set -e

HF_TOKEN="${1:-$HF_TOKEN}"
MODEL_DIR="$(dirname "$0")/model"
VENV_DIR="$(dirname "$0")/venv"

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HuggingFace token required."
    echo "Usage: bash download_model.sh <HF_TOKEN>"
    echo "   or: export HF_TOKEN=<token> && bash download_model.sh"
    exit 1
fi

# Load Python module
module load python/3.11.5

# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating virtual environment..."
    virtualenv --no-download "$VENV_DIR"
fi

# Activate and install dependencies
echo "[2/3] Installing huggingface_hub..."
source "$VENV_DIR/bin/activate"
unset PIP_PREFIX
pip install --no-index --upgrade pip --quiet
pip install --no-index huggingface_hub --quiet

# Download model
echo "[3/3] Downloading model to $MODEL_DIR ..."
mkdir -p "$MODEL_DIR"
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir='$MODEL_DIR',
    token='$HF_TOKEN'
)
print('Download complete!')
"

deactivate
