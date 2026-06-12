#!/bin/bash
#SBATCH --job-name=passk-probe
#SBATCH --account=aip-zhouyang
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# =============================================================================
# pass@k PROBE  — first GPU experiment of the recoverable-space plan (Codex step 1)
#
# Loads the Huikang baseline adapter under vLLM (TP=4, single node) and samples
# n completions per FAILED problem across four stratified categories, then scores
# each with the grader-aligned verify(). Output quantifies how much of the
# current 0.85-baseline's miss set is *recoverable* by answer-filtered
# self-training (RFT/STaR) vs genuinely information-limited.
#
# Read-only on the adapter. No training, no checkpoint writes.
#
# Tunables below mirror passk_probe.py argparse defaults; override as needed.
# IMPORTANT: do NOT add a vLLM SamplingParams seed (the probe never does) —
# vllm 0.17.1 collapses n>1 diversity with a fixed seed. See CLAUDE.md.
# =============================================================================

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
ADAPTER=$PROJECT_DIR/output/huikang-baseline-converted
OUT_DIR=$PROJECT_DIR/training/opd/runs/passk
mkdir -p "$OUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "============================================================"
echo "pass@k probe   Job: $SLURM_JOB_ID   Node: $SLURM_JOB_NODELIST"
echo "Start: $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "============================================================"

# --- inference env (venv-vllm). gcc/opencv MUST load before activate. --------
module purge
module load python/3.11.5 gcc/12.3 opencv/4.11.0
source "$PROJECT_DIR/venv-vllm/bin/activate"
unset PIP_PREFIX

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT_DIR/.hf_home"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# --- knobs (edit here, not in the .py) ---------------------------------------
PER_CAT=50          # failed problems sampled per category (statistically enough, ±~8%)
N=16                # completions per problem (pass@n)
TEMPERATURE=0.9     # T in [0.8, 1.0]; higher -> more diversity, lower fidelity
TOP_P=0.95
TOP_K=-1            # -1 disables top_k
MAX_TOKENS=7680     # grader budget
MAX_MODEL_LEN=8192  # grader budget
CATEGORIES="bit_manipulation,equation_numeric_guess,cryptarithm_deduce,cryptarithm_guess"
TAG="t${TEMPERATURE}_n${N}"

echo ""
echo "[run] per_cat=$PER_CAT n=$N T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
echo "[run] categories=$CATEGORIES  tag=$TAG"
echo ""

python "$PROJECT_DIR/training/opd/passk_probe.py" \
    --adapter_dir "$ADAPTER" \
    --out_dir "$OUT_DIR" \
    --categories "$CATEGORIES" \
    --per_cat "$PER_CAT" \
    --n "$N" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --max_tokens "$MAX_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --max_num_seqs 64 \
    --sample_seed 42 \
    --tag "$TAG"
RC=$?

echo ""
echo "============================================================"
echo "End: $(date)   exit=$RC"
echo "Outputs in: $OUT_DIR"
echo "============================================================"
exit $RC
