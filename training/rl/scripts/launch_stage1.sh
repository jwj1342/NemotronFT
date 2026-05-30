#!/bin/bash
# Convenience: launch Stage 1 RL (equation_numeric_guess).
# Requires Stage 0 smoke to have passed (SMOKE_OK sentinel).
set -eu

PROJECT_DIR=/scratch/jwj/NemotronFT
SMOKE_OK="$PROJECT_DIR/training/rl/runs/stage0/SMOKE_OK"

if [[ ! -f "$SMOKE_OK" ]]; then
    echo "❌ Stage 0 smoke has not passed yet — refusing to launch Stage 1."
    echo "   Missing: $SMOKE_OK"
    exit 1
fi

echo "✅ Smoke OK — submitting Stage 1 (equation_numeric_guess)"
sbatch --export=ALL,\
STAGE_NAME=stage1_equation_numeric_guess,\
STAGE_CATEGORIES=equation_numeric_guess,equation_numeric_deduce,\
N_OUTER=12,N_INNER=50,GROUP_SIZE=8,LR=5e-6,BETA=0.05,CAP_PER_CAT=0 \
    "$PROJECT_DIR/training/rl/scripts/iter.sbatch"
