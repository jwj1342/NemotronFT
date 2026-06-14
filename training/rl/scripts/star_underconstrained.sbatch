#!/bin/bash
#SBATCH --job-name=star-uc
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error
# STaR on the under-constrained (7-example) bit slice: sample the 0.86 model G=16 T=1.0,
# keep CORRECT completions (the model's own long CoT, incl. high-T solves of currently-
# failed problems) -> training target for prior calibration. Multi-node, flaky excluded.
set -u
P=/scratch/jwj/NemotronFT; cd "$P"
POOL=$P/training/rl/runs/hotpool/bit_underconstrained_7ex.jsonl
OUT=$P/training/rl/runs/star_uc; mkdir -p "$OUT"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 /bin/bash -lc "
    module purge; module load python/3.11.5 gcc/12.3 opencv/4.11.0
    source $P/venv-vllm/bin/activate; unset PIP_PREFIX
    export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$P/.hf_home VLLM_CACHE_ROOT=$P/.vllm_cache CUDA_VISIBLE_DEVICES=0,1,2,3
    python $P/training/rl/rollout.py --pool $POOL --adapter_dir $P/output/kienngx-086-renamed \
        --out $OUT/rollout.part\$SLURM_NODEID.jsonl --stats_out $OUT/stats.part\$SLURM_NODEID.json \
        --shard_index \$SLURM_NODEID --shard_total $SLURM_NNODES \
        --group_size 16 --temperature 1.0 --top_p 1.0 --max_tokens 7680 --max_model_len 8192 \
        --max_num_seqs 48 --gpu_memory_utilization 0.85 --tensor_parallel_size 4 || exit 1
"
RC=$?; [[ $RC -ne 0 ]] && { echo "rollout failed $RC"; exit 1; }
cat $OUT/rollout.part*.jsonl > $OUT/rollout.jsonl
echo "=== STaR rollout done $(date); building correct-CoT stats ==="
python - <<'PY'
import json
recs=[json.loads(l) for l in open("/scratch/jwj/NemotronFT/training/rl/runs/star_uc/rollout.jsonl") if l.strip()]
n_prob=len(recs); n_with_correct=0; n_fail_solved=0; total_correct=0
for r in recs:
    cs=[g for g in r["group"] if g["reward"]>0]
    if cs: n_with_correct+=1
    total_correct+=len(cs)
print(f"[star] {n_prob} problems, {n_with_correct} have >=1 correct sample (pass@16), {total_correct} total correct completions")
PY
