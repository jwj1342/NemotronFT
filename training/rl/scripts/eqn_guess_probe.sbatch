#!/bin/bash
#SBATCH --job-name=eqg-probe
#SBATCH --account=aip-zhouyang
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error
# pass@k learnability probe for equation_numeric_guess (16% base, cliff category).
# Sample the true-0.86 model G=16 T=1.0 on all 136 eqn_guess problems; count how many
# have >=1 correct sample. If pass@16 >> 16% base -> correct samples ARE in support ->
# GRPO/STaR has gradient -> attackable (real 0.87 shot). If pass@16 ~ 16% (flat) ->
# under-determined like cryptarithm -> dead. Decides whether to invest a training run.
set -u
P=/scratch/jwj/NemotronFT; cd "$P"
POOL=$P/training/rl/runs/hotpool/eqn_guess_probe.jsonl
OUT=$P/training/rl/runs/eqn_guess_probe; mkdir -p "$OUT"
module purge; module load python/3.11.5 gcc/12.3 opencv/4.11.0
source $P/venv-vllm/bin/activate; unset PIP_PREFIX
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$P/.hf_home VLLM_CACHE_ROOT=$P/.vllm_cache CUDA_VISIBLE_DEVICES=0,1,2,3
python $P/training/rl/rollout.py --pool $POOL --adapter_dir $P/output/kienngx-086-renamed \
    --out $OUT/rollout.jsonl --stats_out $OUT/stats.json \
    --shard_index 0 --shard_total 2 \
    --group_size 16 --temperature 1.0 --top_p 1.0 --max_tokens 7680 --max_model_len 8192 \
    --max_num_seqs 96 --gpu_memory_utilization 0.85 --tensor_parallel_size 4 || exit 1
echo "=== eqn_guess probe rollout done $(date); pass@k stats ==="
python - <<'PY'
import json
recs=[json.loads(l) for l in open("/scratch/jwj/NemotronFT/training/rl/runs/eqn_guess_probe/rollout.jsonl") if l.strip()]
n=len(recs); pk=0; greedy=0; tot_correct=0
for r in recs:
    cs=[g for g in r["group"] if g["reward"]>0]
    if cs: pk+=1
    tot_correct+=len(cs)
print(f"[eqn_guess probe] {n} problems")
print(f"  pass@16 (>=1 correct sample): {pk}/{n} = {100*pk/max(1,n):.1f}%")
print(f"  total correct completions: {tot_correct} (avg {tot_correct/max(1,n):.2f}/problem of 16)")
print(f"  base greedy acc (reference): ~16%")
print(f"  VERDICT: pass@16 >> 16% => LEARNABLE (attack it); pass@16 ~ 16% => under-determined (dead)")
PY
