#!/bin/bash
#SBATCH --job-name=star-mc
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error
# Multi-category STaR: sample the true-0.86 model G=16 T=1.0 over the BROAD multicat
# pool (bit 40 + eqn_deduce 40 = gradient-bearing; cipher/gravity/numeral/unit 12 each =
# saturated anchors). Keep CORRECT completions -> SFT target. This is the SFT analog of
# multicat-GRPO (which HELD LB 0.86 where single-cat regressed): tests whether BROAD-
# distribution self-distillation anchors the shared prior (holds) or still drifts bit
# down like single-cat STaR did (regressed to 0.85). Completes the single/multi x SFT/GRPO 2x2.
set -u
P=/scratch/jwj/NemotronFT; cd "$P"
POOL=$P/training/rl/runs/hotpool/multicat_grpo_train.jsonl
OUT=$P/training/rl/runs/star_multicat; mkdir -p "$OUT"
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
echo "=== STaR multicat rollout done $(date); building correct-CoT SFT data ==="
module purge; module load python/3.11.5; source $P/venv/bin/activate; unset PIP_PREFIX
python - <<'PY'
import json, collections
SRC="/scratch/jwj/NemotronFT/training/rl/runs/star_multicat/rollout.jsonl"
DST="/scratch/jwj/NemotronFT/training/rl/runs/star_multicat/sft_rollout.jsonl"
recs=[json.loads(l) for l in open(SRC) if l.strip()]
out=[]; per_cat=collections.Counter(); n_correct_total=0; n_recovered=0
# baseline greedy correctness is unknown here; "recovered" = a problem whose T=1.0 group
# has correct samples but we cap kept completions to avoid over-weighting easy problems.
CAP=4  # keep at most 4 correct completions per problem (balance saturated vs hard cats)
for r in recs:
    correct=[g for g in r["group"] if g.get("reward",0)>0]
    if not correct: continue
    correct=correct[:CAP]
    for g in correct: g["advantage"]=1.0
    n_correct_total+=len(correct); per_cat[r["category"]]+=1
    out.append({**r, "group": correct})
with open(DST,"w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print(f"[star-mc] {len(out)} problems with >=1 correct -> {n_correct_total} kept completions (cap={CAP})")
print("[star-mc] per-category problems kept:")
for k,v in per_cat.most_common(): print(f"    {k}: {v}")
PY
echo "=== sft_rollout built $(date) ==="
ls -la $OUT/sft_rollout.jsonl
