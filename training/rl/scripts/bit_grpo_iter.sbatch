#!/bin/bash
#SBATCH --job-name=bit-grpo-iter
#SBATCH --account=aip-zhouyang
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.error

# ============================================================================
# REAL iterative on-policy GRPO for bit (addresses the single-rollout critique).
# Each iteration RESAMPLES from the current policy, does a FEW near-on-policy
# steps with a KL-to-WARMSTART anchor (beta>0, ref_logp scored under the fixed
# warmstart adapter — RL's hypothesized edge over RFT), then evaluates T=0 net
# gain on the disjoint holdout. We plot accuracy/reward vs iteration in
# TensorBoard ($HOME/tensorboard_logs/bit_grpo_iter) and submit the best round.
#
# Resumable: a DONE marker is written per completed iteration; on restart the
# driver auto-detects START_ITER and continues (just resubmit this script).
#
# Overridable env: N_ITERS STEPS_PER_ITER GROUP_SIZE TEMP BETA LR POOL HOLDOUT
# ============================================================================
set -u
PROJECT=/scratch/jwj/NemotronFT
# Warmstart from the TRUE 0.86 base: kienngx-086-renamed = lossless kienngx adapter with
# keys renamed model.model->backbone (now IDENTICAL key set to huikang-warmstart → loads
# correctly in HF, no KL explosion) AND it FILE-SUBMITS at LB 0.86 (the earlier "bf16
# ceiling 0.85" was this key-naming bug, not precision). So GRPO now starts at a genuine
# 0.86 and bf16 outputs file-submit at 0.86 + training gain.
STAGE=${STAGE:-$PROJECT/training/rl/runs/bit_grpo_iter_086}
WARM=${WARM:-$PROJECT/output/kienngx-086-renamed}
POOL=${POOL:-$PROJECT/training/rl/runs/hotpool/bit_grpo_train96.jsonl}
HOLDOUT=${HOLDOUT:-$PROJECT/training/rl/runs/hotpool/bit_grpo_holdout.jsonl}
TB=${TB_DIR:-$HOME/tensorboard_logs/bit_grpo_iter_086}

N_ITERS=${N_ITERS:-5}
STEPS_PER_ITER=${STEPS_PER_ITER:-10}
GROUP_SIZE=${GROUP_SIZE:-8}
TEMP=${TEMP:-1.0}
BETA=${BETA:-0.02}
LR=${LR:-5e-6}
MAXLEN=${MAXLEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}   # KV cache holds ~151x@8192 → 64 fills it, ~2x rollout, no extra mem
VLLM_CACHE=${VLLM_CACHE:-$PROJECT/.vllm_cache}   # persist torch.compile across jobs (saves ~30s/load)
MN_ROLLOUT=${MN_ROLLOUT:-1}        # 1 = multi-node rollout+ref (both nodes via RoCE); 0 = single-node
# HPC: keep each job SHORT (4h walltime) — do a bounded # of iterations per job,
# checkpoint each (policy_iter{t}/DONE), then SELF-RESUBMIT to continue the chain.
MAX_ITERS_PER_JOB=${MAX_ITERS_PER_JOB:-1}
SELF_RESUBMIT=${SELF_RESUBMIT:-1}
THIS_SCRIPT=$PROJECT/training/rl/scripts/bit_grpo_iter.sbatch

mkdir -p "$STAGE/rollouts" "$PROJECT/logs" "$TB"
cd "$PROJECT"

echo "============================================================"
echo "ITERATIVE GRPO  job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  $(date)"
echo "N_ITERS=$N_ITERS steps/iter=$STEPS_PER_ITER G=$GROUP_SIZE T=$TEMP beta=$BETA lr=$LR maxlen=$MAXLEN"
echo "pool=$POOL  holdout=$HOLDOUT"
echo "warmstart(ref)=$WARM   stage=$STAGE   tb=$TB"
echo "============================================================"

# --- auto-detect resume point: highest k with policy_iter{k}/DONE ---
START_ITER=0
for k in $(seq 1 "$N_ITERS"); do
    [[ -f "$STAGE/policy_iter${k}/DONE" ]] && START_ITER=$k
done
echo "[driver] resuming at iteration $START_ITER (policy_iter${START_ITER} complete)"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29503

vllm_env() { module purge; module load python/3.11.5 gcc/12.3 opencv/4.11.0; source "$PROJECT/venv-vllm/bin/activate"; unset PIP_PREFIX; export VLLM_CACHE_ROOT="$VLLM_CACHE"; }
train_env() { module purge; module load python/3.11.5; source "$PROJECT/venv/bin/activate"; unset PIP_PREFIX; }

iters_done_this_job=0
for (( t=START_ITER; t<N_ITERS; t++ )); do
    if [[ $iters_done_this_job -ge $MAX_ITERS_PER_JOB ]]; then
        echo "[driver] hit MAX_ITERS_PER_JOB=$MAX_ITERS_PER_JOB for this job — will self-resubmit to continue."
        break
    fi
    # current policy: warmstart at t=0, else policy_iter{t}
    if [[ $t -eq 0 ]]; then CUR="$WARM"; else CUR="$STAGE/policy_iter${t}"; fi
    NEXT="$STAGE/policy_iter$((t+1))"
    ROLL="$STAGE/rollouts/iter${t}.jsonl"
    STATS="$STAGE/rollouts/iter${t}.stats.json"
    EVALJSON="$NEXT/holdout.json"
    echo ""
    echo "########## ITERATION $t  ($(date)) ##########"
    echo "[iter $t] current policy = $CUR  -> $NEXT"

    # guard: stop cleanly if not enough wall-time left for a full iteration
    REMAIN=$(squeue -h -j "$SLURM_JOB_ID" -o "%L" 2>/dev/null)
    echo "[iter $t] slurm time left: $REMAIN"

    # === 1/4 rollout + KL-ref scoring ===
    # MULTI-NODE (default): both nodes generate+score their own pool shard in parallel
    # over RoCE, then merge — ~2x throughput AND no idle node. Each node: rollout(TP4,
    # util 0.85) + score_ref_logp(TP4, util 0.55, separate engine → no logits OOM).
    rm -f ${ROLL}.part* 2>/dev/null
    if [[ $MN_ROLLOUT -eq 1 && $SLURM_NNODES -gt 1 ]]; then
        echo "[iter $t][1/4] MULTI-NODE rollout+ref (${SLURM_NNODES}×TP4, max_num_seqs=$MAX_NUM_SEQS) ..."
        srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 /bin/bash -lc "
            module purge; module load python/3.11.5 gcc/12.3 opencv/4.11.0
            source $PROJECT/venv-vllm/bin/activate; unset PIP_PREFIX
            export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
            export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home
            export VLLM_CACHE_ROOT=$VLLM_CACHE CUDA_VISIBLE_DEVICES=0,1,2,3
            python $PROJECT/training/rl/rollout.py \
                --pool $POOL --adapter_dir $CUR --out ${ROLL}.part\$SLURM_NODEID \
                --shard_index \$SLURM_NODEID --shard_total $SLURM_NNODES \
                --group_size $GROUP_SIZE --temperature $TEMP --top_p 1.0 \
                --max_tokens 7680 --max_model_len $MAXLEN --max_num_seqs $MAX_NUM_SEQS \
                --gpu_memory_utilization 0.85 --tensor_parallel_size 4 || exit 1
            python $PROJECT/training/rl/score_ref_logp.py \
                --rollout ${ROLL}.part\$SLURM_NODEID --ref_adapter_dir $WARM \
                --tensor_parallel_size 4 --max_model_len $MAXLEN \
                --gpu_memory_utilization 0.55 --max_num_seqs 8 || exit 1
        "
        RC=$?; [[ $RC -ne 0 ]] && { echo "[iter $t] multinode rollout/ref failed rc=$RC"; exit 1; }
        cat ${ROLL}.part* > "$ROLL"
    else
        echo "[iter $t][1/4] single-node rollout+ref (max_num_seqs=$MAX_NUM_SEQS) ..."
        vllm_env
        export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
        export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home CUDA_VISIBLE_DEVICES=0,1,2,3
        python "$PROJECT/training/rl/rollout.py" \
            --pool "$POOL" --adapter_dir "$CUR" --out "$ROLL" \
            --group_size "$GROUP_SIZE" --temperature "$TEMP" --top_p 1.0 \
            --max_tokens 7680 --max_model_len "$MAXLEN" --max_num_seqs "$MAX_NUM_SEQS" \
            --gpu_memory_utilization 0.85 --tensor_parallel_size 4 || { echo "[iter $t] rollout failed"; exit 1; }
        python "$PROJECT/training/rl/score_ref_logp.py" \
            --rollout "$ROLL" --ref_adapter_dir "$WARM" \
            --tensor_parallel_size 4 --max_model_len "$MAXLEN" \
            --gpu_memory_utilization 0.55 --max_num_seqs 8 || { echo "[iter $t] ref scoring failed"; exit 1; }
    fi
    # aggregate rollout stats from the (merged) jsonl for TB collapse-watch
    python "$PROJECT/training/rl/compute_rollout_stats.py" "$ROLL" "$STATS" 7680
    # log rollout stats + COLLAPSE-WATCH metrics to TB at step=t (UNQUOTED $() so
    # the key=val tokens word-split into separate argv for tb_log.py nargs="+").
    RSTATS=$(python - "$STATS" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"rollout/reward_mean={d['reward_mean']} rollout/reward_var={d['reward_var']} "
      f"rollout/frac_mixed_groups={d['frac_mixed_groups']} rollout/resp_len_mean={d['resp_len_mean']} "
      f"rollout/frac_truncated={d['frac_truncated']} rollout/boxed_rate={d['boxed_rate']} "
      f"rollout/distinct_ratio={d['distinct_ratio']}")
PY
)
    python "$PROJECT/training/rl/tb_log.py" --dir "$TB" --step "$t" $RSTATS
    sleep 15

    # === 2/3 GRPO train, beta>0 (KL-to-warmstart), few near-on-policy steps ===
    echo "[iter $t][2/4] GRPO $STEPS_PER_ITER steps beta=$BETA across 2x4 L40s ..."
    OFFSET=$(( t * STEPS_PER_ITER ))
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
        /bin/bash -lc "
            module purge; module load python/3.11.5
            source $PROJECT/venv/bin/activate; unset PIP_PREFIX
            export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True
            export PYTHONUNBUFFERED=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=bnxt_re0,bnxt_re1 NCCL_DEBUG=WARN
            accelerate launch \
                --config_file $PROJECT/training/fsdp_multinode_config.yaml \
                --num_machines $SLURM_NNODES --num_processes \$((SLURM_NNODES * 4)) \
                --machine_rank \$SLURM_NODEID \
                --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
                $PROJECT/training/rl/train_grpo.py \
                    --rollout_jsonl $ROLL --adapter_dir $CUR --output_dir $NEXT \
                    --lr $LR --beta $BETA --batch_size 1 --grad_acc 8 \
                    --max_steps $STEPS_PER_ITER --max_length $MAXLEN --seed 1234 \
                    --tb_dir $TB --global_step_offset $OFFSET
        "
    RC=$?; [[ $RC -ne 0 ]] && { echo "[iter $t] train failed rc=$RC"; exit 1; }
    [[ -f "$NEXT/adapter_model.safetensors" ]] || { echo "[iter $t] no adapter produced"; exit 1; }
    sleep 15

    # === 3/3 T=0 holdout net-gain eval (warmstart vs this iteration) ===
    echo "[iter $t][3/4] T=0 holdout net-gain eval ..."
    vllm_env
    export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=$PROJECT/.hf_home CUDA_VISIBLE_DEVICES=0,1,2,3
    python "$PROJECT/training/rl/eval_bit_holdout.py" \
        --holdout "$HOLDOUT" --warm "$WARM" --trained "$NEXT" --tp 4 --out_json "$EVALJSON"
    RC=$?; [[ $RC -ne 0 ]] && { echo "[iter $t] eval failed rc=$RC"; exit 1; }
    HSTATS=$(python - "$EVALJSON" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"holdout/net={d['net']} holdout/recover={d['recover']} holdout/regress={d['regress']} "
      f"holdout/warm_overall={d['warm_overall']} holdout/trained_overall={d['trained_overall']}")
PY
)
    python "$PROJECT/training/rl/tb_log.py" --dir "$TB" --step "$t" $HSTATS

    touch "$NEXT/DONE"
    iters_done_this_job=$(( iters_done_this_job + 1 ))
    echo "[iter $t] DONE -> $NEXT  ($(date))   (iters this job: $iters_done_this_job)"
done

echo ""
echo "=== iterative GRPO segment finished at $(date) ==="
echo "per-iteration holdout net-gain so far:"
HIGHEST_DONE=0
for k in $(seq 1 "$N_ITERS"); do
    if [[ -f "$STAGE/policy_iter${k}/holdout.json" ]]; then
        echo "  policy_iter${k}: $(cat $STAGE/policy_iter${k}/holdout.json)"
        HIGHEST_DONE=$k
    fi
done

# --- self-resubmit chain: short jobs, checkpoint each iter, continue until N_ITERS ---
if [[ $HIGHEST_DONE -lt $N_ITERS && $SELF_RESUBMIT -eq 1 ]]; then
    echo "[driver] $HIGHEST_DONE/$N_ITERS iterations done — resubmitting to continue the chain."
    NEXTJOB=$(sbatch --parsable "$THIS_SCRIPT")
    echo "[driver] resubmitted as job $NEXTJOB"
else
    echo "[driver] chain complete ($HIGHEST_DONE/$N_ITERS) or SELF_RESUBMIT=0 — stopping."
fi
