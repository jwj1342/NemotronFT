#!/bin/bash
#SBATCH --job-name=rl-smoke-mn
#SBATCH --account=aip-zhouyang
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/jwj/NemotronFT/logs/%x-%j.out
#SBATCH --error=/scratch/jwj/NemotronFT/logs/%x-%j.err

# Quick 2-node NCCL all-reduce sanity. Confirms the cross-node interconnect
# is the path NCCL picks. If bandwidth < 1 GB/s, NCCL is on the management
# network and the multi-node FSDP run will be ~unusable.
# Writes MULTINODE_OK sentinel on success.

set -u
PROJECT_DIR=/scratch/jwj/NemotronFT
STAGE_DIR=$PROJECT_DIR/training/rl/runs/stage0
mkdir -p "$STAGE_DIR" "$PROJECT_DIR/logs"

echo "============================================================"
echo "Multi-node NCCL smoke"
echo "Job: $SLURM_JOB_ID  Nodes: $SLURM_NNODES  $(date)"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "============================================================"

module purge && module load python/3.11.5
source "$PROJECT_DIR/venv/bin/activate"; unset PIP_PREFIX

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
# Vulcan: 100 Gbps Ethernet + RoCE (RDMA over Converged Ethernet). Tell NCCL
# to use IB-class transport (covers RoCE), not TCP sockets. Auto-detect the
# Mellanox HCA. If auto-detect fails, set NCCL_IB_HCA=mlx5_0 (or whatever
# `ibstat` shows) and NCCL_IB_GID_INDEX=3.
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB

echo "[net] interfaces on node 0:"
ip -br link | head -20
echo "[net] looking for RDMA devices ..."
ls /sys/class/infiniband/ 2>/dev/null || echo "(none under /sys/class/infiniband)"
ls /sys/class/net/ 2>/dev/null | head -10

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
srun --label --nodes="$SLURM_NNODES" --ntasks-per-node=4 \
    --cpus-per-task=4 \
    /bin/bash -lc "
        source $PROJECT_DIR/venv/bin/activate
        unset PIP_PREFIX
        export RANK=\$SLURM_PROCID
        export WORLD_SIZE=\$SLURM_NTASKS
        export LOCAL_RANK=\$SLURM_LOCALID
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        export NCCL_DEBUG=$NCCL_DEBUG
        python $PROJECT_DIR/training/rl/scripts/nccl_allreduce_test.py
    "
RC=$?

if [[ $RC -eq 0 ]]; then
    touch "$STAGE_DIR/MULTINODE_OK"
    echo ""
    echo "✅ MULTINODE_OK sentinel written: $STAGE_DIR/MULTINODE_OK"
    echo "   (Inspect log: bandwidth >1 GB/s = good; <200 MB/s = wrong interface)"
else
    echo ""
    echo "❌ Multi-node NCCL test failed (rc=$RC)"
    exit 1
fi

echo ""
echo "=== Done at $(date) ==="
