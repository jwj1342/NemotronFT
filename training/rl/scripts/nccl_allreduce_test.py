"""Tiny multi-node NCCL all-reduce bandwidth probe.

Run via srun + accelerate-launch-style env vars: RANK, WORLD_SIZE, LOCAL_RANK,
MASTER_ADDR, MASTER_PORT.
"""
import os
import time
import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    n_mb = 64  # 64 MiB tensor (fp32, so 64 * 1024 * 1024 / 4 = 16M elements)
    x = torch.randn(n_mb * 1024 * 1024 // 4, device=device)

    # Warm up
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    n_iter = 10
    t0 = time.time()
    for _ in range(n_iter):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    alg_bw_mbps = (n_mb * n_iter) / elapsed
    bus_bw_mbps = alg_bw_mbps * 2 * (ws - 1) / ws  # standard NCCL bus-BW formula

    print(
        f"rank {rank}/{ws}  host={os.uname().nodename}  cuda={local_rank}  "
        f"{n_iter}x {n_mb}MB allreduce: total={elapsed*1000:.0f}ms  "
        f"alg_bw={alg_bw_mbps:.0f} MB/s  bus_bw={bus_bw_mbps:.0f} MB/s",
        flush=True,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
