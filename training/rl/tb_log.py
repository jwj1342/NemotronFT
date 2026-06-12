"""Tiny CLI to append scalar(s) to a TensorBoard run dir at a given step.

Used by the iterative-GRPO driver (bash) to log per-iteration rollout + holdout
metrics into the SAME run dir that train_grpo.py writes per-step training curves
to, so everything shows up under one TensorBoard run.

Usage:
  python tb_log.py --dir $HOME/tensorboard_logs/bit_grpo_iter --step 0 \
      rollout/reward_mean=0.51 rollout/reward_var=0.25 holdout/net=3
"""
from __future__ import annotations
import argparse
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("kv", nargs="+", help="tag=value pairs (value parsed as float)")
    args = ap.parse_args()
    w = SummaryWriter(log_dir=args.dir)
    for pair in args.kv:
        if "=" not in pair:
            print(f"[tb_log] skip malformed pair: {pair}")
            continue
        tag, val = pair.split("=", 1)
        try:
            w.add_scalar(tag, float(val), args.step)
        except ValueError:
            print(f"[tb_log] skip non-float: {pair}")
    w.flush()
    w.close()
    print(f"[tb_log] wrote {len(args.kv)} scalar(s) to {args.dir} @ step {args.step}")


if __name__ == "__main__":
    main()
