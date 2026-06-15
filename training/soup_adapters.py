"""Model-soup two (or more) PEFT LoRA adapters that share the SAME key structure
(12010 backbone-named keys, rank-32) by weighted-averaging their tensors.

All ingredients here are warmstart-related (multicat-GRPO was trained FROM
kienngx-086-renamed), so they live close in weight space -> the soup is a mild
interpolation ("delta shrink toward 0.86"), not a destructive merge. Averages in
fp32 then casts back to bf16 to match the file-submittable format.

Usage:
  python soup_adapters.py --out output/soup-086 \
      --inputs output/kienngx-086-renamed:0.5 training/rl/runs/bit_grpo_multicat/policy_iter1:0.5
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="dir:weight pairs, e.g. output/a:0.5 output/b:0.5")
    args = ap.parse_args()

    pairs = []
    for spec in args.inputs:
        d, w = spec.rsplit(":", 1)
        pairs.append((Path(d), float(w)))
    wsum = sum(w for _, w in pairs)
    pairs = [(d, w / wsum) for d, w in pairs]  # normalize
    print(f"[soup] {len(pairs)} ingredients (normalized weights):")
    for d, w in pairs:
        print(f"    {w:.3f}  {d}")

    # key set from the first ingredient
    st_paths = [(d / "adapter_model.safetensors", w) for d, w in pairs]
    with safe_open(str(st_paths[0][0]), framework="pt") as f:
        keys = list(f.keys())
    print(f"[soup] {len(keys)} keys")

    # sanity: all ingredients share the key set
    for p, _ in st_paths[1:]:
        with safe_open(str(p), framework="pt") as f:
            ks = set(f.keys())
        miss = set(keys) ^ ks
        if miss:
            raise SystemExit(f"[soup] key mismatch in {p}: {len(miss)} differing keys (e.g. {list(miss)[:3]})")

    # weighted average per key, fp32 accumulate -> bf16
    handles = [(safe_open(str(p), framework="pt"), w) for p, w in st_paths]
    out = {}
    for k in keys:
        acc = None
        for f, w in handles:
            t = f.get_tensor(k).to(torch.float32) * w
            acc = t if acc is None else acc + t
        out[k] = acc.to(torch.bfloat16)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(outdir / "adapter_model.safetensors"))
    # carry the config from the first ingredient (same rank/targets)
    shutil.copy(pairs[0][0] / "adapter_config.json", outdir / "adapter_config.json")
    total = sum(v.numel() for v in out.values())
    print(f"[soup] wrote {outdir}/adapter_model.safetensors ({len(out)} keys, {total:,} params, bf16)")


if __name__ == "__main__":
    main()
