"""Strip a PEFT adapter saved by HF Trainer down to LoRA-only keys.

HF Trainer + PEFT 0.18 + FSDP ignored_modules saves the frozen
`lm_head.base_layer.weight` (the original 131072x2688 head, ~0.7-1.4 GB)
alongside the 12010 LoRA tensors, bloating the adapter past Kaggle's 2 GB
file-upload limit. The grader applies the LoRA delta on top of the base
model's own lm_head, so the frozen base copy is redundant — drop it.

Usage: python strip_lora.py <src_ckpt_dir> <dst_dir>
Copies adapter_config.json + tokenizer files, rewrites adapter_model.safetensors
with only the "lora_" keys (dtype preserved).
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".safetensors":
            continue
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    tensors = {}
    dropped = []
    cast = 0
    with safe_open(str(src / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if "lora_" in k:
                t = f.get_tensor(k)
                # HF Trainer saves LoRA in fp32 → 2x size (3.55 GB > Kaggle 2 GB limit).
                # Cast to bf16 (grader runs bf16 anyway; matches the warmstart adapter).
                if t.dtype == torch.float32:
                    t = t.to(torch.bfloat16); cast += 1
                tensors[k] = t
            else:
                dropped.append(k)
    save_file(tensors, str(dst / "adapter_model.safetensors"))
    sz = (dst / "adapter_model.safetensors").stat().st_size / 1e9
    print(f"[strip] {src.name}: kept {len(tensors)} LoRA keys ({sz:.2f} GB, {cast} cast fp32→bf16), "
          f"dropped {len(dropped)} non-LoRA {dropped}")


if __name__ == "__main__":
    main()
