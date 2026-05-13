"""
Convert Huikang's Tinker-trained adapter to Kaggle-submission format.

3 steps (ported from resource/tinker-submission-notebook.ipynb cell 24):
  1. Rename keys: base_model.model.model.* -> base_model.model.backbone.*
  2. Unfuse experts.w1/w2 -> per-expert up_proj/down_proj (broadcast)
  3. SVD-merge gate_proj + x_proj LoRA pair -> in_proj LoRA pair (rank 32, lossy)

Output: adapter ready to drop into a submission zip.

Usage:
  python convert_tinker_adapter.py \
      --in_dir  /scratch/jwj/NemotronFT/output/huikang-baseline \
      --out_dir /scratch/jwj/NemotronFT/output/huikang-baseline-converted
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def trained_adapter_key_rename(key_name: str) -> str:
    return key_name.replace("base_model.model.model", "base_model.model.backbone")


def collect_model_key_shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for shard in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes


def convert(in_dir: Path, out_dir: Path, model_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model key shapes (for in_proj dimension lookup).
    print(f"[1/4] Collecting model tensor shapes from {model_dir} ...")
    model_key_shapes = collect_model_key_shapes(model_dir)
    print(f"      {len(model_key_shapes)} model keys collected.")

    # 2. Load adapter tensors.
    in_safetensors = in_dir / "adapter_model.safetensors"
    print(f"\n[2/4] Loading adapter tensors from {in_safetensors} ...")
    adapter_tensors = {}
    with safe_open(in_safetensors, framework="pt", device="cpu") as f:
        for key in f.keys():
            adapter_tensors[key] = f.get_tensor(key)
    print(f"      {len(adapter_tensors)} adapter tensors loaded.")

    # 3. Build conversion plan.
    base_names = set()
    for key in adapter_tensors:
        base = re.sub(r"\.lora_[AB]\.weight$", "", key)
        base_names.add(base)

    mamba_merge_layers: dict[str, dict[str, str]] = {}
    for base in base_names:
        for proj in ("gate_proj", "x_proj"):
            if f".{proj}" in base:
                layer_path = base.rsplit(f".{proj}", 1)[0]
                mamba_merge_layers.setdefault(layer_path, {})[proj] = base
    mamba_merge_bases = set()
    for projs in mamba_merge_layers.values():
        mamba_merge_bases.update(projs.values())

    # 4. Execute conversion.
    print(f"\n[3/4] Converting tensors ...")
    tensors: dict[str, torch.Tensor] = {}

    for base in sorted(base_names):
        lora_A = adapter_tensors[f"{base}.lora_A.weight"]
        lora_B = adapter_tensors[f"{base}.lora_B.weight"]
        renamed = trained_adapter_key_rename(base)

        if ".experts.w3" in base and lora_A.numel() == 0:
            continue
        if base in mamba_merge_bases:
            continue  # handled in SVD pass

        # Expert unfusing: w1 -> per-expert up_proj, w2 -> per-expert down_proj
        if ".experts.w1" in base or ".experts.w2" in base:
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()
            num_experts = lora_A.shape[0]
            proj_name = "up_proj" if ".w1" in base else "down_proj"
            for i in range(num_experts):
                exp_renamed = re.sub(
                    r"\.experts\.w[12]",
                    f".experts.{i}.{proj_name}",
                    renamed,
                )
                tensors[f"{exp_renamed}.lora_A.weight"] = lora_A[i].contiguous()
                tensors[f"{exp_renamed}.lora_B.weight"] = lora_B[i].contiguous()
            continue

        # Direct rename for attn / lm_head / out_proj.
        tensors[f"{renamed}.lora_A.weight"] = lora_A
        tensors[f"{renamed}.lora_B.weight"] = lora_B

    # SVD merge gate_proj + x_proj -> in_proj for every Mamba layer.
    print(f"\n[4/4] SVD-merging Mamba gate_proj + x_proj -> in_proj for "
          f"{len(mamba_merge_layers)} layers ...")
    kept_total, mass_total = 0.0, 0.0
    for layer_path, projs in sorted(mamba_merge_layers.items()):
        renamed_layer = trained_adapter_key_rename(layer_path)
        in_proj_base = f"{renamed_layer}.in_proj"
        model_in_proj_key = (
            renamed_layer.replace("base_model.model.", "") + ".in_proj.weight"
        )
        if model_in_proj_key not in model_key_shapes:
            raise KeyError(f"Could not find {model_in_proj_key} in model")
        in_proj_dim = model_key_shapes[model_in_proj_key][0]

        gate_A = adapter_tensors[f"{projs['gate_proj']}.lora_A.weight"].float()
        gate_B = adapter_tensors[f"{projs['gate_proj']}.lora_B.weight"].float()
        x_A = adapter_tensors[f"{projs['x_proj']}.lora_A.weight"].float()
        x_B = adapter_tensors[f"{projs['x_proj']}.lora_B.weight"].float()
        rank = gate_A.shape[0]

        A_cat = torch.cat([gate_A, x_A], dim=0)
        B_block = torch.zeros(in_proj_dim, 2 * rank)
        B_block[: gate_B.shape[0], :rank] = gate_B
        B_block[gate_B.shape[0] : gate_B.shape[0] + x_B.shape[0], rank:] = x_B

        Q_B, R_B = torch.linalg.qr(B_block)
        Q_A, R_A = torch.linalg.qr(A_cat.T)
        core = R_B @ R_A.T
        U, S, Vh = torch.linalg.svd(core, full_matrices=False)

        k = rank
        new_B = (Q_B @ U[:, :k]) * S[:k].unsqueeze(0)
        new_A = Vh[:k, :] @ Q_A.T

        kept = S[:k].sum().item()
        total = S.sum().item()
        kept_total += kept
        mass_total += total
        print(f"  {layer_path}: kept {kept / total * 100:.1f}%")
        tensors[f"{in_proj_base}.lora_A.weight"] = new_A.to(torch.bfloat16).contiguous()
        tensors[f"{in_proj_base}.lora_B.weight"] = new_B.to(torch.bfloat16).contiguous()

    # Cast everything to bf16 (PEFT default for this model).
    for k, v in list(tensors.items()):
        if v.dtype != torch.bfloat16:
            tensors[k] = v.to(torch.bfloat16).contiguous()

    print(f"\nConverted {len(adapter_tensors)} trained tensors -> "
          f"{len(tensors)} output tensors.")
    print(f"SVD mass kept: {kept_total / mass_total * 100:.2f}% overall.")

    # 5. Write outputs.
    out_safetensors = out_dir / "adapter_model.safetensors"
    save_file(tensors, str(out_safetensors))
    print(f"\nSaved {out_safetensors} ({out_safetensors.stat().st_size / 1024**3:.2f} GB)")

    # Copy + patch adapter_config.json with explicit Kaggle-style target_modules.
    in_config = json.load(open(in_dir / "adapter_config.json"))
    in_config["target_modules"] = [
        "k_proj", "o_proj", "in_proj", "q_proj",
        "up_proj", "v_proj", "down_proj", "out_proj", "lm_head",
    ]
    in_config["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    in_config["inference_mode"] = True
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(in_config, f, indent=2)
    print(f"Wrote {out_dir / 'adapter_config.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--model_dir", default="/scratch/jwj/NemotronFT/model", type=Path)
    args = parser.parse_args()
    convert(args.in_dir, args.out_dir, args.model_dir)


if __name__ == "__main__":
    main()
