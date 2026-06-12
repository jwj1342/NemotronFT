"""Create a PEFT-loadable warmstart copy of huikang-baseline-converted (lossy 0.85).

The conversion misfiled the lm_head LoRA under `base_model.model.backbone.lm_head.*`
but the HF model exposes the head at `base_model.model.lm_head.*` (confirmed against a
fresh train.py LoRA: 6004/6005 modules match; only lm_head prefix differs). Rename so
PeftModel.from_pretrained maps every module → warmstart loads fully (incl. lm_head).
"""
from __future__ import annotations
import shutil, json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

PROJECT = Path("/scratch/jwj/NemotronFT")
SRC = PROJECT / "output/huikang-baseline-converted"
DST = PROJECT / "output/huikang-warmstart"

DST.mkdir(parents=True, exist_ok=True)
# copy config + any aux files (not the safetensors, we rewrite it)
for f in SRC.iterdir():
    if f.suffix == ".safetensors":
        continue
    shutil.copy2(f, DST / f.name)

tensors = {}
renamed = 0
with safe_open(str(SRC / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        nk = k.replace("base_model.model.backbone.lm_head.", "base_model.model.lm_head.")
        if nk != k:
            renamed += 1
        tensors[nk] = f.get_tensor(k)

save_file(tensors, str(DST / "adapter_model.safetensors"))
print(f"[done] {len(tensors)} tensors, renamed {renamed} lm_head keys -> {DST}")

# sanity: confirm no remaining backbone.lm_head, and lm_head now present
keys = list(tensors.keys())
bad = [k for k in keys if "backbone.lm_head" in k]
good = [k for k in keys if k.startswith("base_model.model.lm_head.")]
print(f"   remaining backbone.lm_head keys: {len(bad)}  | lm_head keys now: {len(good)}")
