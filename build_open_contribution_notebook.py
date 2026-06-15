"""Generate the public Open-Contribution Kaggle notebook from OPEN_CONTRIBUTION.md.

Layout: markdown intro + TL;DR, then §1, then the REPRODUCIBLE RECIPE code cells
(key-rename + bf16 cast of the lossless kienngx adapter -> submission.zip), then
§2-§7 as markdown. The notebook both documents the method and reproduces the LB-0.86
submission from public inputs (no private artifacts).
"""
import json, re, pathlib

SRC = pathlib.Path("OPEN_CONTRIBUTION.md").read_text()

# Split into intro (before "## 1") and the "## N ..." sections, preserving headers.
parts = re.split(r"\n(?=## )", SRC)
intro = parts[0]
sections = {}
for p in parts[1:]:
    m = re.match(r"## (\d+)", p)
    if m:
        sections[int(m.group(1))] = p
    else:
        sections.setdefault("tail", "")
        sections["tail"] += "\n" + p

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}

RECIPE = '''\
# ============================================================================
# REPRODUCIBLE RECIPE — the file-submittable LB 0.86 (§1)
# ----------------------------------------------------------------------------
# Our 0.86 = the LOSSLESSLY-converted public adapter (kienngx's, built via
# `tinker_cookbook.build_lora_adapter` from Huikang's winning Tinker adapter)
# + a one-line LoRA KEY RENAME + a bf16 cast. No SVD, no training.
#
# On Kaggle, add the public model `kienngx/nemotron-nano-30b-trained` as an input
# and point SRC at the `tinker-adapter` dir (it ships adapter_config.json +
# adapter_model.safetensors with keys prefixed `base_model.model.model.`).
# ============================================================================
import os, glob, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = glob.glob("/kaggle/input/**/tinker-adapter/*", recursive=True)[0]  # kienngx adapter
OUT = "/kaggle/working"
print("source adapter:", SRC)

# carry adapter_config.json (+ any tokenizer/template files) verbatim
for f in os.listdir(SRC):
    if f != "adapter_model.safetensors":
        shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))

# THE FIX: (1b) rename `base_model.model.model.` -> `base_model.model.backbone.`
# so PEFT/vLLM actually attach the LoRA to the nemotron_h Mamba modules; without
# this the grader silently runs ~= the base model and you score 0.85.
# Also cast fp32 -> bf16 to land at 1.78 GB (< the ~2 GB file-upload limit).
def rename(k):
    return k.replace("base_model.model.model.", "base_model.model.backbone.")

tensors, renamed = {}, 0
with safe_open(os.path.join(SRC, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        nk = rename(k); renamed += (nk != k)
        t = f.get_tensor(k)
        if t.dtype == torch.float32:
            t = t.to(torch.bfloat16)
        tensors[nk] = t

save_file(tensors, os.path.join(OUT, "adapter_model.safetensors"))
print(f"renamed {renamed}/{len(tensors)} keys -> backbone; cast to bf16; saved to {OUT}")
'''

ZIP = '''\
# Package the adapter exactly as the grader expects (adapter_config.json +
# adapter_model.safetensors at the zip root). This is the file-submittable 0.86.
import subprocess, os
os.chdir("/kaggle/working")
subprocess.run("zip -j submission.zip adapter_config.json adapter_model.safetensors", shell=True, check=True)
print("Done — submission.zip is the LB 0.86 adapter (file-submittable, no notebook-submit needed).")
'''

cells = [
    md(intro),
    md(sections[1]),
    code(RECIPE),
    code(ZIP),
    md(sections[2]),
    md(sections[3]),
    md(sections[4]),
    md(sections[5]),
    md(sections[6]),
    md(sections[7] + sections.get("tail", "")),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
pathlib.Path("open_contribution_notebook.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote open_contribution_notebook.ipynb — {len(cells)} cells "
      f"({sum(c['cell_type']=='markdown' for c in cells)} md, {sum(c['cell_type']=='code' for c in cells)} code)")
