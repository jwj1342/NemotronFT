"""Quick test: vLLM inference with NemotronH + LoRA adapter."""

import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "model")
CHECKPOINT_PATH = str(PROJECT_ROOT / "output" / "cot-lora" / "checkpoint-150")

# ── Check adapter exists ────────────────────────────────────────────────────
adapter_cfg = Path(CHECKPOINT_PATH) / "adapter_config.json"
if not adapter_cfg.exists():
    print(f"ERROR: adapter not found at {CHECKPOINT_PATH}")
    sys.exit(1)
print(f"Model:   {MODEL_PATH}")
print(f"Adapter: {CHECKPOINT_PATH}")

# ── Load vLLM ───────────────────────────────────────────────────────────────
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

print("\n[1/3] Loading model with vLLM ...")
t0 = time.time()
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    dtype="auto",
    max_model_len=8192,
    trust_remote_code=True,
    enable_lora=True,
    max_lora_rank=32,
    enable_prefix_caching=True,
)
load_time = time.time() - t0
print(f"[1/3] Model loaded in {load_time:.1f}s")

# ── Prepare test prompts ────────────────────────────────────────────────────
test_prompts = [
    # numeral (should be easy)
    "Convert the number 255 from base 10 to base 16.\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`",
    # unit_conv
    "Convert 1000 meters to kilometers.\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`",
    # equation_numeric
    "Solve for x: 2x + 5 = 15\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`",
]

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=256,
)

lora_request = LoRARequest(
    lora_name="cot-mixed-150",
    lora_int_id=1,
    lora_path=CHECKPOINT_PATH,
)

# ── Inference WITHOUT LoRA (base model) ─────────────────────────────────────
print("\n[2/3] Inference WITHOUT LoRA (base model) ...")
t0 = time.time()
outputs_base = llm.generate(test_prompts, sampling_params)
base_time = time.time() - t0

for i, output in enumerate(outputs_base):
    text = output.outputs[0].text
    n_tokens = len(output.outputs[0].token_ids)
    print(f"  Prompt {i+1}: {n_tokens} tokens, {n_tokens/base_time:.1f} tok/s")
    print(f"  Output: {text[:200]}...")

# ── Inference WITH LoRA ─────────────────────────────────────────────────────
print(f"\n[3/3] Inference WITH LoRA ({CHECKPOINT_PATH}) ...")
t0 = time.time()
outputs_lora = llm.generate(test_prompts, sampling_params, lora_request=lora_request)
lora_time = time.time() - t0

total_tokens = 0
for i, output in enumerate(outputs_lora):
    text = output.outputs[0].text
    n_tokens = len(output.outputs[0].token_ids)
    total_tokens += n_tokens
    print(f"  Prompt {i+1}: {n_tokens} tokens")
    print(f"  Output: {text[:200]}...")

print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Model load:   {load_time:.1f}s")
print(f"  Base (3 prompts): {base_time:.1f}s")
print(f"  LoRA (3 prompts): {lora_time:.1f}s, {total_tokens/lora_time:.1f} tok/s")
print(f"{'='*60}")
