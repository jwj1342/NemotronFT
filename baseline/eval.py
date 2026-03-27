"""
baseline/eval.py — Evaluation utilities for Nemotron LoRA fine-tuning.

Provides:
  - extract_final_answer(): Extract answer from model output (official metric logic)
  - verify(): Check if prediction matches ground truth (1% numeric tolerance)
  - evaluate_samples(): Run inference on a list of (prompt, answer, category) tuples
    and return accuracy metrics.

The evaluation logic is ported directly from the official competition metric:
  official-doc/nvidia-nemotron-metric.ipynb
"""

import math
import re
import sys
import time
from collections import defaultdict

import torch

# ── Official metric functions (ported verbatim) ─────────────────────────────

BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_final_answer(text: str | None) -> str:
    r"""Extract the final answer from model response.

    Prioritizes \boxed{} format, then falls back to other patterns.
    Ported from official nvidia-nemotron-metric.ipynb.
    """
    if text is None:
        return "NOT_FOUND"

    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Check if prediction matches ground truth.

    Numeric: math.isclose(rel_tol=1e-2, abs_tol=1e-5).
    Non-numeric: case-insensitive string comparison.
    Ported from official nvidia-nemotron-metric.ipynb.
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# ── Local evaluation (uses model.generate, not vLLM) ────────────────────────

@torch.no_grad()
def evaluate_samples(
    model,
    tokenizer,
    eval_data: list[tuple[str, str, str]],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> dict:
    """Run batched greedy-decode inference on eval_data and compute accuracy.

    Args:
        model: The (PEFT-wrapped) causal LM.
        tokenizer: The tokenizer.
        eval_data: List of (prompt, ground_truth_answer, category) tuples.
        max_new_tokens: Max tokens to generate per sample.
        batch_size: Number of samples to generate in parallel.

    Returns:
        Dict with keys: accuracy, num_correct, num_total, per_category, details.
    """
    was_training = model.training
    model.eval()

    # Ensure left-padding for batched generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []
    num_correct = 0
    num_total = len(eval_data)
    num_batches = (num_total + batch_size - 1) // batch_size
    processed = 0

    # Pre-build all chat strings
    chat_strings = []
    for prompt, answer, category in eval_data:
        user_content = prompt + BOXED_INSTRUCTION
        chat_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        chat_strings.append(chat_input)

    t0 = time.time()
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_total)
        batch_chats = chat_strings[batch_start:batch_end]
        batch_meta = eval_data[batch_start:batch_end]
        cur_bs = len(batch_chats)

        batch_t0 = time.time()

        # Tokenize with left-padding for uniform length
        inputs = tokenizer(
            batch_chats,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        prompt_lengths = (inputs["attention_mask"]).sum(dim=1)  # per-sample real token count

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode each sample in the batch
        for j in range(cur_bs):
            prompt_len = inputs["input_ids"].shape[1]  # padded length
            generated_ids = outputs[j][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            _, answer, category = batch_meta[j]
            predicted = extract_final_answer(generated_text)
            correct = verify(str(answer), predicted)

            if correct:
                num_correct += 1
                per_category[category]["correct"] += 1
            per_category[category]["total"] += 1

            details.append({
                "category": category,
                "answer": answer,
                "predicted": predicted,
                "correct": correct,
            })

        processed += cur_bs
        elapsed = time.time() - t0
        batch_elapsed = time.time() - batch_t0
        avg_per_sample = elapsed / processed
        eta = avg_per_sample * (num_total - processed)
        sys.stdout.write(
            f"\r[eval] {processed}/{num_total} "
            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left, "
            f"{batch_elapsed:.1f}s/batch({cur_bs}), {avg_per_sample:.1f}s/sample] "
            f"acc={num_correct}/{processed}   "
        )
        sys.stdout.flush()

    print()  # newline after progress bar

    # Restore original padding side
    tokenizer.padding_side = orig_padding_side

    cat_results = {}
    for cat in sorted(per_category):
        c = per_category[cat]
        cat_results[cat] = {
            "accuracy": c["correct"] / c["total"] if c["total"] > 0 else 0.0,
            "correct": c["correct"],
            "total": c["total"],
        }

    if was_training:
        model.train()

    return {
        "accuracy": num_correct / num_total if num_total > 0 else 0.0,
        "num_correct": num_correct,
        "num_total": num_total,
        "per_category": cat_results,
        "details": details,
    }
