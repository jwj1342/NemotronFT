"""
eval/extract.py — Answer extraction and verification utilities.

Ported verbatim from the official Kaggle metric notebook so that local
evaluation matches the leaderboard grader exactly.

  - BOXED_INSTRUCTION: the prompt suffix every question should carry.
  - extract_final_answer(): pull a \\boxed{...} value out of model output.
  - verify(): compare prediction to ground truth (numeric 1% rel tol, otherwise
    case-insensitive string compare).

Reference: official-doc/nvidia-nemotron-metric.ipynb
"""

from __future__ import annotations

import math
import re


BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_final_answer(text: str | None) -> str:
    r"""Extract the final answer from a model response.

    Prioritizes the last non-empty `\boxed{...}` (with balanced braces so
    answers can themselves contain `}`). Falls back to common "final answer is:"
    patterns, then the last number, then the last non-empty line.
    """
    if text is None:
        return "NOT_FOUND"

    def _extract_boxed(s: str) -> list[str]:
        results: list[str] = []
        tag = r"\boxed{"
        start = 0
        while True:
            idx = s.find(tag, start)
            if idx == -1:
                break
            depth = 1
            i = idx + len(tag)
            while i < len(s) and depth > 0:
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                i += 1
            results.append(s[idx + len(tag) : i - 1])
            start = i
        return results

    matches = _extract_boxed(text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Return True iff `predicted` matches `stored_answer`.

    Numeric values use `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`. Anything
    that doesn't parse as a float falls through to case-insensitive string
    equality.
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()
