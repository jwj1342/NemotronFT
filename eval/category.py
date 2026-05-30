"""Huikang's 9-category detector — pure function, no heavy deps.

Extracted from eval/eval_full_train.py so it can be imported in venvs that
don't have pandas installed (e.g. the training venv).
"""
from __future__ import annotations

import re


def detect_category(prompt: str) -> str:
    if "secret bit manipulation rule transforms 8-bit binary numbers" in prompt:
        return "bit_manipulation"
    if "secret encryption rules are used on text" in prompt:
        return "cipher"
    if "secret set of transformation rules is applied to equations" in prompt:
        after_header = prompt.split("Below are a few examples:\n", 1)[1]
        examples_text, rest = after_header.split("\nNow, determine the result for: ", 1)
        question_text = rest.strip()
        if any(c.isdigit() for c in examples_text):
            q_match = re.fullmatch(r"(\d+)(\D)(\d+)", question_text)
            if q_match and re.search(
                r"\d" + re.escape(q_match.group(2)) + r"\d", examples_text
            ):
                return "equation_numeric_deduce"
            return "equation_numeric_guess"
        if len(question_text) == 5:
            q_op = question_text[2]
            for ex_line in examples_text.strip().splitlines():
                inp = ex_line.split(" = ")[0].strip()
                if len(inp) == 5 and inp[2] == q_op:
                    return "cryptarithm_deduce"
        return "cryptarithm_guess"
    if "gravitational constant has been secretly changed" in prompt:
        return "gravity"
    if "converted into a different numeral system" in prompt:
        return "numeral"
    if "secret unit conversion is applied to measurements" in prompt:
        return "unit_conversion"
    return "unknown"
