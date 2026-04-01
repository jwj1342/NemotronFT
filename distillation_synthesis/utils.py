from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from prompt import DISTILLATION_COT_USER_PROMPT


def prompt_format(
    replacements: Sequence[str] | Mapping[str, str],
    template: str = DISTILLATION_COT_USER_PROMPT,
) -> str:
    """Replace placeholders like [question] in a prompt template."""
    if not template or not template.strip():
        raise ValueError("template must be a non-empty string")

    placeholders = []
    seen = set()
    for match in re.finditer(r"\[[^\[\]]+\]", template):
        placeholder = match.group(0)
        if placeholder not in seen:
            placeholders.append(placeholder)
            seen.add(placeholder)

    if isinstance(replacements, Mapping):
        normalized_replacements: dict[str, str] = {}
        for key, value in replacements.items():
            normalized_key = key if key.startswith("[") and key.endswith("]") else f"[{key}]"
            normalized_replacements[normalized_key] = value
    elif isinstance(replacements, Sequence) and not isinstance(replacements, (str, bytes, bytearray)):
        if len(replacements) != len(placeholders):
            raise ValueError(
                "replacement count does not match placeholder count: "
                f"expected {len(placeholders)}, got {len(replacements)}"
            )
        normalized_replacements = dict(zip(placeholders, replacements, strict=True))
    else:
        raise TypeError("replacements must be a mapping or a non-string sequence")

    formatted_prompt = template
    for placeholder in placeholders:
        if placeholder not in normalized_replacements:
            raise ValueError(f"missing replacement for placeholder: {placeholder}")
        value = normalized_replacements[placeholder]
        formatted_prompt = formatted_prompt.replace(placeholder, value)

    return formatted_prompt.strip()
