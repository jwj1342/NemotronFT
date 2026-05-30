"""Shared helpers for the RL pipeline: JSONL io + prompt template."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "model"
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from extract import BOXED_INSTRUCTION  # type: ignore  # noqa: E402


def write_jsonl(path: Path | str, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[dict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path | str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_user_content(problem: str) -> str:
    return problem + BOXED_INSTRUCTION


def apply_chat_template_for_user(tokenizer, problem: str) -> str:
    """Build the exact pre-generation text used at both rollout and train time."""
    user_content = build_user_content(problem)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
