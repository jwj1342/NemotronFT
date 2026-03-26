from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm import StepLLM

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


INPUT_FIELDS = ("id", "prompt", "answer", "label")
OUTPUT_FIELDS = [
    "source_index",
    "id",
    "prompt",
    "answer",
    "label",
    "teacher_response",
    "teacher_reasoning",
    "teacher_content",
    "teacher_boxed_answer",
    "answer_match",
    "status",
    "error",
]
SUCCESS_STATUS = "success"
ERROR_STATUS = "error"


@dataclass
class DistillStats:
    total_input: int
    pending: int
    skipped: int
    success: int
    error: int


def read_input_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        missing_fields = [field for field in INPUT_FIELDS if field not in fieldnames]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        return list(reader)


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    with output_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        required_fields = {"id", "status"}
        if fieldnames and not required_fields.issubset(fieldnames):
            missing = ", ".join(sorted(required_fields - set(fieldnames)))
            raise ValueError(f"Output CSV is missing required columns for resume: {missing}")

        completed_ids = set()
        for row in reader:
            if row.get("status") in {SUCCESS_STATUS, ERROR_STATUS} and row.get("id"):
                completed_ids.add(row["id"])
        return completed_ids


def extract_braced_content(text: str, open_brace_index: int) -> tuple[str, int] | None:
    if open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None

    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index + 1:index], index
    return None


def extract_last_boxed_answer(text: str) -> str | None:
    token = r"\boxed"
    search_from = 0
    last_boxed_answer = None

    while True:
        token_index = text.find(token, search_from)
        if token_index == -1:
            break

        brace_index = token_index + len(token)
        while brace_index < len(text) and text[brace_index].isspace():
            brace_index += 1

        if brace_index < len(text) and text[brace_index] == "{":
            extracted = extract_braced_content(text, brace_index)
            if extracted is not None:
                last_boxed_answer = extracted[0].strip()
                search_from = extracted[1] + 1
                continue

        search_from = token_index + len(token)

    return last_boxed_answer


def unwrap_command(text: str, command: str) -> tuple[str, bool]:
    prefix = f"\\{command}"
    if not text.startswith(prefix):
        return text, False

    index = len(prefix)
    while index < len(text) and text[index].isspace():
        index += 1

    if index >= len(text) or text[index] != "{":
        return text, False

    extracted = extract_braced_content(text, index)
    if extracted is None or extracted[1] != len(text) - 1:
        return text, False

    return extracted[0].strip(), True


def unwrap_outer_braces(text: str) -> tuple[str, bool]:
    if not text.startswith("{") or not text.endswith("}"):
        return text, False

    extracted = extract_braced_content(text, 0)
    if extracted is None or extracted[1] != len(text) - 1:
        return text, False

    return extracted[0].strip(), True


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    changed = True
    while changed and normalized:
        changed = False

        wrappers = [
            ("$$", "$$"),
            ("$", "$"),
            (r"\(", r"\)"),
            (r"\[", r"\]"),
        ]
        for prefix, suffix in wrappers:
            if normalized.startswith(prefix) and normalized.endswith(suffix):
                inner = normalized[len(prefix):len(normalized) - len(suffix)].strip()
                if inner != normalized:
                    normalized = inner
                    changed = True

        for command in ("boxed", "text", "mathrm", "textrm", "mathbf", "operatorname"):
            unwrapped, did_unwrap = unwrap_command(normalized, command)
            if did_unwrap:
                normalized = unwrapped
                changed = True

        unwrapped_braces, did_unwrap_braces = unwrap_outer_braces(normalized)
        if did_unwrap_braces:
            normalized = unwrapped_braces
            changed = True

        normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def compute_answer_match(teacher_boxed_answer: str | None, gold_answer: str) -> str:
    if not teacher_boxed_answer:
        return ""

    return str(normalize_answer(teacher_boxed_answer) == normalize_answer(gold_answer)).lower()


def parse_teacher_response(text: str) -> dict[str, str]:
    response_text = text or ""
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)\Z", response_text, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        content = match.group(2).strip()
    else:
        reasoning = ""
        content = response_text.strip()

    return {
        "teacher_response": response_text,
        "teacher_reasoning": reasoning,
        "teacher_content": content,
        "teacher_boxed_answer": extract_last_boxed_answer(content or response_text) or "",
    }


def build_output_row(
    source_index: int,
    input_row: dict[str, str],
    response_text: str | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    row = {
        "source_index": source_index,
        "id": input_row["id"],
        "prompt": input_row["prompt"],
        "answer": input_row["answer"],
        "label": input_row["label"],
        "teacher_response": "",
        "teacher_reasoning": "",
        "teacher_content": "",
        "teacher_boxed_answer": "",
        "answer_match": "",
        "status": SUCCESS_STATUS,
        "error": "",
    }

    if error is not None:
        row["status"] = ERROR_STATUS
        row["error"] = f"{type(error).__name__}: {error}"
        return row

    parsed = parse_teacher_response(response_text or "")
    row.update(parsed)
    row["answer_match"] = compute_answer_match(
        row["teacher_boxed_answer"],
        input_row["answer"],
    )
    return row


def print_run_summary(
    input_path: Path,
    output_path: Path,
    total_rows: int,
    pending_rows: int,
    skipped_rows: int,
    concurrency: int,
    resume: bool,
) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Pending rows: {pending_rows}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Concurrency: {concurrency}")
    print(f"Resume: {resume}")


async def worker(
    source_index: int,
    input_row: dict[str, str],
    llm: Any,
    queue: asyncio.Queue[dict[str, Any] | None],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> None:
    async with semaphore:
        try:
            response_text = await llm.get_response(
                input_row["prompt"],
                max_tokens=max_tokens,
            )
            output_row = build_output_row(
                source_index=source_index,
                input_row=input_row,
                response_text=response_text,
            )
        except Exception as exc:  # noqa: BLE001
            output_row = build_output_row(
                source_index=source_index,
                input_row=input_row,
                error=exc,
            )

    await queue.put(output_row)


async def writer(
    queue: asyncio.Queue[dict[str, Any] | None],
    output_path: Path,
    total: int,
    resume: bool,
    show_progress: bool,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    append_mode = resume and output_path.exists() and output_path.stat().st_size > 0
    file_mode = "a" if append_mode else "w"
    should_write_header = not append_mode

    progress_bar = None
    use_tqdm = show_progress and tqdm is not None and sys.stderr.isatty()
    if use_tqdm and total > 0:
        progress_bar = tqdm(total=total, desc="Distilling", dynamic_ncols=True)

    success_count = 0
    error_count = 0
    processed_count = 0
    log_interval = max(1, min(50, total // 10 or 1))

    with output_path.open(file_mode, encoding="utf-8", newline="") as dst:
        csv_writer = csv.DictWriter(dst, fieldnames=OUTPUT_FIELDS)
        if should_write_header:
            csv_writer.writeheader()
            dst.flush()

        while True:
            output_row = await queue.get()
            if output_row is None:
                break

            csv_writer.writerow(output_row)
            dst.flush()

            processed_count += 1
            if output_row["status"] == SUCCESS_STATUS:
                success_count += 1
            else:
                error_count += 1

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(success=success_count, error=error_count)
            elif show_progress and total > 0:
                if processed_count % log_interval == 0 or processed_count == total:
                    print(
                        f"Progress: {processed_count}/{total} "
                        f"(success={success_count}, error={error_count})",
                        file=sys.stderr,
                    )

    if progress_bar is not None:
        progress_bar.close()

    return {"success": success_count, "error": error_count}


async def distill_dataset(
    input_path: Path,
    output_path: Path,
    llm: Any,
    concurrency: int,
    max_tokens: int,
    resume: bool,
    show_progress: bool,
) -> DistillStats:
    input_rows = read_input_rows(input_path)
    completed_ids = load_completed_ids(output_path) if resume else set()

    pending_rows: list[tuple[int, dict[str, str]]] = []
    for source_index, row in enumerate(input_rows, start=1):
        if row["id"] in completed_ids:
            continue
        pending_rows.append((source_index, row))

    print_run_summary(
        input_path=input_path,
        output_path=output_path,
        total_rows=len(input_rows),
        pending_rows=len(pending_rows),
        skipped_rows=len(input_rows) - len(pending_rows),
        concurrency=concurrency,
        resume=resume,
    )

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    semaphore = asyncio.Semaphore(concurrency)
    writer_task = asyncio.create_task(
        writer(
            queue=queue,
            output_path=output_path,
            total=len(pending_rows),
            resume=resume,
            show_progress=show_progress,
        )
    )

    worker_tasks = [
        asyncio.create_task(
            worker(
                source_index=source_index,
                input_row=row,
                llm=llm,
                queue=queue,
                semaphore=semaphore,
                max_tokens=max_tokens,
            )
        )
        for source_index, row in pending_rows
    ]

    try:
        if worker_tasks:
            await asyncio.gather(*worker_tasks)
    finally:
        await queue.put(None)

    writer_stats = await writer_task
    return DistillStats(
        total_input=len(input_rows),
        pending=len(pending_rows),
        skipped=len(input_rows) - len(pending_rows),
        success=writer_stats["success"],
        error=writer_stats["error"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher distillation over a CSV dataset.")
    parser.add_argument("--input", default="data/sample1200-train.csv", help="Input CSV path")
    parser.add_argument(
        "--output",
        default="data/sample1200-train-distilled.csv",
        help="Output CSV path",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum tokens per request")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing output file",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a progress bar or progress logs",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output CSV paths must be different")

    llm = StepLLM()
    if not llm.api_key:
        raise RuntimeError("STEP_API_KEY is not configured")

    stats = await distill_dataset(
        input_path=input_path,
        output_path=output_path,
        llm=llm,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        resume=args.resume,
        show_progress=args.progress,
    )

    print(
        "Finished distillation: "
        f"success={stats.success}, "
        f"error={stats.error}, "
        f"skipped={stats.skipped}, "
        f"total_input={stats.total_input}"
    )


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
