from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm import StepLLM
from prompt import DISTILLATION_COT_SYSTEM_PROMPT, DISTILLATION_COT_USER_PROMPT
from utils import prompt_format

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
    "teacher_finish_reason",
    "teacher_prompt_tokens",
    "teacher_completion_tokens",
    "teacher_total_tokens",
    "teacher_cot",
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


def validate_output_schema(fieldnames: list[str] | None) -> None:
    if fieldnames != OUTPUT_FIELDS:
        expected = ", ".join(OUTPUT_FIELDS)
        got = ", ".join(fieldnames or [])
        raise ValueError(
            "Output CSV schema is incompatible with the current COT distillation format. "
            f"Expected columns: {expected}. "
            f"Got: {got or '<empty>'}. "
            "Use a new --output path or remove the old output file before resuming."
        )


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    with output_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        validate_output_schema(reader.fieldnames)

        completed_ids = set()
        for row in reader:
            if row.get("status") in {SUCCESS_STATUS, ERROR_STATUS} and row.get("id"):
                completed_ids.add(row["id"])
        return completed_ids


def parse_distillation_response(response_text: str) -> str:
    if not response_text.strip():
        raise ValueError("model response is empty")

    payload = json.loads(response_text)
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")

    cot = payload.get("cot")
    if not isinstance(cot, str):
        raise ValueError("model response JSON must contain a string field 'cot'")
    if not cot.strip():
        raise ValueError("model response field 'cot' must be a non-empty string")

    return cot


def build_output_row(
    source_index: int,
    input_row: dict[str, str],
    response_payload: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    row = {
        "source_index": source_index,
        "id": input_row["id"],
        "prompt": input_row["prompt"],
        "answer": input_row["answer"],
        "label": input_row["label"],
        "teacher_response": "",
        "teacher_finish_reason": "",
        "teacher_prompt_tokens": "",
        "teacher_completion_tokens": "",
        "teacher_total_tokens": "",
        "teacher_cot": "",
        "status": SUCCESS_STATUS,
        "error": "",
    }

    if response_payload is not None:
        row["teacher_response"] = str(response_payload.get("raw_response", "") or "")
        row["teacher_finish_reason"] = str(response_payload.get("finish_reason", "") or "")
        row["teacher_prompt_tokens"] = response_payload.get("prompt_tokens", "") or ""
        row["teacher_completion_tokens"] = response_payload.get("completion_tokens", "") or ""
        row["teacher_total_tokens"] = response_payload.get("total_tokens", "") or ""

    if error is not None:
        row["status"] = ERROR_STATUS
        row["error"] = f"{type(error).__name__}: {error}"
        return row

    content = str((response_payload or {}).get("content", "") or "")
    row["teacher_cot"] = parse_distillation_response(content)
    return row


def print_run_summary(
    input_path: Path,
    output_path: Path,
    total_rows: int,
    pending_rows: int,
    skipped_rows: int,
    concurrency: int,
    temperature: float,
    resume: bool,
) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Pending rows: {pending_rows}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Concurrency: {concurrency}")
    print(f"Temperature: {temperature}")
    print(f"Resume: {resume}")


async def worker(
    source_index: int,
    input_row: dict[str, str],
    llm: Any,
    queue: asyncio.Queue[dict[str, Any] | None],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> None:
    async with semaphore:
        response_payload: dict[str, Any] | None = None
        try:
            formatted_prompt = prompt_format(
                [input_row["prompt"], input_row["answer"]], 
                DISTILLATION_COT_USER_PROMPT
            )
            response_payload = await llm.get_response_payload(
                formatted_prompt,
                system_prompt=DISTILLATION_COT_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            output_row = build_output_row(
                source_index=source_index,
                input_row=input_row,
                response_payload=response_payload,
            )
        except Exception as exc:  # noqa: BLE001
            output_row = build_output_row(
                source_index=source_index,
                input_row=input_row,
                response_payload=response_payload,
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
    temperature: float,
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
        temperature=temperature,
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
                temperature=temperature,
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
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
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
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
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
        temperature=args.temperature,
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
