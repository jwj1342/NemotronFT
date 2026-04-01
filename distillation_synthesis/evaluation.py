from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt import SYSTEM_PROMPT

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


INPUT_FIELDS = ("id", "category", "answer", "query")
OUTPUT_FIELDS = [
    "source_index",
    "id",
    "category",
    "query",
    "answer",
    "model_provider",
    "model_name",
    "model_response",
    "extracted_answer",
    "is_correct",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "status",
    "error",
]
SUCCESS_STATUS = "success"
ERROR_STATUS = "error"
TRUE_VALUE = "true"
FALSE_VALUE = "false"


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
            if limit <= 0:
                raise


set_csv_field_size_limit()


@dataclass
class EvalStats:
    total_input: int
    pending: int
    skipped: int
    correct: int
    incorrect: int
    error: int


@dataclass
class SummaryCounts:
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    error: int = 0


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
            "Output CSV schema is incompatible with the current evaluation format. "
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


def parse_bool_flag(value: str) -> bool:
    return str(value).strip().lower() == TRUE_VALUE


def extract_boxed_answer(response_text: str) -> str:
    if not response_text.strip():
        raise ValueError("model response is empty")

    marker = "boxed{"
    start = response_text.rfind(marker)
    if start < 0:
        raise ValueError("model response does not contain a \\boxed{} answer")

    cursor = start + len(marker)
    depth = 1
    extracted_chars: list[str] = []

    while cursor < len(response_text):
        char = response_text[cursor]
        if char == "{":
            depth += 1
            extracted_chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = "".join(extracted_chars).strip()
                if not answer:
                    raise ValueError("boxed answer is empty")
                return answer
            extracted_chars.append(char)
        else:
            extracted_chars.append(char)
        cursor += 1

    raise ValueError("unterminated \\boxed{} answer")


def build_output_row(
    source_index: int,
    input_row: dict[str, str],
    model_provider: str,
    model_name: str,
    response_payload: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    content = str((response_payload or {}).get("content", "") or "")
    row = {
        "source_index": source_index,
        "id": input_row["id"],
        "category": input_row["category"],
        "query": input_row["query"],
        "answer": input_row["answer"],
        "model_provider": model_provider,
        "model_name": model_name,
        "model_response": content,
        "extracted_answer": "",
        "is_correct": FALSE_VALUE,
        "finish_reason": str((response_payload or {}).get("finish_reason", "") or ""),
        "prompt_tokens": (response_payload or {}).get("prompt_tokens", "") or "",
        "completion_tokens": (response_payload or {}).get("completion_tokens", "") or "",
        "total_tokens": (response_payload or {}).get("total_tokens", "") or "",
        "status": SUCCESS_STATUS,
        "error": "",
    }

    if error is not None:
        row["status"] = ERROR_STATUS
        row["error"] = f"{type(error).__name__}: {error}"
        return row

    extracted_answer = extract_boxed_answer(content)
    row["extracted_answer"] = extracted_answer
    row["is_correct"] = (
        TRUE_VALUE if extracted_answer.strip() == input_row["answer"].strip() else FALSE_VALUE
    )
    return row


def build_llm(provider: str, model: str | None) -> Any:
    from llm import SiliconflowLLM, StepLLM

    if provider == "step":
        return StepLLM(model=model) if model else StepLLM()
    if provider == "siliconflow":
        return SiliconflowLLM(model=model) if model else SiliconflowLLM()
    raise ValueError(f"Unsupported provider: {provider}")


def print_run_summary(
    input_path: Path,
    output_path: Path,
    provider: str,
    model_name: str,
    total_rows: int,
    pending_rows: int,
    skipped_rows: int,
    concurrency: int,
    temperature: float,
    resume: bool,
) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
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
    model_provider: str,
    model_name: str,
    queue: asyncio.Queue[dict[str, Any] | None],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> None:
    async with semaphore:
        response_payload: dict[str, Any] | None = None
        try:
            response_payload = await llm.get_response_payload(
                input_row["query"],
                system_prompt=SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            output_row = build_output_row(
                source_index=source_index,
                input_row=input_row,
                model_provider=model_provider,
                model_name=model_name,
                response_payload=response_payload,
            )
        except Exception as exc:  # noqa: BLE001
            output_row = build_output_row(
                source_index=source_index,
                input_row=input_row,
                model_provider=model_provider,
                model_name=model_name,
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
        progress_bar = tqdm(total=total, desc="Evaluating", dynamic_ncols=True)

    correct_count = 0
    incorrect_count = 0
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
            if output_row["status"] == ERROR_STATUS:
                error_count += 1
            elif parse_bool_flag(output_row["is_correct"]):
                correct_count += 1
            else:
                incorrect_count += 1

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    correct=correct_count,
                    incorrect=incorrect_count,
                    error=error_count,
                )
            elif show_progress and total > 0:
                if processed_count % log_interval == 0 or processed_count == total:
                    print(
                        f"Progress: {processed_count}/{total} "
                        f"(correct={correct_count}, incorrect={incorrect_count}, error={error_count})",
                        file=sys.stderr,
                    )

    if progress_bar is not None:
        progress_bar.close()

    return {
        "correct": correct_count,
        "incorrect": incorrect_count,
        "error": error_count,
    }


def summarize_output(output_path: Path) -> tuple[SummaryCounts, dict[str, SummaryCounts]]:
    overall = SummaryCounts()
    per_category: dict[str, SummaryCounts] = defaultdict(SummaryCounts)

    if not output_path.exists() or output_path.stat().st_size == 0:
        return overall, per_category

    rows_by_id: dict[str, dict[str, str]] = {}
    with output_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        validate_output_schema(reader.fieldnames)
        for row in reader:
            row_id = row.get("id")
            if row_id:
                rows_by_id[row_id] = row

    for row in rows_by_id.values():
        category = row["category"]
        overall.total += 1
        per_category[category].total += 1

        if row["status"] == ERROR_STATUS:
            overall.error += 1
            per_category[category].error += 1
        elif parse_bool_flag(row["is_correct"]):
            overall.correct += 1
            per_category[category].correct += 1
        else:
            overall.incorrect += 1
            per_category[category].incorrect += 1

    return overall, dict(sorted(per_category.items()))


def format_accuracy(correct: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(correct / total) * 100:.2f}%"


def print_final_summary(
    input_path: Path,
    output_path: Path,
    stats: EvalStats,
    overall: SummaryCounts,
    per_category: dict[str, SummaryCounts],
) -> None:
    print(f"Finished evaluation for {input_path} -> {output_path}")
    print(
        "Overall: "
        f"total={stats.total_input}, "
        f"pending={stats.pending}, "
        f"skipped={stats.skipped}, "
        f"correct={overall.correct}, "
        f"incorrect={overall.incorrect}, "
        f"error={overall.error}, "
        f"accuracy={format_accuracy(overall.correct, stats.total_input)}"
    )

    if not per_category:
        return

    print("Per-category:")
    for category, counts in per_category.items():
        print(
            f"  {category}: "
            f"total={counts.total}, "
            f"correct={counts.correct}, "
            f"incorrect={counts.incorrect}, "
            f"error={counts.error}, "
            f"accuracy={format_accuracy(counts.correct, counts.total)}"
        )


async def evaluate_dataset(
    input_path: Path,
    output_path: Path,
    llm: Any,
    provider: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    resume: bool,
    show_progress: bool,
) -> EvalStats:
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
        provider=provider,
        model_name=llm.model,
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
                model_provider=provider,
                model_name=llm.model,
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

    await writer_task
    overall, _ = summarize_output(output_path)
    return EvalStats(
        total_input=len(input_rows),
        pending=len(pending_rows),
        skipped=len(input_rows) - len(pending_rows),
        correct=overall.correct,
        incorrect=overall.incorrect,
        error=overall.error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation over a CSV dataset.")
    parser.add_argument("--input", default="data/evaluation_dataset.csv", help="Input CSV path")
    parser.add_argument(
        "--output",
        default="data/evaluation_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--provider",
        choices=("step", "siliconflow"),
        default="step",
        help="Model provider",
    )
    parser.add_argument("--model", default=None, help="Override the provider's default model")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum concurrent requests")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=254000,
        help="Maximum tokens per request",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
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

    llm = build_llm(args.provider, args.model)
    if not llm.api_key:
        env_var = "STEP_API_KEY" if args.provider == "step" else "SILICONFLOW_API_KEY"
        raise RuntimeError(f"{env_var} is not configured")

    stats = await evaluate_dataset(
        input_path=input_path,
        output_path=output_path,
        llm=llm,
        provider=args.provider,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        resume=args.resume,
        show_progress=args.progress,
    )
    overall, per_category = summarize_output(output_path)
    print_final_summary(
        input_path=input_path,
        output_path=output_path,
        stats=stats,
        overall=overall,
        per_category=per_category,
    )


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
