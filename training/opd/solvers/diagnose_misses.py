"""Diagnose which Huikang-solved cryptarithm puzzles our extended solver misses.

For each miss, dump:
  - The 5 examples + query
  - Huikang's investigation (the actual rule he found)
  - Our hypothesis (if we got any partial answer)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))

from extended_solver import solve_problem  # type: ignore


CONFIGS = [
    (10, False), (10, True),
    (9, False), (9, True),
    (11, False), (8, False), (12, False),
]


def load_problems():
    out = []
    with open(PROJECT_ROOT / "resource" / "winning-solution" / "problems.jsonl") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def load_data(pid: str):
    fp = PROJECT_ROOT / "resource" / "winning-solution" / "problems" / f"{pid}.jsonl"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.loads(f.readline())


def load_huikang_inv(pid: str) -> str:
    fp = PROJECT_ROOT / "resource" / "winning-solution" / "investigations" / f"{pid}.txt"
    if not fp.exists():
        return ""
    return fp.read_text()


def try_solve(pid_data):
    pid, data = pid_data
    gt = str(data["answer"])
    for base, post_rev in CONFIGS:
        try:
            pred, info = solve_problem(data, base=base, post_reverse=post_rev)
        except Exception:
            continue
        if pred is not None and pred == gt:
            return pid, True, (base, post_rev, info)
        if pred is not None:
            # wrong prediction — keep first attempted answer for diagnostic
            return pid, False, ("wrong_pred", pred, gt, base, post_rev)
    return pid, False, None


def main():
    problems = load_problems()
    solved = [p for p in problems
              if p.get("category") in ("cryptarithm_deduce", "cryptarithm_guess")
              and p.get("status") == "rule_found"]
    tasks = []
    for p in solved:
        d = load_data(p["id"])
        if d is None or "answer" not in d:
            continue
        tasks.append((p["id"], d))

    print(f"Checking {len(tasks)} Huikang-solved cryptarithm puzzles", flush=True)
    misses = []
    with mp.Pool(8) as pool:
        for pid, ok, extra in pool.imap_unordered(try_solve, tasks, chunksize=2):
            if not ok:
                misses.append((pid, extra))

    print(f"\n=== MISSED {len(misses)}/{len(tasks)} ===\n")
    for pid, extra in misses:
        data = load_data(pid)
        inv = load_huikang_inv(pid)
        print(f"---------------- {pid} ----------------")
        print(f"category: {data.get('category', '?')}")
        # Examples
        print("examples:")
        for e in data["examples"]:
            print(f"  {e['input_value']} = {e['output_value']}")
        print(f"query: {data['question']}  → answer: {data['answer']}")
        # Huikang's investigation
        print("\nHuikang's investigation:")
        for line in inv.splitlines():
            if line.strip():
                print(f"  {line}")
        # Our outcome
        print(f"\nOur outcome: {extra}")
        print()


if __name__ == "__main__":
    main()
