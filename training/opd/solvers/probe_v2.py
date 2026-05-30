"""Probe v2: use extended_solver (joint per-op-symbol search) to find new
coverage. First validates on Huikang's solved puzzles, then probes unsolved.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))

from extended_solver import solve_problem, OP_NAMES  # type: ignore


def load_problems():
    out = []
    with open(PROJECT_ROOT / "resource" / "winning-solution" / "problems.jsonl") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def load_problem(pid: str):
    fp = PROJECT_ROOT / "resource" / "winning-solution" / "problems" / f"{pid}.jsonl"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.loads(f.readline())


# Outer configs to try, in order of likelihood
CONFIGS = [
    (10, False),  # most likely: standard base-10
    (10, True),   # reverse result
    (9, False),
    (9, True),
    (11, False),
    (8, False),
    (12, False),
]


def solve_one(task: dict) -> dict:
    pid = task["id"]
    data = task["data"]
    gt = str(data["answer"])
    for base, post_rev in CONFIGS:
        try:
            pred, info = solve_problem(data, base=base, post_reverse=post_rev)
        except Exception as e:
            continue
        if pred is not None and pred == gt:
            return {
                "id": pid,
                "solved": True,
                "config": (base, post_rev),
                "mapping": info[0],
                "op_assign": info[1],
                "predicted": pred,
            }
    return {"id": pid, "solved": False}


def run_set(problems_subset: list[dict], label: str, n_workers: int, incremental_path: Path | None = None) -> tuple[int, list]:
    """Run probe over a subset, return (n_solved, list_of_solved_results).
    If incremental_path is given, append each solved result immediately."""
    tasks = []
    for p in problems_subset:
        data = load_problem(p["id"])
        if data is None or "examples" not in data or "answer" not in data:
            continue
        tasks.append({"id": p["id"], "data": data})

    print(f"\n[{label}] launching {len(tasks)} tasks on {n_workers} workers", flush=True)
    t0 = time.time()
    solved_results = []
    completed = 0
    inc_file = open(incremental_path, "w") if incremental_path else None
    try:
        with mp.Pool(n_workers) as pool:
            for res in pool.imap_unordered(solve_one, tasks, chunksize=2):
                completed += 1
                if res["solved"]:
                    solved_results.append(res)
                    if inc_file:
                        inc_file.write(json.dumps(res) + "\n")
                        inc_file.flush()
                if completed % 50 == 0 or completed == len(tasks):
                    dt = time.time() - t0
                    print(f"[{label}] {completed}/{len(tasks)} solved={len(solved_results)} "
                          f"elapsed={dt:.1f}s", flush=True)
    finally:
        if inc_file:
            inc_file.close()
    return len(solved_results), solved_results


def main() -> None:
    problems = load_problems()
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", str(min(8, mp.cpu_count()))))
    print(f"Workers: {n_workers}", flush=True)

    crypto = [p for p in problems if p.get("category") in ("cryptarithm_deduce", "cryptarithm_guess")]
    solved_huikang = [p for p in crypto if p.get("status") == "rule_found"]
    unsolved = [p for p in crypto if p.get("status") in ("rule_unknown", "hypothesis_formed")]

    print(f"\nCryptarithm total: {len(crypto)}")
    print(f"  Huikang rule_found:        {len(solved_huikang)}")
    print(f"  unsolved (target):         {len(unsolved)}")

    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Stage 1: sanity-check our solver on Huikang's solved set
    print("\n=== STAGE 1: sanity-check on Huikang's solved cryptarithm ===")
    n_recovered, _ = run_set(solved_huikang, "sanity", n_workers,
                              incremental_path=runs_dir / "probe_v2_sanity.jsonl")
    print(f"\nRecovered {n_recovered}/{len(solved_huikang)} = {n_recovered/max(1,len(solved_huikang))*100:.1f}%")
    if n_recovered / max(1, len(solved_huikang)) < 0.5:
        print("WARNING: extended solver fails to recover most of Huikang's known solutions.")
        print("         There is likely a bug in extended_solver.py — investigate before trusting Stage 2.")

    # Stage 2: probe the unsolved set — write incrementally to survive timeout
    print("\n=== STAGE 2: probe unsolved cryptarithm ===")
    n_new, new_results = run_set(unsolved, "probe", n_workers,
                                  incremental_path=runs_dir / "probe_v2_results.jsonl")
    print(f"\nNewly solved: {n_new}/{len(unsolved)} = {n_new/max(1,len(unsolved))*100:.1f}%")

    # Breakdown by config
    print("\n=== Config breakdown (newly solved) ===")
    cfg_count: Counter = Counter()
    for r in new_results:
        cfg_count[r["config"]] += 1
    for cfg, n in cfg_count.most_common():
        print(f"  base={cfg[0]} post_rev={cfg[1]}: {n}")

    # Breakdown by op_assign (which new ops surfaced)
    print("\n=== Op usage in newly-solved puzzles ===")
    op_count: Counter = Counter()
    for r in new_results:
        for op_sym, op_name in r["op_assign"].items():
            op_count[op_name] += 1
    for op, n in op_count.most_common():
        print(f"  {op:18s} {n}")

    print(f"\nProbe results saved incrementally to {runs_dir / 'probe_v2_results.jsonl'}")


if __name__ == "__main__":
    main()
