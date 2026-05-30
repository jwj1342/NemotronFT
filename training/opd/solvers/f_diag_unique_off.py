"""F-100 retest with unique=False (Codex priority test).

Same 100-puzzle sample (seeded), same WIDE configs, but bijective mapping
constraint dropped.  If solve rate jumps, then bijection assumption was the
culprit — not the operator set.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOLVER_DIR = PROJECT_ROOT / "training" / "opd" / "solvers"
sys.path.insert(0, str(SOLVER_DIR))

from measurements import measure, parse_prompt  # type: ignore
import extended_solver as ES  # type: ignore
from f_diagnostic import (  # type: ignore
    _is_f_bucket, WIDE_CONFIGS,
)


PROBLEMS_DIR = PROJECT_ROOT / "resource" / "winning-solution" / "problems"
PROBLEMS_FILE = PROJECT_ROOT / "resource" / "winning-solution" / "problems.jsonl"
PROBE_RESULTS = SOLVER_DIR / "runs" / "probe_v2_results.jsonl"
OUT = SOLVER_DIR / "runs" / "f_diag_unique_off.jsonl"
SUMMARY = SOLVER_DIR / "runs" / "f_diag_unique_off_summary.json"


def _solve_one_config(data, base, post_reverse, branch_cap, unique):
    gt = str(data["answer"])
    examples = []
    for ex in data["examples"]:
        inp = ex["input_value"]
        out = ex["output_value"]
        if len(inp) != 5:
            continue
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))
    query = tuple(data["question"])
    if len(query) != 5 or not examples:
        return {"solved": False, "predicted": None, "n_cands": 0, "hit_cap": False}

    solver = ES.ExtendedSolver(examples, query, base=base, post_reverse=post_reverse, unique=unique)
    solver.branch_cap = branch_cap
    hit_cap = False
    try:
        solver._process(0)
    except ES._BranchCap:
        hit_cap = True
    pred = None
    if solver.answers:
        pred, _ = solver.answers.most_common(1)[0]
    return {
        "solved": pred == gt,
        "predicted": pred,
        "n_cands": len(solver.answers),
        "hit_cap": hit_cap,
    }


def _solve_with(data, configs, branch_cap, unique):
    gt = str(data["answer"])
    aggregate = {"solved": False, "predicted": None, "n_cands": 0, "hit_cap": False}
    for base, post in configs:
        r = _solve_one_config(data, base, post, branch_cap, unique)
        if r["hit_cap"]:
            aggregate["hit_cap"] = True
        aggregate["n_cands"] += r["n_cands"]
        if r["solved"]:
            aggregate["solved"] = True
            aggregate["predicted"] = r["predicted"]
            return aggregate
        if r["predicted"] is not None and aggregate["predicted"] is None:
            aggregate["predicted"] = r["predicted"]
    return aggregate


def _worker(task):
    pid = task["id"]
    data = task["data"]
    r_strict = _solve_with(data, WIDE_CONFIGS, branch_cap=10_000_000, unique=True)
    r_loose = _solve_with(data, WIDE_CONFIGS, branch_cap=10_000_000, unique=False)
    return {"id": pid, "gt": str(data["answer"]),
            "strict": r_strict, "loose": r_loose}


def main():
    random.seed(42)
    solved = set()
    with open(PROBE_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if r["solved"]:
                solved.add(r["id"])

    candidates = []
    with open(PROBLEMS_FILE) as f:
        for line in f:
            p = json.loads(line)
            if (
                p.get("category", "").startswith("cryptarithm")
                and p.get("status") != "rule_found"
                and p["id"] not in solved
            ):
                try:
                    with open(PROBLEMS_DIR / f"{p['id']}.jsonl") as pf:
                        d = json.loads(pf.readline())
                except Exception:
                    continue
                parsed = parse_prompt(d.get("prompt", ""))
                if not parsed or len(parsed.query) != 5:
                    continue
                if _is_f_bucket(parsed):
                    candidates.append({"id": p["id"], "data": d})
    sample = random.sample(candidates, k=min(100, len(candidates)))
    print(f"F-bucket sample (seed 42): {len(sample)}")

    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    t0 = time.time()
    rows = []
    with mp.Pool(nw) as pool, open(OUT, "w") as f_out:
        for i, r in enumerate(pool.imap_unordered(_worker, sample, chunksize=1), 1):
            rows.append(r)
            f_out.write(json.dumps(r) + "\n")
            f_out.flush()
            if i % 10 == 0 or i == len(sample):
                print(f"  {i}/{len(sample)} elapsed={time.time()-t0:.1f}s", flush=True)

    strict_solved = sum(1 for r in rows if r["strict"]["solved"])
    loose_solved = sum(1 for r in rows if r["loose"]["solved"])
    only_loose = sum(1 for r in rows if (not r["strict"]["solved"]) and r["loose"]["solved"])
    only_strict = sum(1 for r in rows if r["strict"]["solved"] and not r["loose"]["solved"])

    summary = {
        "n": len(rows),
        "strict_unique_true_solved": strict_solved,
        "loose_unique_false_solved": loose_solved,
        "loose_helps": only_loose,
        "strict_only": only_strict,
    }
    print("\n=== UNIQUE=FALSE TEST SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
