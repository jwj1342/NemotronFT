"""F-bucket diagnostic — Codex's recommendation.

Goal: on a stratified sample of 100 F-bucket puzzles, label each as
  - `solved`            : prediction matches GT under some config
  - `multi_feasible`    : search produced multiple candidate answers, none right
  - `no_feasible`       : search exhausted with zero candidates
  - `timeout_or_cap`    : hit branch cap before exhausting search

Run two configs (NOT three — keep noise down):
  1. CURR: base∈{8..12}, branch_cap=1.5M, unique=True   (production setting)
  2. WIDE: base∈{7..16}, branch_cap=10M,  unique=True   (broader budget)

If `WIDE` solves substantially more → bottleneck is budget/base.
If `WIDE` also fails → bottleneck is solver family (need structural change).

unique=False stays OFF (Codex: it pollutes the diagnostic).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOLVER_DIR = PROJECT_ROOT / "training" / "opd" / "solvers"
sys.path.insert(0, str(SOLVER_DIR))

from measurements import measure, parse_prompt  # type: ignore
import extended_solver as ES  # type: ignore


PROBLEMS_DIR = PROJECT_ROOT / "resource" / "winning-solution" / "problems"
PROBLEMS_FILE = PROJECT_ROOT / "resource" / "winning-solution" / "problems.jsonl"
PROBE_RESULTS = SOLVER_DIR / "runs" / "probe_v2_results.jsonl"
OUT = SOLVER_DIR / "runs" / "f_diagnostic.jsonl"
SUMMARY = SOLVER_DIR / "runs" / "f_diagnostic_summary.json"


def _is_f_bucket(parsed) -> bool:
    m = measure(parsed)
    if m.is_pure_concat or m.is_pure_revconcat or m.position_lock:
        return False
    if m.alphabet_size > 12:
        return False
    if m.rhs_in_lhs_rate > 0.85:
        return False
    if m.alphabet_size <= 12 and not m.rhs_len_uniform:
        return True
    return False


def _solve_with_config(data, configs, branch_cap):
    """Try all (base, post_reverse) in configs; return summary dict."""
    gt = str(data["answer"])
    total_branches = 0
    hit_cap = False
    total_candidates = 0  # union of unique candidate answers across configs
    candidate_set: set = set()
    predicted = None
    for base, post_rev in configs:
        try:
            pred, info = _solve_problem_instrumented(
                data, base=base, post_reverse=post_rev, branch_cap=branch_cap
            )
        except Exception:
            continue
        total_branches += getattr(_solve_problem_instrumented, "_last_branches", 0)
        if getattr(_solve_problem_instrumented, "_last_hit_cap", False):
            hit_cap = True
        cands = getattr(_solve_problem_instrumented, "_last_candidates", [])
        for c in cands:
            candidate_set.add(c)
        if pred is not None and predicted is None:
            predicted = pred
        if pred == gt:
            return {
                "solved": True,
                "predicted": pred,
                "hit_cap": hit_cap,
                "branches": total_branches,
                "n_candidates": len(candidate_set),
            }
    return {
        "solved": False,
        "predicted": predicted,
        "hit_cap": hit_cap,
        "branches": total_branches,
        "n_candidates": len(candidate_set),
    }


def _solve_problem_instrumented(data, base, post_reverse, branch_cap):
    """Wrap extended_solver.solve_problem but record branch count / hit_cap /
    candidate-answer set in module-level slots."""
    _solve_problem_instrumented._last_branches = 0
    _solve_problem_instrumented._last_hit_cap = False
    _solve_problem_instrumented._last_candidates = []

    examples = []
    op_syms = []
    for ex in data["examples"]:
        inp = ex["input_value"]
        out = ex["output_value"]
        if len(inp) != 5:
            continue
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))
        op_syms.append(inp[2])
    query = tuple(data["question"])
    if len(query) != 5:
        return None, ({}, {})

    # Pure-concat detection same as extended_solver.solve_problem
    def _is_trivial_concat(ex, b):
        s0, s1, _, s3, s4, rsyms = ex
        if len(rsyms) != 4:
            return None
        if rsyms == (s0, s1, s3, s4):
            return "concat"
        if rsyms == (s3, s4, s0, s1):
            return "rev_concat"
        return None

    concat_ops = {}
    nonconcat = set()
    for ex in examples:
        kind = _is_trivial_concat(ex, base)
        if kind is not None:
            concat_ops.setdefault(ex[2], kind)
        else:
            nonconcat.add(ex[2])
    concat_only = {op: k for op, k in concat_ops.items() if op not in nonconcat}

    q_op = query[2]
    if q_op in concat_only:
        kind = concat_only[q_op]
        if kind == "concat":
            return query[0] + query[1] + query[3] + query[4], ({}, {q_op: "concat"})
        return query[3] + query[4] + query[0] + query[1], ({}, {q_op: "rev_concat"})

    arith = [ex for ex in examples if ex[2] not in concat_only]
    if not arith:
        return None, ({}, {})

    solver = ES.ExtendedSolver(arith, query, base=base, post_reverse=post_reverse, unique=True)
    solver.branch_cap = branch_cap

    try:
        solver._process(0)
    except ES._BranchCap:
        _solve_problem_instrumented._last_hit_cap = True

    _solve_problem_instrumented._last_branches = solver.branch_count
    _solve_problem_instrumented._last_candidates = list(solver.answers.keys())
    if solver.answers:
        best, _ = solver.answers.most_common(1)[0]
        return best, solver.answer_info.get(best, ({}, {}))
    return None, ({}, {})


CURR_CONFIGS = [(10, False), (10, True), (9, False), (9, True),
                (11, False), (8, False), (12, False)]
WIDE_CONFIGS = [(10, False), (10, True), (9, False), (9, True),
                (11, False), (8, False), (12, False),
                (7, False), (13, False), (16, False), (16, True)]


def _worker(task):
    pid = task["id"]
    data = task["data"]
    res_curr = _solve_with_config(data, CURR_CONFIGS, branch_cap=1_500_000)
    res_wide = _solve_with_config(data, WIDE_CONFIGS, branch_cap=10_000_000)
    return {
        "id": pid,
        "gt": str(data["answer"]),
        "curr": res_curr,
        "wide": res_wide,
    }


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

    print(f"F-bucket candidates: {len(candidates)}")
    sample = random.sample(candidates, k=min(100, len(candidates)))
    print(f"Sampling: {len(sample)}")

    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    t0 = time.time()
    results = []
    with mp.Pool(nw) as pool, open(OUT, "w") as f_out:
        for i, r in enumerate(pool.imap_unordered(_worker, sample, chunksize=1), 1):
            results.append(r)
            f_out.write(json.dumps(r) + "\n")
            f_out.flush()
            if i % 10 == 0 or i == len(sample):
                dt = time.time() - t0
                print(f"  {i}/{len(sample)} elapsed={dt:.1f}s", flush=True)

    # Aggregate
    def label(r_cfg):
        if r_cfg["solved"]:
            return "solved"
        if r_cfg["n_candidates"] == 0 and r_cfg["hit_cap"]:
            return "timeout_or_cap"
        if r_cfg["n_candidates"] == 0:
            return "no_feasible"
        return "multi_feasible"

    labels_curr = Counter(label(r["curr"]) for r in results)
    labels_wide = Counter(label(r["wide"]) for r in results)
    moved = sum(1 for r in results if (not r["curr"]["solved"]) and r["wide"]["solved"])

    summary = {
        "n_sample": len(results),
        "curr_labels": dict(labels_curr),
        "wide_labels": dict(labels_wide),
        "wide_helps_count": moved,
    }
    print("\n=== F DIAGNOSTIC SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
