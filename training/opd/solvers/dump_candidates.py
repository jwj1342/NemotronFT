"""Dump full candidate-answer set for selected multi_feasible puzzles.
For Codex's structural diagnosis: see what kinds of ambiguity the solver
runs into.
"""
import json
import sys
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")
sys.path.insert(0, str(PROJECT / "training" / "opd" / "solvers"))
import extended_solver as ES  # type: ignore

PROBLEMS = PROJECT / "resource" / "winning-solution" / "problems"

# Picks: many-cands, few-cands, structural-outlier
PICKS = ["844f826c", "a1230047", "82109e34"]

CONFIGS = [(10, False), (10, True), (9, False), (9, True),
           (11, False), (8, False), (12, False)]


def solve_all(data, configs, unique):
    examples = []
    for ex in data["examples"]:
        inp = ex["input_value"]
        out = ex["output_value"]
        if len(inp) != 5:
            continue
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))
    query = tuple(data["question"])
    union_candidates: dict[str, int] = {}
    info_by_cand: dict[str, dict] = {}
    for base, post in configs:
        solver = ES.ExtendedSolver(examples, query, base=base, post_reverse=post, unique=unique)
        solver.branch_cap = 500_000
        try:
            solver._process(0)
        except ES._BranchCap:
            pass
        for cand, ct in solver.answers.items():
            union_candidates[cand] = union_candidates.get(cand, 0) + ct
            if cand not in info_by_cand:
                mapping, op_assign = solver.answer_info.get(cand, ({}, {}))
                info_by_cand[cand] = {"base": base, "post_rev": post,
                                       "mapping": dict(mapping), "op_assign": dict(op_assign)}
    return union_candidates, info_by_cand


import sys as _sys
_sys.stdout.reconfigure(line_buffering=True)
for pid in PICKS:
    d = json.loads(open(PROBLEMS / f"{pid}.jsonl").readline())
    print(f"\n===== {pid} =====")
    print(f"q_op = {d['question'][2]!r}    gt = {d['answer']!r}")
    print("Examples:")
    for e in d["examples"]:
        print(f"  {e['input_value']} = {e['output_value']}")
    print(f"Query: {d['question']}")
    cands, info = solve_all(d, CONFIGS, unique=True)
    print(f"\nUnique=True candidates: {len(cands)}")
    sorted_cands = sorted(cands.items(), key=lambda kv: -kv[1])[:8]
    for cand, ct in sorted_cands:
        meta = info[cand]
        print(f"  {cand!r:25s} freq={ct:5d}  base={meta['base']:2d} post={meta['post_rev']}  ops={meta['op_assign']}")
    cands2, info2 = solve_all(d, CONFIGS, unique=False)
    print(f"\nUnique=False candidates: {len(cands2)}")
    for cand, ct in sorted(cands2.items(), key=lambda kv: -kv[1])[:8]:
        meta = info2[cand]
        print(f"  {cand!r:25s} freq={ct:5d}  base={meta['base']:2d} post={meta['post_rev']}  ops={meta['op_assign']}")
