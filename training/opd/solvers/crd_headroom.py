"""Quantify the stateless-probe headroom: for every cryptarithm problem, classify the
QUERY's operation (via crd_solver's deduced op_map) and cross with baseline correctness.
RECOVERABLE = solver can solve it (so we can render a correct stateless trace) AND the
baseline model got it wrong. Tells us which op families (esp concat/rconcat — pure
symbol passthrough) carry the headroom for the expert's stateless-feature-probe CoT.
"""
from __future__ import annotations
import sys, json, glob, csv
from collections import defaultdict, Counter
from pathlib import Path
PROJECT = Path("/scratch/jwj/NemotronFT")
sys.path.insert(0, str(PROJECT / "training/opd/solvers"))
from crd_solver import solve, parse_prompt  # type: ignore

csv.field_size_limit(10**7)
prompts = {}
for r in csv.DictReader(open(PROJECT / "resource/winning-solution/train.csv")):
    prompts[r["id"]] = r["prompt"]

stats = defaultdict(lambda: {"total": 0, "baseOK": 0, "solverOK": 0, "recoverable": 0})
catn = Counter()
rows = []
for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
    for line in open(shard):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["category"].startswith("cryptarithm") or r["id"] not in prompts:
            continue
        catn[r["category"]] += 1
        rows.append((r["id"], prompts[r["id"]], str(r["answer"]), bool(r["correct"]), r["category"]))

print(f"running solver on {len(rows)} cryptarithm problems ...", flush=True)
for i, (pid, prompt, ans, base_ok, cat) in enumerate(rows):
    try:
        res = solve(prompt, ans)
        facts, query, op_syms = parse_prompt(prompt)
        qop = res.get("op_map", {}).get(query[1], "UNDEDUCED") if query else "PARSEFAIL"
        solver_ok = bool(res.get("correct"))
    except Exception as e:
        qop, solver_ok = "ERROR", False
    s = stats[qop]
    s["total"] += 1
    s["baseOK"] += base_ok
    s["solverOK"] += solver_ok
    if solver_ok and not base_ok:
        s["recoverable"] += 1
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(rows)}", flush=True)

print("\ncategory counts:", dict(catn))
print(f"\n{'query_op':12s} {'total':>6s} {'baseOK':>7s} {'baseAcc':>8s} {'solverOK':>9s} {'RECOVERABLE':>12s}")
tot_recov = 0
for op in sorted(stats, key=lambda o: -stats[o]["recoverable"]):
    s = stats[op]
    tot_recov += s["recoverable"]
    acc = s["baseOK"] / s["total"] * 100 if s["total"] else 0
    print(f"{op:12s} {s['total']:>6d} {s['baseOK']:>7d} {acc:>7.1f}% {s['solverOK']:>9d} {s['recoverable']:>12d}")
print(f"\nTOTAL recoverable (solver-solvable & baseline-wrong) = {tot_recov} = {tot_recov/9500*100:.2f}pp of benchmark")
concat_recov = stats["concat"]["recoverable"] + stats["rconcat"]["recoverable"]
print(f"  of which concat/rconcat (purely stateless) = {concat_recov} = {concat_recov/9500*100:.2f}pp")
