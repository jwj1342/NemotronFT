"""Cross-check: do problems the branch audit flags as 'converging' (branch_width
not None) agree with the reference solver's correctness/reachability? And how
many converge total vs solver-correct (sanity on the per-pp convergence notion)."""
import sys, os, json, glob, csv
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crd_solver as S
import crd_branch_audit as A

PROJECT = Path("/scratch/jwj/NemotronFT")
ids = set()
for f in glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        if d.get("category") == "cryptarithm_deduce":
            ids.add(d["id"])
rows = {}
with open(PROJECT / "data/train.csv") as f:
    for r in csv.DictReader(f):
        if r["id"] in ids:
            rows[r["id"]] = r

conv = 0
conv_solver_correct = 0
conv_solver_reachable = 0
conv_solver_wrong = 0
for pid, v in rows.items():
    facts, query, opsyms = S.parse_prompt(v["prompt"])
    if facts is None:
        continue
    sign = S._find_sign_symbol(facts, opsyms)
    fa = S._digit_alphabet(facts, query, None, opsyms)
    res = A.best_branch_width(facts, query, opsyms, sign, v["answer"].strip(), fa, 20)
    if res["branch_width"] is not None:
        conv += 1
        sv = S.solve(v["prompt"], v["answer"].strip())
        if sv.get("correct"):
            conv_solver_correct += 1
        elif sv.get("reachable"):
            conv_solver_reachable += 1
        else:
            conv_solver_wrong += 1
            print(f"  MISMATCH id={pid}: audit converges but solver not reachable "
                  f"(solver fail={sv.get('fail_reason')}, n_pred={sv.get('n_distinct_predictions')})")

print(f"\naudit-converging (bw<=20): {conv}")
print(f"  of which solver-correct:   {conv_solver_correct}")
print(f"  of which solver-reachable: {conv_solver_reachable}")
print(f"  of which solver-unreach:   {conv_solver_wrong}")
