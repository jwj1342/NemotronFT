"""Headroom for operator-fingerprinting on equation_numeric (plain digits, NO cipher).
For each problem, take the query glyph's examples and find the op (from the 13-op DSL)
whose MAGNITUDE matches all of them; if unique, the op is determinable by fingerprinting
(the model only had to read the examples, not assume the glyph's conventional meaning).
exact = predicted string also matches the answer (sign/encoding included). Cross with
baseline correctness -> recoverable subset. guess problems (query glyph absent from
examples) should fall out as non-determinable."""
from __future__ import annotations
import sys, json, glob, csv, re
from collections import Counter
from pathlib import Path
PROJECT = Path("/scratch/jwj/NemotronFT")
sys.path.insert(0, str(PROJECT / "training/opd/solvers"))
from crd_solver import _apply_op, ALL_OPS  # type: ignore

csv.field_size_limit(10**7)
prompts = {r["id"]: r["prompt"] for r in csv.DictReader(open(PROJECT / "resource/winning-solution/train.csv"))}
EX = re.compile(r"^\s*(\d+)(\D)(\d+)\s*=\s*(\S+)\s*$")
QY = re.compile(r"determine the result for:\s*(\d+)(\D)(\d+)\s*$")
digits = lambda s: "".join(c for c in str(s) if c.isdigit())

stat = Counter()
det_recov = exact_recov = 0
det_by_cat = Counter(); exact_by_cat = Counter(); wrong_by_cat = Counter()
for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
    for line in open(shard):
        if not line.strip():
            continue
        r = json.loads(line)
        cat = r["category"]
        if not cat.startswith("equation_numeric") or r["id"] not in prompts:
            continue
        p = prompts[r["id"]]
        exs = []
        for ln in p.splitlines():
            m = EX.match(ln)
            if m:
                exs.append((int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)))
        qm = QY.search(p)
        if not qm:
            continue
        qa, qg, qb = int(qm.group(1)), qm.group(2), int(qm.group(3))
        gex = [(a, b, res) for (a, g, b, res) in exs if g == qg]
        base_ok = bool(r["correct"]); ans = str(r["answer"])
        if not base_ok:
            wrong_by_cat[cat] += 1
        cands = []
        for op in ALL_OPS:
            if gex and all(_apply_op(op, a, b) is not None and digits(_apply_op(op, a, b)) == digits(res)
                           for (a, b, res) in gex):
                cands.append(op)
        qmags = {digits(_apply_op(op, qa, qb)) for op in cands if _apply_op(op, qa, qb) is not None}
        determinable = bool(gex) and len(qmags) == 1
        exact = False
        if determinable:
            for op in cands:
                v = _apply_op(op, qa, qb)
                if v is not None and (str(v) == ans or digits(v) == ans or str(v).lstrip("+-") == ans):
                    exact = True
                    break
        if determinable and not base_ok:
            det_recov += 1; det_by_cat[cat] += 1
        if exact and not base_ok:
            exact_recov += 1; exact_by_cat[cat] += 1
        stat[(cat, base_ok, determinable)] += 1

print("=== equation_numeric operator-fingerprinting headroom ===")
print(f"wrong by category: {dict(wrong_by_cat)}")
print(f"op-DETERMINABLE & baseline-wrong (fingerprintable): {det_recov} = {det_recov/9500*100:.2f}pp")
print(f"   by cat: {dict(det_by_cat)}")
print(f"EXACT-predictable & baseline-wrong (recoverable): {exact_recov} = {exact_recov/9500*100:.2f}pp")
print(f"   by cat: {dict(exact_by_cat)}")
print(f"\n(gap det-exact = sign/encoding mismatches, possibly fixable with correct sign rendering)")
print("breakdown (cat, baseline_correct, determinable) -> count:")
for k in sorted(stat):
    print(f"  {k}: {stat[k]}")
