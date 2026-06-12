"""
Deduce salvage broken down by:
  - operator type: plaintext (+,-,*) vs cipher-encoded
  - n_op (examples sharing the question operator)
and computing, per (op_type, n_op) cell:
  - unique / multi / no_solution split
  - grammar_hits_true rate
This sharpens the capacity-vs-info split: plaintext-op + many examples
should be the MOST salvageable cell (operation is known, cipher constrainable).

Fast: uses the early-exit-free enumeration but with a hard per-combo mapping cap
and only enumerates combos until distinct answers stabilize.
"""
from __future__ import annotations
import sys, csv, json, re, random, collections
csv.field_size_limit(10**7)
from pathlib import Path
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from crypt_salvage_analysis import (parse_prompt, analyze_one, load_fails, load_prompts)

PLAINTEXT_OPS = set("+-*")

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nper = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    random.seed(seed)
    fails = load_fails()["cryptarithm_deduce"]
    random.shuffle(fails)
    sample = fails[:nper]
    ids = {r["id"] for r in sample}
    P = load_prompts(ids)
    # cells keyed by (op_type, nop_band)
    cells = collections.defaultdict(lambda: {"n":0,"unique":0,"multi":0,
            "no_solution":0,"ns_target_unseen":0,"ns_grammar_gap":0,
            "grammar_true":0,"unique_eq_true":0})
    for r in sample:
        ex, q, op, nop = parse_prompt(P[r["id"]])
        if ex is None or nop == 0:
            continue
        optype = "plaintext" if op in PLAINTEXT_OPS else "cipher_op"
        band = "1" if nop == 1 else ("2" if nop == 2 else "3+")
        key = (optype, band)
        c = cells[key]
        c["n"] += 1
        res = analyze_one(ex, q, op, r["answer"])
        d = res["distinct_answers"]
        if res["grammar_hits_true"]:
            c["grammar_true"] += 1
        if len(d) == 0:
            c["no_solution"] += 1
            if not res["any_combo_consistent"]:
                c["ns_grammar_gap"] += 1
            elif res["target_symbol_unseen"]:
                c["ns_target_unseen"] += 1
        elif len(d) == 1:
            c["unique"] += 1
            if r["answer"] in d:
                c["unique_eq_true"] += 1
        else:
            c["multi"] += 1
    out = {f"{k[0]}|nop={k[1]}": v for k, v in sorted(cells.items())}
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
