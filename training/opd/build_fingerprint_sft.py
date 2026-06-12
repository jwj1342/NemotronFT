"""Build the unified operator-fingerprinting SFT set: teach the model to READ each
disguised operator from the examples (test candidate ops; reject conventional glyph
meanings) instead of assuming. Covers equation_numeric (plain digits) + cryptarithm
concat/rconcat (symbol passthrough). Synthetic eqn for volume/generalization (clean
str encoding, self-verified) + real problems (rendered, kept only when the trace's
prediction matches the true answer -> no poison).
Output: data/fingerprint_sft.csv (id,prompt,answer,generated_cot,type) + a held-out
data/fingerprint_holdout.jsonl for net-gain eval.
"""
from __future__ import annotations
import sys, json, glob, csv, random
from pathlib import Path
PROJECT = Path("/scratch/jwj/NemotronFT")
sys.path.insert(0, str(PROJECT / "training/opd/solvers"))
from crd_solver import _apply_op  # type: ignore
from eqn_render_forward import render_eqn  # type: ignore
from crd_render_forward import render_concat  # type: ignore

RNG = random.Random(20260610)
GLYPHS = list("!@#$%^&*+-/<>?:`~|][}{")  # non-digit operator symbols
# ops with clean str() encoding (positives str, negatives "-N")
SYN_OPS = ["add", "sub", "rsub", "abs_sub", "mul", "concat", "rconcat", "maxmod", "add+1", "mul-1"]


def _enc(v: int) -> str:
    return str(v)  # competition writes plain ints (negatives as -N), matches "25-97 = -72"


def gen_synthetic_eqn():
    """One synthetic equation_numeric_deduce problem: 2-3 ops, each with >=2 examples,
    query uses one of them (deducible). Returns (prompt, answer) or None."""
    nops = RNG.randint(2, 3)
    ops = RNG.sample(SYN_OPS, nops)
    glyphs = RNG.sample(GLYPHS, nops)
    op_of = dict(zip(glyphs, ops))
    lines = []
    for g, op in op_of.items():
        for _ in range(RNG.randint(2, 3)):
            a, b = RNG.randint(10, 99), RNG.randint(10, 99)
            v = _apply_op(op, a, b)
            if v is None:
                continue
            lines.append(f"{a}{g}{b} = {_enc(v)}")
    if len(lines) < 4:
        return None
    RNG.shuffle(lines)
    qg = RNG.choice(glyphs)
    qa, qb = RNG.randint(10, 99), RNG.randint(10, 99)
    qv = _apply_op(op_of[qg], qa, qb)
    if qv is None:
        return None
    prompt = ("In Alice's Wonderland, a secret set of transformation rules is applied to "
              "equations. Below are a few examples:\n" + "\n".join(lines) +
              f"\nNow, determine the result for: {qa}{qg}{qb}")
    return prompt, _enc(qv)


def main():
    csv.field_size_limit(10**7)
    prompts = {r["id"]: r["prompt"] for r in csv.DictReader(open(PROJECT / "resource/winning-solution/train.csv"))}
    rows, holdout = [], []
    n_syn_ok = n_syn_try = 0

    # 1) synthetic eqn (volume + generalization), self-verified
    while n_syn_ok < 4000 and n_syn_try < 20000:
        n_syn_try += 1
        g = gen_synthetic_eqn()
        if not g:
            continue
        prompt, ans = g
        out = render_eqn(prompt, ans)
        if out and out[1] == ans:  # trace's prediction matches -> correct, learnable
            rows.append({"id": f"syn_eqn_{n_syn_ok}", "prompt": prompt, "answer": ans,
                         "generated_cot": out[0], "type": "equation_numeric_deduce"})
            n_syn_ok += 1
    print(f"[synthetic eqn] {n_syn_ok} traces (from {n_syn_try} tries)")

    # 2) real problems: render fingerprint/passthrough traces, keep only correct ones
    real_eqn = real_concat = 0
    for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
        for line in open(shard):
            if not line.strip():
                continue
            r = json.loads(line)
            cat = r["category"]; pid = r["id"]
            if pid not in prompts:
                continue
            ans = str(r["answer"]); prompt = prompts[pid]
            out = None; typ = None
            if cat == "equation_numeric_deduce":
                out = render_eqn(prompt, ans); typ = cat
            elif cat.startswith("cryptarithm"):
                out = render_concat(prompt, ans); typ = "cryptarithm_deduce"
            if out and out[1] == ans:
                rec = {"id": pid, "prompt": prompt, "answer": ans,
                       "generated_cot": out[0], "type": typ}
                # hold out problems the BASELINE got WRONG (the net-gain test set)
                if not r["correct"]:
                    holdout.append({"id": pid, "prompt": prompt, "gt": ans,
                                    "category": cat, "greedy_correct": False})
                else:
                    rows.append(rec)
                if cat.startswith("cryptarithm"):
                    real_concat += 1
                else:
                    real_eqn += 1
    print(f"[real] eqn-deduce rendered={real_eqn}  crypt-concat rendered={real_concat}")

    RNG.shuffle(rows)
    OUT = PROJECT / "data/fingerprint_sft.csv"
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot", "type"])
        w.writeheader()
        w.writerows(rows)
    HOLD = PROJECT / "data/fingerprint_holdout.jsonl"
    with open(HOLD, "w") as f:
        for h in holdout:
            f.write(json.dumps(h) + "\n")
    from collections import Counter
    print(f"[done] {len(rows)} train rows -> {OUT}   ({dict(Counter(r['type'] for r in rows))})")
    print(f"[done] {len(holdout)} baseline-WRONG holdout (recoverable test) -> {HOLD}")


if __name__ == "__main__":
    main()
