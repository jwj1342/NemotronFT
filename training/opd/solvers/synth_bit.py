"""
synth_bit.py — synthetic bit_manipulation problem generator (DRAFT, not for training).

Purpose
-------
Reverse-engineered from the Kaggle bit_manipulation data (eval shards + train.csv,
1602 problems). Generates self-labeled synthetic puzzles that match the real
generator's structure, so they could be used to *calibrate the model's gate-family
prior* (e.g. small RFT/SFT mix or prior-probe), NOT to replace the model.

Reverse-engineered generator facts (all measured, see bit_diag*.py):
  * Each output bit is an INDEPENDENT boolean function of the 8 input bits.
  * Inputs are uniform random over all 256 bytes (P(bit=1)=~0.50 at every MSB
    position; all 256 byte values observed; inputs i.i.d. across examples).
  * #examples per problem in {7,8,9,10}, roughly uniform.
  * The query target is NEVER equal to any example input (0/1602) — generator
    rejects collisions, so a "read-off" shortcut never applies.
  * Gate family ≤ 3-input almost always; a tiny tail (6/1600 failure columns,
    ~0.4%) needs >4-input functions. We model ≤3-input + an optional rare
    4-input tail.
  * Gate-family prior (lowest-arity canonical estimate from uniquely-determined
    columns, GOOD sample of 300 problems / 468 resolvable columns):
        IDENTITY ~24%, XNOR ~22%, XOR ~15%, NOT ~9%, AND ~7%, OR ~5%,
        OR(_,NOT)/OR(NOT,_) ~8% combined, CONST0 ~4%, CHO ~1.5%, rest <1%.
    (This is consistent with bit_solver_v2.FAMILY_PRIOR, which was measured on
    4800 columns; we reuse a blended prior below.)

IMPORTANT honesty note: the round-trip self-consistency check below verifies the
SYNTH problem is well-formed (enumerator finds the planted gate among the
consistent set). It does NOT verify the gate is *uniquely* recoverable — like the
real data, most planted gates are 1-of-many consistent gates given only 7-10
examples (under-constrained). That under-constraint is a faithful property of the
real generator, not a bug in this synth.
"""
from __future__ import annotations

import random
from itertools import combinations, product

# Reuse the canonical gate tables from the project's solver.
import sys
sys.path.insert(0, "/scratch/jwj/NemotronFT/training/opd/solvers")
from bit_solver import _GATES_2, _GATES_3, _GATES_4  # noqa: E402

# Blended gate-family prior (sum normalized at sample time). Keys are the
# (family, arity) gate generators.
_PRIOR = {
    "IDENTITY": 0.30,
    "XNOR": 0.16,
    "XOR": 0.13,
    "NOT": 0.07,
    "AND": 0.07,
    "OR": 0.05,
    "OR(_,NOT)": 0.04,
    "OR(NOT,_)": 0.03,
    "CONST0": 0.04,
    "CONST1": 0.015,
    "NAND": 0.015,
    "NOR": 0.01,
    "AND(_,NOT)": 0.01,
    "AND(NOT,_)": 0.01,
    "CHO": 0.015,
    "MAJ": 0.01,
    "PAR3": 0.008,
    "AX": 0.005,
    "OX": 0.005,
    "XO": 0.005,
    "AO": 0.004,
    "OA": 0.004,
    "XA": 0.004,
}

_G2 = {n: f for n, f in _GATES_2}
_G3 = {n: f for n, f in _GATES_3}


def _sample_gate(rng: random.Random):
    """Sample a gate (family, callable taking the 8-bit input list) per the prior."""
    fams = list(_PRIOR)
    weights = [_PRIOR[f] for f in fams]
    fam = rng.choices(fams, weights=weights, k=1)[0]
    if fam == "CONST0":
        return fam, (lambda bits: 0)
    if fam == "CONST1":
        return fam, (lambda bits: 1)
    if fam == "IDENTITY":
        j = rng.randrange(8)
        return f"IDENTITY(in[{j}])", (lambda bits, j=j: bits[j])
    if fam == "NOT":
        j = rng.randrange(8)
        return f"NOT(in[{j}])", (lambda bits, j=j: 1 - bits[j])
    if fam in _G2:
        fn = _G2[fam]
        j, k = rng.sample(range(8), 2)
        return f"{fam}(in[{j}],in[{k}])", (lambda bits, fn=fn, j=j, k=k: fn(bits[j], bits[k]))
    if fam in _G3:
        fn = _G3[fam]
        j = rng.randrange(8); k = rng.randrange(8); l = rng.randrange(8)
        return f"{fam}(in[{j}],in[{k}],in[{l}])", (lambda bits, fn=fn, j=j, k=k, l=l: fn(bits[j], bits[k], bits[l]))
    # fallback
    j = rng.randrange(8)
    return f"IDENTITY(in[{j}])", (lambda bits, j=j: bits[j])


def generate_problem(rng: random.Random):
    """Generate one synthetic bit problem with ground-truth labels.

    Returns dict with: examples [(in_byte,out_byte)], target (byte), answer (8-char str),
    gates [per-output-bit family labels].
    """
    n_ex = rng.choice([7, 8, 9, 10])
    gates = [_sample_gate(rng) for _ in range(8)]  # one gate per output-bit position

    def apply(byte: int) -> int:
        bits = [(byte >> (7 - j)) & 1 for j in range(8)]
        out = 0
        for o in range(8):
            _, fn = gates[o]
            out |= (fn(bits) & 1) << (7 - o)
        return out

    # i.i.d. uniform inputs, distinct example inputs
    seen = set()
    examples = []
    while len(examples) < n_ex:
        b = rng.randrange(256)
        if b in seen:
            continue
        seen.add(b)
        examples.append((b, apply(b)))
    # target distinct from all examples (matches generator: 0/1602 collisions)
    while True:
        t = rng.randrange(256)
        if t not in seen:
            break
    answer = f"{apply(t):08b}"
    return {
        "examples": examples,
        "target": t,
        "answer": answer,
        "gates": [g[0] for g in gates],
    }


def render_prompt(prob) -> str:
    lines = [
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
        "binary numbers. The transformation involves operations like bit shifts, "
        "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.",
        "",
        "Here are some examples of input -> output:",
    ]
    for inp, out in prob["examples"]:
        lines.append(f"{inp:08b} -> {out:08b}")
    lines.append("")
    lines.append(f"Now, determine the output for: {prob['target']:08b}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Round-trip self-consistency check: the bit_diag enumerator must find the
# planted gate's *value* among the consistent set for every output column.
# (Under-constraint means MANY gates are consistent; we only assert the planted
# target value is in the consistent value-set, i.e. the problem is well-formed.)
# --------------------------------------------------------------------------

def _enum_consistent_values(out_col, inp_bits, n, tgt_bits):
    vals = set()
    if all(x == 0 for x in out_col):
        vals.add(0)
    if all(x == 1 for x in out_col):
        vals.add(1)
    for j in range(8):
        col = [inp_bits[e][j] for e in range(n)]
        if col == out_col:
            vals.add(tgt_bits[j])
        if [1 - x for x in col] == out_col:
            vals.add(1 - tgt_bits[j])
    for _, fn in _GATES_2:
        for j, k in combinations(range(8), 2):
            if all(fn(inp_bits[e][j], inp_bits[e][k]) == out_col[e] for e in range(n)):
                vals.add(fn(tgt_bits[j], tgt_bits[k]))
    for _, fn in _GATES_3:
        for j, k, l in product(range(8), repeat=3):
            if all(fn(inp_bits[e][j], inp_bits[e][k], inp_bits[e][l]) == out_col[e] for e in range(n)):
                vals.add(fn(tgt_bits[j], tgt_bits[k], tgt_bits[l]))
    return vals


def roundtrip_check(prob) -> bool:
    inp_ex = [e[0] for e in prob["examples"]]
    out_ex = [e[1] for e in prob["examples"]]
    target = prob["target"]
    n = len(inp_ex)
    inp_bits = [[(v >> (7 - j)) & 1 for j in range(8)] for v in inp_ex]
    tgt_bits = [(target >> (7 - j)) & 1 for j in range(8)]
    for o in range(8):
        out_col = [(out_ex[e] >> (7 - o)) & 1 for e in range(n)]
        true_bit = int(prob["answer"][o])
        vals = _enum_consistent_values(out_col, inp_bits, n, tgt_bits)
        if true_bit not in vals:
            return False  # planted gate not even in the (≤3-input) consistent set
    return True


if __name__ == "__main__":
    rng = random.Random(20260602)
    N = 300
    ok = 0
    uniq_cols = 0
    total_cols = 0
    for _ in range(N):
        prob = generate_problem(rng)
        if roundtrip_check(prob):
            ok += 1
        # measure determinability (same under-constraint as real data)
        inp_ex = [e[0] for e in prob["examples"]]
        out_ex = [e[1] for e in prob["examples"]]
        target = prob["target"]
        n = len(inp_ex)
        inp_bits = [[(v >> (7 - j)) & 1 for j in range(8)] for v in inp_ex]
        tgt_bits = [(target >> (7 - j)) & 1 for j in range(8)]
        for o in range(8):
            out_col = [(out_ex[e] >> (7 - o)) & 1 for e in range(n)]
            vals = _enum_consistent_values(out_col, inp_bits, n, tgt_bits)
            total_cols += 1
            if len(vals) == 1:
                uniq_cols += 1
    print(f"generated {N} synthetic problems")
    print(f"round-trip self-consistent (planted value in ≤3-input consistent set): {ok}/{N}")
    print(f"uniquely-determined columns: {uniq_cols}/{total_cols} ({uniq_cols/total_cols*100:.1f}%) "
          f"(real failures: 213/1600=13.3%; under-constraint reproduced)")
    print("\nSample rendered prompt:\n")
    print(render_prompt(generate_problem(random.Random(7))))
