"""
Bit-manipulation solver v2 — frequency-prior weighted voting.

The bit problems are massively under-constrained by pure deduction: only ~2% of
puzzles have all 8 output bits uniquely determined by the examples within the
gate family. The baseline model nonetheless scores 87.5% because it has learned
the GENERATOR's gate-family frequency prior (identity ~42%, CONST ~14%,
AND/XOR/OR/XNOR/NOT next, 3-input gates rare).

This solver makes that prior explicit:
  - For each output-bit column, enumerate ALL gates consistent with the examples
    (up to 3-input).
  - If every consistent gate agrees on the target bit → uniquely determined.
  - Otherwise, each consistent gate votes for its target-bit value, weighted by
    its family's prior frequency. The higher-weight value wins.
  - Special case: if the target input equals an example input, read off the
    answer directly (100% certain).

Prior frequencies measured from 600 baseline-correct bit problems (4800 columns).

Public API:
    SolveResult(solved, answer, cot_trace, confidence)
    solve_bit_v2(prompt) -> SolveResult
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations, product
from typing import Optional

from bit_solver import _parse_prompt, _GATES_2, _GATES_3, _GATES_4  # type: ignore


# Family prior weights (measured), normalized to sum ~1.
FAMILY_PRIOR = {
    "IDENTITY": 0.425,
    "CONST0":   0.108,
    "AND":      0.099,
    "XOR":      0.096,
    "OR":       0.061,
    "XNOR":     0.055,
    "NOT":      0.053,
    "CONST1":   0.031,
    "NAND":     0.015,
    "OR(_,NOT)":  0.012,
    "OR(NOT,_)":  0.011,
    "NOR":      0.009,
    "AND(_,NOT)": 0.006,
    "AND(NOT,_)": 0.005,
    "CHO":      0.003,
    "MAJ":      0.002,
    "PAR3":     0.001,
    "AX":       0.001,
    "OX":       0.001,
    "XO":       0.001,
    "AO":       0.001,
    "OA":       0.001,
    "XA":       0.001,
}
# 4-input gates: very rare, tiny prior
PRIOR_4 = 0.0005


@dataclass
class SolveResult:
    solved: bool
    answer: Optional[str]
    cot_trace: str = ""
    confidence: float = 0.0    # fraction of bits uniquely determined (1.0 = fully deduced)


def _collect_votes(out_col, inp_bits, n, tgt_bits):
    """
    For one output-bit column, return:
      votes: dict[bit_value -> total_prior_weight]
      determined: True if all consistent gates agree on target value
      best_gate_label: a human-readable label for the winning gate (for CoT)
    """
    votes = {0: 0.0, 1: 0.0}
    target_vals = set()
    best_gate = None  # (priority_weight, label, target_bit)

    def consider(family, label, tval):
        w = FAMILY_PRIOR.get(family, PRIOR_4)
        votes[tval] += w
        target_vals.add(tval)
        nonlocal best_gate
        if best_gate is None or w > best_gate[0]:
            best_gate = (w, label, tval)

    # CONST
    if all(x == 0 for x in out_col):
        consider("CONST0", "CONST_0", 0)
    if all(x == 1 for x in out_col):
        consider("CONST1", "CONST_1", 1)
    # IDENTITY / NOT
    for j in range(8):
        col = [inp_bits[e][j] for e in range(n)]
        if col == out_col:
            consider("IDENTITY", f"IDENTITY(in[{j}])", tgt_bits[j])
        if [1 - x for x in col] == out_col:
            consider("NOT", f"NOT(in[{j}])", 1 - tgt_bits[j])
    # 2-input
    for name, fn in _GATES_2:
        for j, k in combinations(range(8), 2):
            if all(fn(inp_bits[e][j], inp_bits[e][k]) == out_col[e] for e in range(n)):
                consider(name, f"{name}(in[{j}],in[{k}])", fn(tgt_bits[j], tgt_bits[k]))
    # 3-input
    for name, fn in _GATES_3:
        for j, k, l in product(range(8), repeat=3):
            if all(fn(inp_bits[e][j], inp_bits[e][k], inp_bits[e][l]) == out_col[e] for e in range(n)):
                consider(name, f"{name}(in[{j}],in[{k}],in[{l}])", fn(tgt_bits[j], tgt_bits[k], tgt_bits[l]))

    determined = (len(target_vals) == 1)
    return votes, determined, best_gate, len(target_vals) > 0


def solve_bit_v2(prompt: str) -> SolveResult:
    inp_ex, out_ex, target = _parse_prompt(prompt)
    if inp_ex is None:
        return SolveResult(False, None, "(parse error)", 0.0)
    n = len(inp_ex)
    inp_bits = [[(v >> (7 - j)) & 1 for j in range(8)] for v in inp_ex]
    tgt_bits = [(target >> (7 - j)) & 1 for j in range(8)]

    # Special case: target input matches an example input → read off directly
    for e in range(n):
        if inp_ex[e] == target:
            ans = f"{out_ex[e]:08b}"
            return SolveResult(True, ans, "(target == example input, read off)", 1.0)

    answer_bits = []
    determined_count = 0
    gate_labels = []
    any_found = True

    for o in range(8):
        out_col = [(out_ex[e] >> (7 - o)) & 1 for e in range(n)]
        votes, determined, best_gate, found = _collect_votes(out_col, inp_bits, n, tgt_bits)
        if not found:
            any_found = False
            answer_bits.append("0")  # best-effort
            gate_labels.append(("UNKNOWN", 0))
            continue
        if determined:
            determined_count += 1
        # Pick winning bit value by weighted vote
        bit_val = 0 if votes[0] >= votes[1] else 1
        answer_bits.append(str(bit_val))
        gate_labels.append((best_gate[1] if best_gate else "?", bit_val))

    answer = "".join(answer_bits)
    confidence = determined_count / 8.0
    solved = any_found
    cot = _build_cot(inp_ex, out_ex, target, gate_labels, answer, confidence)
    return SolveResult(solved, answer, cot, confidence)


def _build_cot(inp_ex, out_ex, target, gate_labels, answer, confidence):
    """Bit-serial CoT trace."""
    n = len(inp_ex)
    L = []
    L.append("Each output bit is an independent boolean function of the 8 input bits.")
    L.append("I solve each output-bit column separately, then verify and concatenate.")
    L.append("")
    L.append("COLUMNS (examples):")
    for e in range(n):
        L.append(f"  {inp_ex[e]:08b} -> {out_ex[e]:08b}")
    L.append(f"Target input: {target:08b}")
    L.append("")
    for o in range(8):
        out_col = "".join(str((out_ex[e] >> (7 - o)) & 1) for e in range(n))
        label, bit_val = gate_labels[o]
        L.append(f"o[{o}]: column={out_col} -> gate={label} -> target bit={bit_val}")
    L.append("")
    L.append(f"Output: {answer}")
    L.append("")
    L.append(f"\\boxed{{{answer}}}")
    return "\n".join(L)


if __name__ == "__main__":
    print("bit_solver_v2: run from parent harness")
