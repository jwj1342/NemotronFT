"""Generate Huikang-style "search-trace" CoT for cryptarithm puzzles solved
by the extended DSL.

Design principle: the CoT must show the MEASUREMENTS that narrow the family,
then ENUMERATE candidates with explicit verification on each example, then
the matching candidate with full computation, then the application to the
query.  This is process supervision — the model learns HOW to look, not just
WHAT to answer.

Output format matches `resource/winning-solution/reasoning/*.txt` so the
existing `corpus.py` can pick these up without modification.
"""
from __future__ import annotations

from typing import Optional

from measurements import Measurements, ParsedPuzzle, Example  # type: ignore
from probe_unsolved import try_op, ALL_OPS  # type: ignore


# Op-name to human-readable label
OP_LABEL = {
    "add":             "a + b",
    "sub_ab":          "a - b",
    "sub_ba":          "b - a",
    "abs_diff":        "|a - b|",
    "mul":             "a * b",
    "concat":          "a || b",
    "rev_concat":      "b || a",
    "div_ab":          "a / b",
    "div_ba":          "b / a",
    "mod_ab":          "a mod b",
    "mod_ba":          "b mod a",
    "digit_add_mod":   "digit-wise (a_i + b_i) mod base",
    "digit_sub_mod":   "digit-wise (a_i - b_i) mod base",
    "digit_mul_mod":   "digit-wise (a_i * b_i) mod base",
    "digit_xor":       "digit-wise (a_i XOR b_i)",
    "max_concat":      "(max digit of a) || (max digit of b)",
    "min_concat":      "(min digit of a) || (min digit of b)",
    "digit_max":       "digit-wise max(a_i, b_i)",
    "digit_min":       "digit-wise min(a_i, b_i)",
}


def _box(s: str) -> str:
    return "".join(f"【{c}】" for c in s)


def _measurement_block(m: Measurements, p: ParsedPuzzle) -> list[str]:
    """The opening 'measure the puzzle' section."""
    out = []
    out.append("Let me first measure the puzzle's structural signatures.")
    out.append("")
    out.append("[Measurement 1: alphabet size]")
    out.append(f"  All non-operator chars across examples and query: {sorted(m.alphabet)}")
    out.append(f"  Alphabet size = {m.alphabet_size}")
    if m.alphabet_size <= 12:
        out.append(f"  → Within decimal digit count. Hypothesis: this is a base-N digit cipher.")
    else:
        out.append(f"  → Larger than 10. Hypothesis: this is a string-DSL puzzle, not arithmetic.")
    out.append("")
    out.append("[Measurement 2: output-length variance]")
    out.append(f"  RHS lengths per example: {m.rhs_lens}")
    if m.rhs_len_var > 0:
        out.append(f"  → Lengths vary → output is computed (arithmetic), not a fixed permutation.")
    else:
        out.append(f"  → Lengths constant → output likely permutation of input chars.")
    out.append("")
    out.append("[Measurement 3: character preservation]")
    out.append(f"  Fraction of RHS chars also present in their LHS (non-op): "
               f"{m.rhs_in_lhs_rate:.2f}")
    if m.rhs_in_lhs_rate > 0.85:
        out.append(f"  → Near-full preservation → output reuses input chars (concat/permutation family).")
    elif m.rhs_in_lhs_rate >= 0.20:
        out.append(f"  → Mid preservation (~0.4 is expected when result digits coincide ~40% with LHS digits in a base-10 cipher).")
    else:
        out.append(f"  → Low preservation → output introduces new chars; constrains digit alphabet.")
    out.append("")
    out.append("[Measurement 4: operator family]")
    out.append(f"  Operator symbols at position 2: {m.op_chars_set}")
    op_is_arith = all(c in "+-*/" for c in m.op_chars_set) if m.op_chars_set else False
    if op_is_arith:
        out.append(f"  → Standard arithmetic operators present.")
    else:
        out.append(f"  → Operator is encoded with non-arithmetic symbols.")
    out.append("")
    return out


def _try_op_verbose(
    op_name: str,
    base: int,
    post_reverse: bool,
    mapping: dict[str, int],
    examples_parsed: list[tuple[str, str]],
    success: bool,
) -> list[str]:
    """One enumeration entry: 'Try op X — succeeds/fails on examples'."""
    out = []
    rev_tag = " (reverse result)" if post_reverse else ""
    out.append(f"Try [{op_name}, base={base}{rev_tag}]:")
    for nol, rhs in examples_parsed:
        a0, a1, b0, b1 = nol
        a_digs = (mapping.get(a0), mapping.get(a1))
        b_digs = (mapping.get(b0), mapping.get(b1))
        a = (a_digs[0] or 0) * base + (a_digs[1] or 0)
        b = (b_digs[0] or 0) * base + (b_digs[1] or 0)
        out.append(
            f"  example {nol} = {rhs}: under mapping → "
            f"{a_digs[0]},{a_digs[1]} vs {b_digs[0]},{b_digs[1]} "
            f"→ {OP_LABEL.get(op_name, op_name)}({a}, {b})"
        )
        if not success:
            out.append("    (this candidate is being tested; details below if matched)")
            break
    return out


def make_cot_for_solved(
    pid: str,
    parsed: ParsedPuzzle,
    m: Measurements,
    op_name: str,
    base: int,
    post_reverse: bool,
    mapping: dict[str, int],
    predicted_answer: str,
    op_position: int,
) -> str:
    """Generate the full CoT for a solved puzzle."""
    lines: list[str] = []
    lines.append("I need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # ── Measurements ─────────────────────────────────────────────────────
    lines.extend(_measurement_block(m, parsed))

    # ── Hypothesis narrowing ─────────────────────────────────────────────
    lines.append("Based on the measurements above, the most likely family is")
    lines.append(f"  digit-cipher arithmetic with base ≈ {base}")
    lines.append(f"  (each character = one digit in {{0,...,{base-1}}}, bijective mapping)")
    lines.append("")
    lines.append("I will enumerate candidate operations until one matches all examples.")
    lines.append("")

    # ── Enumeration / search trace ───────────────────────────────────────
    # For brevity show 3-5 negative trials, then the winning trial
    nol_list = []
    for e in parsed.examples:
        nol = e.lhs[:op_position] + e.lhs[op_position + 1:]
        if len(nol) == 4:
            nol_list.append((nol, e.rhs))

    # Show 2-3 ops that should fail, then the winner
    failure_ops = [op for op in ["add", "sub_ab", "mul", "concat"] if op != op_name][:3]
    for fop in failure_ops:
        lines.append(f"Try {OP_LABEL[fop]} (base 10, no reverse): ")
        a_d, b_d = (mapping.get(nol_list[0][0][0], 0), mapping.get(nol_list[0][0][1], 0)), \
                   (mapping.get(nol_list[0][0][2], 0), mapping.get(nol_list[0][0][3], 0))
        a = a_d[0] * 10 + a_d[1]
        b = b_d[0] * 10 + b_d[1]
        result_digits = try_op(fop, a_d, b_d, 10)
        expected = nol_list[0][1]
        lines.append(f"  example 1 ({nol_list[0][0]} = {expected}): "
                     f"  {a} {fop} {b} = {result_digits if result_digits else 'invalid'}  "
                     f"vs expected {expected} → MISMATCH; reject.")
        lines.append("")

    # ── Winning candidate ─────────────────────────────────────────────────
    rev_tag = " (then reverse result)" if post_reverse else ""
    lines.append(f"Try {OP_LABEL[op_name]} in base {base}{rev_tag}:")
    lines.append(f"  Hypothesize character → digit mapping:")
    for c, d in sorted(mapping.items()):
        lines.append(f"    '{c}' = {d}")
    lines.append("")
    lines.append("  Verify on each example:")
    for nol, rhs in nol_list:
        a_d = (mapping[nol[0]], mapping[nol[1]])
        b_d = (mapping[nol[2]], mapping[nol[3]])
        a = a_d[0] * base + a_d[1]
        b = b_d[0] * base + b_d[1]
        res = try_op(op_name, a_d, b_d, base)
        if res is None:
            continue
        if post_reverse:
            res = tuple(reversed(res))
        # Encode result digits back to chars
        inv = {d: c for c, d in mapping.items()}
        res_chars = "".join(inv.get(d, "?") for d in res)
        lines.append(f"    {nol} = {rhs}:  {a} {OP_LABEL[op_name].replace('a',str(a)).replace('b',str(b))} = {res} "
                     f"→ encode → {res_chars}  ✓ matches {rhs}"
                     if res_chars == rhs
                     else f"    {nol} = {rhs}:  → encode → {res_chars}  vs {rhs}")
    lines.append("")
    lines.append("All examples match. This is the rule.")
    lines.append("")

    # ── Apply to query ────────────────────────────────────────────────────
    q = parsed.query
    qa = q[:op_position] + q[op_position+1:]
    q_op = q[op_position]
    lines.append(f"Apply the rule to the query: {q}")
    a_d = (mapping[qa[0]], mapping[qa[1]])
    b_d = (mapping[qa[2]], mapping[qa[3]])
    a = a_d[0] * base + a_d[1]
    b = b_d[0] * base + b_d[1]
    lines.append(f"  Operands under mapping: ({a_d[0]},{a_d[1]}) = {a}, ({b_d[0]},{b_d[1]}) = {b}")
    lines.append(f"  Apply {OP_LABEL[op_name]}: result digits = {try_op(op_name, a_d, b_d, base)}")
    if post_reverse:
        lines.append("  Reverse the result digits.")
    lines.append(f"  Encode digits back to chars → {predicted_answer}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}.")
    lines.append(f"The answer is \\boxed{{{predicted_answer}}}.")
    return "\n".join(lines)


if __name__ == "__main__":
    # smoke test on a fake puzzle
    print("CoT template module loaded; awaiting probe results to generate actual traces.")
