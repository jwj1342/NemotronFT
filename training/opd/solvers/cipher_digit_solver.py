"""
Cipher-Digit solver — efficient O(10000) approach.

Problem type: all characters (including digits 0–9) are encrypted by a
bijective cipher.  Input format:  XY<op>ZW = R  where:
  - X, Y  encode digits A, B  (the left 2-digit operand)
  - op    is the (encrypted) operator symbol — always at position 2
  - Z, W  encode digits C, D  (the right 2-digit operand)
  - R     is the cipher-encoded numeric result

Algorithm
---------
1. Parse: extract raw example pairs (input_str, output_str) and question string.
2. Operator symbol  = input_str[2]  (same for all examples with the same op).
3. For each combo (digit_ordering, operation, output_format) in COMBOS
   (frequency-ordered, imported from symbol_digit_solver):
   a. Skip operator-prefixed/suffixed formats — the operator is encrypted so
      the literal op_char is meaningless inside the output.
   b. For EX1: enumerate all (L, R) pairs where L ∈ [0,99], R ∈ [0,99]:
        A = L // 10,  B = L % 10,  C = R // 10,  D = R % 10
        expected_output = apply_combo(combo, A, B, C, D, op_sym)
        - Reject if output string length ≠ EX1 output cipher string length.
        - Build partial mapping:
            {sym_A: A, sym_B: B, sym_C: C, sym_D: D}
            ∪ {out_sym[i]: int(out_str_digit[i]) for i in range(len(out))}
        - Reject on any mapping conflict (two symbols → same digit or vice versa).
        - Save candidate (L, R, partial_mapping).
   c. For each EX1 candidate, verify on EX2:
        Decode EX2 input symbols via partial_mapping where possible.
        Enumerate free (L2, R2) consistent with partial mapping constraints.
        Compute expected output for EX2, extend mapping, check consistency.
        If verified, check EX3 if available (further pruning).
   d. If verified, decode target input, apply combo, encode result back to cipher.
4. Output digits must all appear in the mapping (seen in examples).

COMBOS are imported from symbol_digit_solver (same directory).

Public API
----------
    SolveResult  – dataclass (solved, answer, cot_trace)
    solve_cipher_digit(prompt: str) -> SolveResult
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import shared infrastructure
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from symbol_digit_solver import COMBOS, apply_combo  # type: ignore


# ---------------------------------------------------------------------------
# Data model (compatible with symbol_digit_solver.SolveResult interface)
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    solved: bool
    answer: Optional[str]
    cot_trace: str = ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_prompt(prompt: str):
    """
    Extract raw example pairs and the question input from a cipher-digit prompt.

    Lines of the form:   XXXXX = YYYY         → example
    Lines of the form:   determine the result for: XXXXX
                     or  result for: XXXXX     → question

    Returns:
        (raw_examples, question_raw, op_sym)
        raw_examples: list of (input_str, output_str)
        question_raw: str
        op_sym: str  (character at position 2 in all inputs)
    """
    raw_examples: list[tuple[str, str]] = []
    question_raw: Optional[str] = None

    for line in prompt.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Question line (several phrasings)
        if re.search(r'(?:determine|result for)', line, re.IGNORECASE):
            m = re.search(r':\s*(\S+)\s*$', line)
            if m:
                question_raw = m.group(1)
            continue
        # Example line: non-whitespace = non-whitespace
        m = re.match(r'^(\S+)\s*=\s*(\S+)$', line)
        if m:
            raw_examples.append((m.group(1), m.group(2)))

    if not raw_examples or question_raw is None:
        return None, None, None

    # Operator comes from the QUESTION (position 2), not from the first example.
    # Different examples in the same problem can use different operators, each with
    # its own rule. We must filter to examples that share the question's operator.
    if len(question_raw) < 5:
        return None, None, None
    op_sym = question_raw[2]

    # Keep only examples whose input has exactly 5 chars and operator matches question
    valid = [(inp, out) for inp, out in raw_examples
             if len(inp) == 5 and inp[2] == op_sym]

    if len(valid) < 2:
        return None, None, None

    return valid, question_raw, op_sym


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------



def _build_partial_mapping(
    inp: str, cipher_out: str, numeric_out: str,
    A: int, B: int, C: int, D: int,
) -> Optional[tuple[dict, dict]]:
    """
    Given:
        inp          = cipher input string (len 5, inp[2] = op_sym, skip)
        cipher_out   = cipher-encoded output string from the problem
        numeric_out  = the digit-string result from apply_combo
        A,B,C,D      = candidate digit values for inp[0],inp[1],inp[3],inp[4]

    Returns (mapping, rev_mapping) where mapping: sym→digit, rev_mapping: digit→sym
    or None if there is a contradiction.
    """
    if len(cipher_out) != len(numeric_out):
        return None

    mapping: dict[str, int] = {}
    rev: dict[int, str] = {}

    # Assign input symbols
    assignments = [
        (inp[0], A),
        (inp[1], B),
        (inp[3], C),
        (inp[4], D),
    ]
    for sym, dig in assignments:
        if sym in mapping:
            if mapping[sym] != dig:
                return None
        elif dig in rev:
            if rev[dig] != sym:
                return None
        else:
            mapping[sym] = dig
            rev[dig] = sym

    # Assign output symbols
    for csym, dchar in zip(cipher_out, numeric_out):
        dig = int(dchar)
        if csym in mapping:
            if mapping[csym] != dig:
                return None
        elif dig in rev:
            if rev[dig] != csym:
                return None
        else:
            mapping[csym] = dig
            rev[dig] = csym

    return mapping, rev


def _merge_mappings(
    base_mapping: dict, base_rev: dict,
    new_mapping: dict, new_rev: dict,
) -> Optional[tuple[dict, dict]]:
    """
    Try to merge new_mapping into base_mapping.
    Returns merged (mapping, rev) or None on conflict.
    """
    m = dict(base_mapping)
    r = dict(base_rev)
    for sym, dig in new_mapping.items():
        if sym in m:
            if m[sym] != dig:
                return None
        elif dig in r:
            if r[dig] != sym:
                return None
        else:
            m[sym] = dig
            r[dig] = sym
    return m, r


# ---------------------------------------------------------------------------
# CoT trace generator
# ---------------------------------------------------------------------------

def _build_cot_trace(
    raw_examples: list[tuple[str, str]],
    question_raw: str,
    op_sym: str,
    combo: tuple,
    mapping: dict,
    answer: str,
) -> str:
    """Build a structured CoT trace for a solved cipher-digit problem."""
    order, op_name, fmt_name = combo
    inv = {v: k for k, v in mapping.items()}

    lines: list[str] = []
    lines.append("I need to decode the cipher and determine the rule from examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    lines.append("S1: IDENTIFY STRUCTURE")
    lines.append(f"  Operator cipher symbol: '{op_sym}' (always at position 2 in input)")
    lines.append(f"  Input format: AB{op_sym}CD where A,B,C,D are encrypted digits")
    lines.append(f"  Output is the cipher-encoded numeric result")
    lines.append("")

    lines.append("S2: CIPHER MAPPING (inferred)")
    for sym in sorted(mapping.keys()):
        lines.append(f"  '{sym}' -> {mapping[sym]}")
    lines.append("")

    lines.append("S3: RULE (combo found)")
    lines.append(f"  digit_order={order}  operation={op_name}  output_format={fmt_name}")
    lines.append("")

    lines.append("S4: VERIFY ON EXAMPLES")
    for inp, cipher_out in raw_examples[:4]:
        if len(inp) != 5 or inp[2] != op_sym:
            continue
        sA, sB, sC, sD = inp[0], inp[1], inp[3], inp[4]
        if any(s not in mapping for s in [sA, sB, sC, sD]):
            lines.append(f"  {inp} = {cipher_out}  (incomplete mapping, skip)")
            continue
        A, B, C, D = mapping[sA], mapping[sB], mapping[sC], mapping[sD]
        numeric_out = apply_combo(combo, A, B, C, D, op_sym)
        # Re-encode to cipher
        if numeric_out is None:
            lines.append(f"  {inp} = {cipher_out}  (apply_combo returned None)")
            continue
        cipher_predicted = ""
        ok = True
        for dchar in numeric_out:
            d = int(dchar)
            if d not in inv:
                ok = False
                break
            cipher_predicted += inv[d]
        match = "YES" if (ok and cipher_predicted == cipher_out) else "NO"
        lines.append(f"  {inp} = {cipher_out}:  "
                     f"A={A},B={B},C={C},D={D}  {order}|{op_name}|{fmt_name}"
                     f" -> {numeric_out} -> cipher '{cipher_predicted}'  {match}")
    lines.append("")

    lines.append("S5: APPLY TO TARGET")
    if len(question_raw) == 5 and question_raw[2] == op_sym:
        sA, sB, sC, sD = question_raw[0], question_raw[1], question_raw[3], question_raw[4]
        qA = mapping.get(sA, "?")
        qB = mapping.get(sB, "?")
        qC = mapping.get(sC, "?")
        qD = mapping.get(sD, "?")
        if "?" not in [str(x) for x in [qA, qB, qC, qD]]:
            numeric_ans = apply_combo(combo, qA, qB, qC, qD, op_sym)
            lines.append(f"  Target: {question_raw}  A={qA},B={qB},C={qC},D={qD}")
            lines.append(f"  {order}|{op_name}|{fmt_name} -> {numeric_ans} -> cipher '{answer}'")
        else:
            lines.append(f"  Target: {question_raw}  (partial mapping)")
    lines.append("")
    lines.append(f"S6: ANS={answer}")
    lines.append("")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_cipher_digit(prompt: str) -> SolveResult:
    """
    Solve a cipher-digit problem.

    Algorithm:
    1. Parse prompt → raw_examples, question_raw, op_sym
    2. For each combo in COMBOS (skip op-prefixed formats):
       a. Enumerate all (L, R) pairs for EX1 → build partial mapping → collect candidates
       b. For each EX1 candidate, verify on EX2 (extend mapping)
       c. If consistent, optionally verify on EX3
       d. Decode target, apply combo, encode result to cipher
    """
    raw_examples, question_raw, op_sym = _parse_prompt(prompt)
    if raw_examples is None:
        return SolveResult(solved=False, answer=None, cot_trace="(parse error)")

    ex1_inp, ex1_out = raw_examples[0]
    ex2_inp, ex2_out = raw_examples[1]
    ex3 = raw_examples[2] if len(raw_examples) > 2 else None

    ex1_out_len = len(ex1_out)
    ex2_out_len = len(ex2_out)

    for combo in COMBOS:
        order, op_name, fmt_name = combo

        # Skip operator-prefixed/suffixed formats — operator symbol is encrypted,
        # so embedding it in the output string is meaningless for cipher problems.
        if "op_" in fmt_name:
            continue

        # ----------------------------------------------------------------
        # EX1 candidate generation: enumerate all (L, R) in 0..99
        # ----------------------------------------------------------------
        ex1_candidates: list[tuple[int, int, dict, dict]] = []  # (L, R, mapping, rev)

        for L in range(100):
            A = L // 10
            B = L % 10
            for R in range(100):
                C = R // 10
                D = R % 10

                numeric_out = apply_combo(combo, A, B, C, D, op_sym)
                if numeric_out is None:
                    continue
                # Skip negative/non-digit results — cipher output is always all-digits
                if not numeric_out.isdigit():
                    continue
                if len(numeric_out) != ex1_out_len:
                    continue

                # Build partial mapping from this candidate
                result = _build_partial_mapping(ex1_inp, ex1_out, numeric_out, A, B, C, D)
                if result is None:
                    continue
                m, r = result
                ex1_candidates.append((L, R, m, r))

        if not ex1_candidates:
            continue

        # ----------------------------------------------------------------
        # EX2 verification — break as soon as we find the first match
        # ----------------------------------------------------------------
        verified_candidates: list[tuple[dict, dict]] = []

        for L1, R1, m1, r1 in ex1_candidates:
            # Extract EX2 input symbols
            sA2 = ex2_inp[0]; sB2 = ex2_inp[1]; sC2 = ex2_inp[3]; sD2 = ex2_inp[4]

            # Determine digit ranges for EX2 operands, constrained by mapping
            A2_range = [m1[sA2]] if sA2 in m1 else list(range(10))
            B2_range = [m1[sB2]] if sB2 in m1 else list(range(10))
            C2_range = [m1[sC2]] if sC2 in m1 else list(range(10))
            D2_range = [m1[sD2]] if sD2 in m1 else list(range(10))

            for A2 in A2_range:
                for B2 in B2_range:
                    for C2 in C2_range:
                        for D2 in D2_range:
                            numeric_out2 = apply_combo(combo, A2, B2, C2, D2, op_sym)
                            if numeric_out2 is None or not numeric_out2.isdigit():
                                continue
                            if len(numeric_out2) != ex2_out_len:
                                continue
                            result2 = _build_partial_mapping(
                                ex2_inp, ex2_out, numeric_out2, A2, B2, C2, D2
                            )
                            if result2 is None:
                                continue
                            merged = _merge_mappings(m1, r1, *result2)
                            if merged is None:
                                continue
                            verified_candidates.append(merged)
                            # Early exit: first verified match is enough
                            break
                        if verified_candidates:
                            break
                    if verified_candidates:
                        break
                if verified_candidates:
                    break
            if verified_candidates:
                break  # stop scanning ex1_candidates for this combo

        if not verified_candidates:
            continue

        # ----------------------------------------------------------------
        # EX3 pruning (optional — further filters)
        # ----------------------------------------------------------------
        if ex3 is not None:
            ex3_inp, ex3_out = ex3
            ex3_out_len = len(ex3_out)
            pruned: list[tuple[dict, dict]] = []

            for m_cand, r_cand in verified_candidates:
                sA3 = ex3_inp[0]; sB3 = ex3_inp[1]
                sC3 = ex3_inp[3]; sD3 = ex3_inp[4]

                A3_range = [m_cand[sA3]] if sA3 in m_cand else list(range(10))
                B3_range = [m_cand[sB3]] if sB3 in m_cand else list(range(10))
                C3_range = [m_cand[sC3]] if sC3 in m_cand else list(range(10))
                D3_range = [m_cand[sD3]] if sD3 in m_cand else list(range(10))

                for A3 in A3_range:
                    for B3 in B3_range:
                        for C3 in C3_range:
                            for D3 in D3_range:
                                numeric_out3 = apply_combo(combo, A3, B3, C3, D3, op_sym)
                                if numeric_out3 is None or not numeric_out3.isdigit():
                                    continue
                                if len(numeric_out3) != ex3_out_len:
                                    continue
                                result3 = _build_partial_mapping(
                                    ex3_inp, ex3_out, numeric_out3, A3, B3, C3, D3
                                )
                                if result3 is None:
                                    continue
                                m3, r3 = result3
                                merged3 = _merge_mappings(m_cand, r_cand, m3, r3)
                                if merged3 is None:
                                    continue
                                pruned.append(merged3)
                                break   # found a consistent A3 for this candidate
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

            if pruned:
                verified_candidates = pruned

        if not verified_candidates:
            continue

        # ----------------------------------------------------------------
        # Apply to target using the first (best) verified candidate
        # ----------------------------------------------------------------
        # Prefer the candidate with the most assignments (most constrained)
        verified_candidates.sort(key=lambda mc: -len(mc[0]))
        m_final, r_final = verified_candidates[0]

        if len(question_raw) != 5 or question_raw[2] != op_sym:
            continue

        sQA = question_raw[0]; sQB = question_raw[1]
        sQC = question_raw[3]; sQD = question_raw[4]

        if any(s not in m_final for s in [sQA, sQB, sQC, sQD]):
            continue

        qA = m_final[sQA]; qB = m_final[sQB]
        qC = m_final[sQC]; qD = m_final[sQD]

        numeric_ans = apply_combo(combo, qA, qB, qC, qD, op_sym)
        if numeric_ans is None or not numeric_ans.isdigit():
            continue

        # Re-encode answer back to cipher symbols
        inv_final = {v: k for k, v in m_final.items()}
        cipher_ans = ""
        ok = True
        for dchar in numeric_ans:
            d = int(dchar)
            if d not in inv_final:
                ok = False
                break
            cipher_ans += inv_final[d]
        if not ok:
            continue

        cot = _build_cot_trace(raw_examples, question_raw, op_sym,
                               combo, m_final, cipher_ans)
        return SolveResult(solved=True, answer=cipher_ans, cot_trace=cot)

    return SolveResult(solved=False, answer=None, cot_trace="(no combo matched)")


# ---------------------------------------------------------------------------
# CLI smoke test (does NOT run heavy computation on login node)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Would test here, run from parent")
    print("(cipher_digit_solver.py: writes SolveResult with .solved, .answer, .cot_trace)")
    print("")
    print("Public API:")
    print("  from cipher_digit_solver import SolveResult, solve_cipher_digit")
    print("  result = solve_cipher_digit(prompt_string)")
    print("  if result.solved:")
    print("      print(result.answer)    # cipher-encoded result string")
    print("      print(result.cot_trace)")
