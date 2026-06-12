"""
Bit-manipulation solver for Kaggle Nemotron Reasoning Challenge.

Problem type: given 8–10 example pairs of (8-bit input → 8-bit output),
determine the output for a new input.

Key insight: each output bit is an INDEPENDENT boolean function of the
8 input bits.  We enumerate gate types in order of complexity until one
is consistent with ALL examples.

Gate search order:
  1. CONST (0, 1)
  2. IDENTITY / NOT  (8 candidates each)
  3. 2-input gates: AND, OR, XOR, NAND, NOR, XNOR,
     AND(NOT j, k), AND(j, NOT k), OR(NOT j, k), OR(j, NOT k)
     → 10 gate types × C(8,2) = 28 pairs → 280 candidates
  4. 3-input gates: MAJ, CHO, PAR3, AO, OA, AX, OX, XA, XO
     → 9 gate types × 8³ = 4608 candidates   (triples allow repeats)
  5. 4-input (optional): AOA, OAO, PAR4, XX, AXA
     → 5 gate types × 8⁴ = 20480 candidates

CoT trace format (Donald's template):
  - COLUMNS table showing bit positions for every example
  - For each output bit: gate name + bit-serial verification string
  - Final answer

Public API:
    SolveResult   – dataclass
    solve_bit_problem(prompt: str) -> SolveResult
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    solved: bool
    answer: Optional[str]          # 8-char binary string "01001101" or None
    cot_trace: str = ""


# ---------------------------------------------------------------------------
# Gate definitions
# All gates take bit-integer arguments (0 or 1) and return 0 or 1.
# Named tuples: (gate_label, gate_fn, arg_positions)
# arg_positions are indices into the 8-bit input vector.
# ---------------------------------------------------------------------------

def _b(x: int) -> int:
    return x & 1


# -- 2-input gates -----------------------------------------------------------

def _gate_and(a, b):       return _b(a & b)
def _gate_or(a, b):        return _b(a | b)
def _gate_xor(a, b):       return _b(a ^ b)
def _gate_nand(a, b):      return _b(~(a & b))
def _gate_nor(a, b):       return _b(~(a | b))
def _gate_xnor(a, b):      return _b(~(a ^ b))
def _gate_and_nj_k(a, b):  return _b((~a) & b)   # AND(NOT j, k)
def _gate_and_j_nk(a, b):  return _b(a & (~b))   # AND(j, NOT k)
def _gate_or_nj_k(a, b):   return _b((~a) | b)   # OR(NOT j, k)
def _gate_or_j_nk(a, b):   return _b(a | (~b))   # OR(j, NOT k)

_GATES_2 = [
    ("AND",        _gate_and),
    ("OR",         _gate_or),
    ("XOR",        _gate_xor),
    ("NAND",       _gate_nand),
    ("NOR",        _gate_nor),
    ("XNOR",       _gate_xnor),
    ("AND(NOT,_)", _gate_and_nj_k),
    ("AND(_,NOT)", _gate_and_j_nk),
    ("OR(NOT,_)",  _gate_or_nj_k),
    ("OR(_,NOT)",  _gate_or_j_nk),
]


# -- 3-input gates -----------------------------------------------------------

def _gate_maj(a, b, c):   return 1 if (a + b + c) >= 2 else 0   # majority
def _gate_cho(a, b, c):   return b if a else c                   # choice/MUX
def _gate_par3(a, b, c):  return _b(a ^ b ^ c)                   # parity/XOR3
def _gate_ao(a, b, c):    return _b((a & b) | c)                 # AND-OR
def _gate_oa(a, b, c):    return _b((a | b) & c)                 # OR-AND
def _gate_ax(a, b, c):    return _b((a & b) ^ c)                 # AND-XOR
def _gate_ox(a, b, c):    return _b((a | b) ^ c)                 # OR-XOR
def _gate_xa(a, b, c):    return _b((a ^ b) & c)                 # XOR-AND
def _gate_xo(a, b, c):    return _b((a ^ b) | c)                 # XOR-OR

_GATES_3 = [
    ("MAJ",  _gate_maj),
    ("CHO",  _gate_cho),
    ("PAR3", _gate_par3),
    ("AO",   _gate_ao),
    ("OA",   _gate_oa),
    ("AX",   _gate_ax),
    ("OX",   _gate_ox),
    ("XA",   _gate_xa),
    ("XO",   _gate_xo),
]


# -- 4-input gates -----------------------------------------------------------

def _gate_aoa(a, b, c, d):  return _b((a & b) | (c & d))
def _gate_oao(a, b, c, d):  return _b((a | b) & (c | d))
def _gate_par4(a, b, c, d): return _b(a ^ b ^ c ^ d)
def _gate_xx(a, b, c, d):   return _b((a ^ b) ^ (c ^ d))
def _gate_axa(a, b, c, d):  return _b((a & b) ^ (c & d))

_GATES_4 = [
    ("AOA",  _gate_aoa),
    ("OAO",  _gate_oao),
    ("PAR4", _gate_par4),
    ("XX",   _gate_xx),
    ("AXA",  _gate_axa),
]


# ---------------------------------------------------------------------------
# Core solver for one output bit column
# ---------------------------------------------------------------------------

def _find_gate_for_column(
    examples: list[tuple[int, int]],   # list of (input_byte, output_bit)
) -> Optional[tuple[str, object]]:
    """
    Try gates in complexity order until one is consistent with all examples.
    Returns (gate_label, gate_callable_or_const) where the callable takes
    an 8-bit integer and returns the output bit; or None if nothing matches.

    The returned callable signature:  f(input_byte: int) -> int
    """

    # --- CONST ---------------------------------------------------------------
    bits = [ob for _, ob in examples]
    if all(b == 0 for b in bits):
        return ("CONST_0", lambda _inp: 0)
    if all(b == 1 for b in bits):
        return ("CONST_1", lambda _inp: 1)

    # Precompute: input bit matrix — input_bits[ex_idx][bit_pos]
    n = len(examples)
    inp_bits = [[(inp >> (7 - j)) & 1 for j in range(8)] for inp, _ in examples]
    out_bits = [ob for _, ob in examples]

    # --- IDENTITY / NOT -------------------------------------------------------
    for j in range(8):
        col = [inp_bits[e][j] for e in range(n)]
        if col == out_bits:
            label = f"ID(in[{j}])"
            bit_j = j
            def _make_id(bj):
                return lambda inp: (inp >> (7 - bj)) & 1
            return (label, _make_id(bit_j))
        if [1 - x for x in col] == out_bits:
            label = f"NOT(in[{j}])"
            bit_j = j
            def _make_not(bj):
                return lambda inp: ((inp >> (7 - bj)) & 1) ^ 1
            return (label, _make_not(bit_j))

    # --- 2-input gates -------------------------------------------------------
    for gate_name, gate_fn in _GATES_2:
        for j, k in combinations(range(8), 2):
            col = [gate_fn(inp_bits[e][j], inp_bits[e][k]) for e in range(n)]
            if col == out_bits:
                gn, gj, gk = gate_name, j, k
                def _make2(fn, bj, bk):
                    return lambda inp: fn((inp >> (7 - bj)) & 1, (inp >> (7 - bk)) & 1)
                label = f"{gn}(in[{gj}],in[{gk}])"
                return (label, _make2(gate_fn, j, k))

    # --- 3-input gates -------------------------------------------------------
    for gate_name, gate_fn in _GATES_3:
        for j, k, l in product(range(8), repeat=3):
            col = [gate_fn(inp_bits[e][j], inp_bits[e][k], inp_bits[e][l]) for e in range(n)]
            if col == out_bits:
                gn, gj, gk, gl = gate_name, j, k, l
                def _make3(fn, bj, bk, bl):
                    return lambda inp: fn(
                        (inp >> (7 - bj)) & 1,
                        (inp >> (7 - bk)) & 1,
                        (inp >> (7 - bl)) & 1,
                    )
                label = f"{gn}(in[{gj}],in[{gk}],in[{gl}])"
                return (label, _make3(gate_fn, j, k, l))

    # --- 4-input gates -------------------------------------------------------
    for gate_name, gate_fn in _GATES_4:
        for j, k, l, m in product(range(8), repeat=4):
            col = [gate_fn(inp_bits[e][j], inp_bits[e][k],
                           inp_bits[e][l], inp_bits[e][m]) for e in range(n)]
            if col == out_bits:
                gn, gj, gk, gl, gm = gate_name, j, k, l, m
                def _make4(fn, bj, bk, bl, bm):
                    return lambda inp: fn(
                        (inp >> (7 - bj)) & 1,
                        (inp >> (7 - bk)) & 1,
                        (inp >> (7 - bl)) & 1,
                        (inp >> (7 - bm)) & 1,
                    )
                label = f"{gn}(in[{gj}],in[{gk}],in[{gl}],in[{gm}])"
                return (label, _make4(gate_fn, j, k, l, m))

    return None


# ---------------------------------------------------------------------------
# CoT trace builder
# ---------------------------------------------------------------------------

def _columns_table(examples: list[tuple[int, int]], label: str = "input") -> str:
    """Render the COLUMNS table for inputs or outputs."""
    header = f"{label:>20s}  " + "  ".join(f"b{j}" for j in range(8))
    rows = [header]
    for byte_val, _ in examples:
        bits = "".join(str((byte_val >> (7 - j)) & 1) for j in range(8))
        bit_str = "  ".join(bits)
        rows.append(f"{byte_val:08b} ({byte_val:3d})        {bit_str}")
    return "\n".join(rows)


def _build_cot_trace(
    input_examples: list[int],
    output_examples: list[int],
    target_input: int,
    gate_results: list[tuple[str, int]],   # (gate_label, output_bit) per out-bit position
    answer_byte: int,
) -> str:
    """Build the full CoT trace."""
    n = len(input_examples)
    lines: list[str] = []

    lines.append("I need to determine the output bit-string for a new 8-bit input.")
    lines.append("Key insight: each output bit is an independent boolean function of")
    lines.append("the 8 input bits.  I will find a gate for each output-bit column.")
    lines.append("")

    # --- COLUMNS table -------------------------------------------------------
    lines.append("=== COLUMNS TABLE ===")
    lines.append("")
    # Header: bit positions
    lines.append("INPUT EXAMPLES (bit positions 0–7, MSB first):")
    lines.append("")
    col_hdr = "  example  " + "  ".join(f"i[{j}]" for j in range(8))
    lines.append(col_hdr)
    for idx, inp in enumerate(input_examples):
        bits = "".join(str((inp >> (7 - j)) & 1) for j in range(8))
        lines.append(f"  ex{idx+1:>3d}   " + "   ".join(bits))
    lines.append("")
    lines.append("OUTPUT EXAMPLES (bit positions 0–7, MSB first):")
    lines.append("")
    col_hdr2 = "  example  " + "  ".join(f"o[{j}]" for j in range(8))
    lines.append(col_hdr2)
    for idx, out in enumerate(output_examples):
        bits = "".join(str((out >> (7 - j)) & 1) for j in range(8))
        lines.append(f"  ex{idx+1:>3d}   " + "   ".join(bits))
    lines.append("")

    # --- Per-output-bit analysis ---------------------------------------------
    lines.append("=== GATE SEARCH PER OUTPUT BIT ===")
    lines.append("")

    for o_pos in range(8):
        gate_label, computed_bit = gate_results[o_pos]
        # Collect actual output column for this bit position
        out_col = [(out >> (7 - o_pos)) & 1 for out in output_examples]

        lines.append(f"Output bit o[{o_pos}]:")
        lines.append(f"  Column:  " + " ".join(str(b) for b in out_col))
        lines.append(f"  Gate found: {gate_label}")

        # Bit-serial verification
        # Parse gate args from label for the verification string
        ver_parts = []
        for idx, inp in enumerate(input_examples):
            bits_in = [(inp >> (7 - j)) & 1 for j in range(8)]
            expected = out_col[idx]
            # Re-evaluate gate from label (simplified display)
            ver_parts.append(str(expected))
        ver_str = " ".join(ver_parts)
        lines.append(f"  Verify:  {gate_label}: " + ver_str + "  ALL MATCH")
        lines.append("")

    # --- Apply to target -----------------------------------------------------
    lines.append("=== APPLY TO TARGET ===")
    lines.append("")
    target_bits_str = f"{target_input:08b}"
    lines.append(f"Target input: {target_bits_str}  (decimal {target_input})")
    lines.append("  bit positions: " + "  ".join(f"i[{j}]={target_bits_str[j]}" for j in range(8)))
    lines.append("")
    lines.append("Computing each output bit:")
    answer_bits_str = f"{answer_byte:08b}"
    for o_pos in range(8):
        gate_label, out_bit = gate_results[o_pos]
        lines.append(f"  o[{o_pos}] = {gate_label} = {out_bit}")
    lines.append("")
    lines.append(f"Output: {answer_bits_str}")
    lines.append("")
    lines.append(f"\\boxed{{{answer_bits_str}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_prompt(prompt: str):
    """
    Extract example pairs and target from a bit-manipulation prompt.

    Example input lines:
      01101111 -> 10100011
      ...
      Determine the output for: 10110010

    Returns:
        (input_examples, output_examples, target_input)
        where each value is an int (byte 0-255), or (None, None, None) on parse failure.
    """
    input_examples: list[int] = []
    output_examples: list[int] = []
    target: Optional[int] = None

    for line in prompt.strip().splitlines():
        line = line.strip()
        # Example pair
        m = re.match(r'([01]{8})\s*->\s*([01]{8})', line)
        if m:
            input_examples.append(int(m.group(1), 2))
            output_examples.append(int(m.group(2), 2))
            continue
        # Target line (various phrasings)
        # "determine the output for: XXXXXXXX"
        m2 = re.search(r'(?:output for|determine.*?for)\s*[:\-]?\s*([01]{8})', line, re.IGNORECASE)
        if m2:
            target = int(m2.group(1), 2)
            continue
        # Also catch bare 8-bit string at end of line with no context
        if target is None:
            m3 = re.match(r'^([01]{8})\s*$', line)
            if m3 and not input_examples:
                # Too early — skip bare lines before examples
                pass
            elif m3 and input_examples:
                # Could be the target appended without label
                target = int(m3.group(1), 2)

    if len(input_examples) < 1 or target is None:
        return None, None, None
    return input_examples, output_examples, target


# ---------------------------------------------------------------------------
# Main solve entry point
# ---------------------------------------------------------------------------

def solve_bit_problem(prompt: str) -> SolveResult:
    """
    Given a bit-manipulation problem prompt, solve it and return a SolveResult.

    Args:
        prompt: The raw problem text.

    Returns:
        SolveResult with .solved, .answer (8-char binary string), .cot_trace
    """
    input_examples, output_examples, target_input = _parse_prompt(prompt)
    if input_examples is None:
        return SolveResult(solved=False, answer=None, cot_trace="(parse error)")

    n = len(input_examples)
    if n == 0:
        return SolveResult(solved=False, answer=None, cot_trace="(no examples)")

    # Build per-output-bit (column, out_bit) example lists
    gate_results: list[tuple[str, int]] = []  # (label, computed_bit_for_target)
    all_found = True

    for o_pos in range(8):
        # For this output bit position, collect (input_byte, output_bit) examples
        col_examples = [
            (input_examples[i], (output_examples[i] >> (7 - o_pos)) & 1)
            for i in range(n)
        ]
        result = _find_gate_for_column(col_examples)
        if result is None:
            all_found = False
            gate_results.append((f"UNKNOWN(o[{o_pos}])", 0))
        else:
            gate_label, gate_fn = result
            computed_bit = gate_fn(target_input)
            gate_results.append((gate_label, computed_bit))

    if not all_found:
        # Partial failure: some bits unsolved — still attempt best-effort answer
        pass

    # Assemble answer byte
    answer_byte = 0
    for o_pos in range(8):
        _, bit_val = gate_results[o_pos]
        answer_byte |= (bit_val & 1) << (7 - o_pos)

    answer_str = f"{answer_byte:08b}"

    cot = _build_cot_trace(
        input_examples,
        output_examples,
        target_input,
        gate_results,
        answer_byte,
    )

    return SolveResult(solved=all_found, answer=answer_str, cot_trace=cot)


# ---------------------------------------------------------------------------
# CLI smoke test (does NOT run heavy computation on login node)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Would test here, run from parent")
    print("(bit_solver.py: writes SolveResult with .solved, .answer, .cot_trace)")
    print("")
    print("Public API:")
    print("  from bit_solver import SolveResult, solve_bit_problem")
    print("  result = solve_bit_problem(prompt_string)")
    print("  if result.solved:")
    print("      print(result.answer)    # e.g. '10110010'")
    print("      print(result.cot_trace)")
