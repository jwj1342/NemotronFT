"""
Symbol-Digit and Cipher-Digit solver using frequency-ordered brute-force combo scan.

Based on methodology from Kaggle discussion #684283:
- Input: AB<op>CD where A,B,C,D are single digits, op is an arbitrary symbol
- 4 digit orderings × 14 operations × 14 output formats = 784 possible combos
- Try combos in frequency order; verify first match against EX2
- For cipher-digit: crack symbol→digit mapping first, then treat as symbol-digit

The 47 highest-frequency combos cover ~99% of the training distribution.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Digit ordering helpers
# ---------------------------------------------------------------------------

def _lrpairs(A: int, B: int, C: int, D: int):
    """All 4 digit orderings as (name, L, R)."""
    return [
        ("AB_CD", A * 10 + B, C * 10 + D),
        ("BA_DC", B * 10 + A, D * 10 + C),
        ("AB_DC", A * 10 + B, D * 10 + C),
        ("BA_CD", B * 10 + A, C * 10 + D),
    ]


def _rev_str(n: int) -> Optional[str]:
    """Reverse the digit string of n (may have leading zeros)."""
    s = str(abs(n))
    return s[::-1]


def _dsum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


# ---------------------------------------------------------------------------
# Operation table
# ---------------------------------------------------------------------------

def _apply_op(op_name: str, L: int, R: int) -> Optional[int]:
    if op_name == "add":
        return L + R
    elif op_name == "mul":
        return L * R
    elif op_name == "sub_lr":
        return L - R
    elif op_name == "sub_rl":
        return R - L
    elif op_name == "abs_sub":
        return abs(L - R)
    elif op_name == "cat_lr":
        return int(f"{L:02d}{R:02d}") if L < 100 and R < 100 else int(f"{L}{R}")
    elif op_name == "cat_rl":
        return int(f"{R:02d}{L:02d}") if L < 100 and R < 100 else int(f"{R}{L}")
    elif op_name == "addm1":    # add - 1
        return L + R - 1
    elif op_name == "add1":     # add + 1
        return L + R + 1
    elif op_name == "mulsub1":  # mul - 1
        return L * R - 1
    elif op_name == "muladd1":  # mul + 1
        return L * R + 1
    elif op_name == "dsum_add":
        return _dsum(L) + _dsum(R)
    elif op_name == "dsum_mul":
        return _dsum(L) * _dsum(R)
    return None


# ---------------------------------------------------------------------------
# Output format table
# ---------------------------------------------------------------------------

def _apply_fmt(fmt_name: str, raw_int: int, op_char: str = "") -> Optional[str]:
    """Convert raw integer result to output string per format."""
    if fmt_name == "raw":
        return str(raw_int)
    elif fmt_name == "raw_neg":
        return str(-raw_int)
    elif fmt_name == "rev":
        return _rev_str(raw_int)
    elif fmt_name == "rev_neg":
        s = _rev_str(raw_int)
        return str(-int(s)) if s else None
    elif fmt_name == "abs":
        return str(abs(raw_int))
    elif fmt_name == "zpad2":
        return f"{abs(raw_int):02d}"
    elif fmt_name == "dsum":
        return str(_dsum(raw_int))
    elif fmt_name == "op_pre":
        return op_char + str(raw_int)
    elif fmt_name == "op_suf":
        return str(raw_int) + op_char
    elif fmt_name == "op_pre_rev":
        return op_char + _rev_str(raw_int)
    elif fmt_name == "op_suf_rev":
        return _rev_str(raw_int) + op_char
    elif fmt_name == "op_pre_abs":
        return op_char + str(abs(raw_int))
    elif fmt_name == "op_suf_abs":
        return str(abs(raw_int)) + op_char
    return None


# ---------------------------------------------------------------------------
# Frequency-ordered combo list (47 high-frequency combos first)
# ---------------------------------------------------------------------------
# Format: (digit_order, op_name, fmt_name)
# Ordering approximates training distribution frequency

COMBOS = [
    # Top-frequency (from post: BA_DC|add|rev ~13%, BA_DC|mul|rev ~12%)
    ("BA_DC", "add",     "rev"),
    ("BA_DC", "mul",     "rev"),
    ("AB_CD", "cat_lr",  "raw"),
    ("BA_DC", "cat_lr",  "rev"),
    ("AB_CD", "sub_lr",  "raw"),
    ("BA_DC", "mulsub1", "rev"),
    ("AB_CD", "mul",     "raw"),
    ("AB_CD", "add",     "raw"),
    ("BA_DC", "muladd1", "rev"),
    ("AB_CD", "muladd1", "raw"),
    ("BA_DC", "addm1",   "rev"),
    ("BA_DC", "sub_lr",  "rev"),
    ("AB_CD", "cat_rl",  "raw"),
    ("BA_DC", "cat_rl",  "rev"),
    ("AB_CD", "abs_sub", "raw"),
    ("BA_DC", "abs_sub", "rev"),
    ("AB_CD", "add1",    "raw"),
    ("BA_DC", "add1",    "rev"),
    ("AB_CD", "mulsub1", "raw"),
    ("AB_CD", "addm1",   "raw"),
    # AB orderings with rev
    ("AB_CD", "add",     "rev"),
    ("AB_CD", "mul",     "rev"),
    ("AB_CD", "sub_lr",  "rev"),
    ("AB_CD", "abs_sub", "rev"),
    ("AB_CD", "mulsub1", "rev"),
    ("AB_CD", "muladd1", "rev"),
    ("AB_CD", "addm1",   "rev"),
    ("AB_CD", "add1",    "rev"),
    # BA_DC raw variants
    ("BA_DC", "add",     "raw"),
    ("BA_DC", "mul",     "raw"),
    ("BA_DC", "sub_lr",  "raw"),
    ("BA_DC", "abs_sub", "raw"),
    ("BA_DC", "cat_lr",  "raw"),
    ("BA_DC", "addm1",   "raw"),
    ("BA_DC", "add1",    "raw"),
    # Cross orderings
    ("AB_DC", "add",     "raw"),
    ("AB_DC", "mul",     "raw"),
    ("AB_DC", "add",     "rev"),
    ("AB_DC", "mul",     "rev"),
    ("BA_CD", "add",     "raw"),
    ("BA_CD", "mul",     "raw"),
    ("BA_CD", "add",     "rev"),
    ("BA_CD", "mul",     "rev"),
    # digit-sum ops
    ("AB_CD", "dsum_add","raw"),
    ("BA_DC", "dsum_add","raw"),
    ("AB_CD", "dsum_mul","raw"),
    # sub_lr with rev (confirmed: BA_DC|sub_lr|rev works for many `-` problems)
    ("AB_CD", "sub_lr",  "rev"),
    ("BA_DC", "sub_lr",  "rev"),
    # sub_rl
    ("AB_CD", "sub_rl",  "raw"),
    ("BA_DC", "sub_rl",  "raw"),
    ("AB_CD", "sub_rl",  "rev"),
    ("BA_DC", "sub_rl",  "rev"),
    # abs_sub with BA_DC
    ("BA_DC", "abs_sub", "raw"),
    ("BA_DC", "abs_sub", "rev"),
    # addm1/add1 with BA_DC
    ("BA_DC", "addm1",   "raw"),
    ("BA_DC", "add1",    "raw"),
    # operator-prefixed/suffixed formats
    ("AB_CD", "add",     "op_pre"),
    ("AB_CD", "mul",     "op_pre"),
    ("AB_CD", "add",     "op_suf"),
    ("AB_CD", "mul",     "op_suf"),
    ("BA_DC", "add",     "op_pre"),
    ("BA_DC", "mul",     "op_pre"),
    ("BA_DC", "add",     "op_suf"),
    ("BA_DC", "mul",     "op_suf"),
    ("AB_CD", "cat_lr",  "op_pre"),
    ("AB_CD", "cat_lr",  "op_suf"),
    ("AB_CD", "sub_lr",  "op_pre"),
    ("AB_CD", "sub_lr",  "op_suf"),
    ("AB_CD", "abs_sub", "op_pre"),
    ("AB_CD", "abs_sub", "op_suf"),
    # abs format
    ("AB_CD", "sub_lr",  "abs"),
    ("BA_DC", "sub_lr",  "abs"),
    ("AB_CD", "sub_rl",  "abs"),
    ("BA_DC", "sub_rl",  "abs"),
    # zpad2
    ("AB_CD", "dsum_add","zpad2"),
    ("BA_DC", "dsum_add","zpad2"),
    ("AB_CD", "abs_sub", "zpad2"),
]


# ---------------------------------------------------------------------------
# Parse a numeric equation line: "AB<op>CD = result"
# ---------------------------------------------------------------------------

def parse_eq_line(line: str):
    """
    Parse "AB<op>CD = result" where AB, CD are 2-digit numbers, op is any char.
    Returns (A, B, op_char, C, D, result_str) or None.
    """
    line = line.strip()
    m = re.match(r'^(\d)(\d)(.)(.\d)(\d)\s*=\s*(.+)$', line)
    if m:
        A, B, op, C, D, res = m.groups()
        return int(A), int(B), op, int(C), int(D), res.strip()
    # Also handle: <two-digit><op><two-digit> = <result>
    m = re.match(r'^(\d{2})([^\d\s])(\d{2})\s*=\s*(.+)$', line)
    if m:
        lhs1, op, lhs2, res = m.groups()
        A, B = int(lhs1[0]), int(lhs1[1])
        C, D = int(lhs2[0]), int(lhs2[1])
        return A, B, op, C, D, res.strip()
    return None


def parse_question_line(line: str):
    """Parse "AB<op>CD" (no = part). Returns (A, B, op_char, C, D) or None."""
    line = line.strip()
    m = re.match(r'^(\d{2})([^\d\s])(\d{2})$', line)
    if m:
        lhs1, op, lhs2 = m.groups()
        A, B = int(lhs1[0]), int(lhs1[1])
        C, D = int(lhs2[0]), int(lhs2[1])
        return A, B, op, C, D
    return None


def apply_combo(combo_name: tuple, A: int, B: int, C: int, D: int, op_char: str) -> Optional[str]:
    """Apply (digit_order, op, fmt) combo and return result string or None."""
    order, op_name, fmt_name = combo_name
    pairs = {name: (L, R) for name, L, R in _lrpairs(A, B, C, D)}
    L, R = pairs.get(order, (None, None))
    if L is None:
        return None
    raw = _apply_op(op_name, L, R)
    if raw is None:
        return None
    return _apply_fmt(fmt_name, raw, op_char)


# ---------------------------------------------------------------------------
# Solve a single symbol-digit problem
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    solved: bool
    combo: Optional[tuple]   # (digit_order, op, fmt)
    answer: Optional[str]
    scan_steps: list         # list of (combo, attempted_result, matched) for trace generation


def solve_symbol_digit(examples: list, question: tuple) -> SolveResult:
    """
    examples: list of (A, B, op_char, C, D, result_str)
    question: (A, B, op_char, C, D)
    """
    if len(examples) < 2:
        return SolveResult(False, None, None, [])

    ex1 = examples[0]
    ex2 = examples[1]
    A1, B1, op1, C1, D1, r1 = ex1
    A2, B2, op2, C2, D2, r2 = ex2
    op_char = op1  # all examples use the same op char

    scan_steps = []
    for combo in COMBOS:
        res1 = apply_combo(combo, A1, B1, C1, D1, op_char)
        matched_ex1 = (res1 == r1)
        scan_steps.append((combo, res1, matched_ex1))
        if not matched_ex1:
            continue
        # Verify against EX2
        res2 = apply_combo(combo, A2, B2, C2, D2, op_char)
        if res2 != r2:
            continue
        # Optional: verify against EX3 if available
        if len(examples) > 2:
            ex3 = examples[2]
            A3, B3, op3, C3, D3, r3 = ex3
            res3 = apply_combo(combo, A3, B3, C3, D3, op_char)
            if res3 != r3:
                continue
        # Locked — apply to question
        qA, qB, qop, qC, qD = question
        answer = apply_combo(combo, qA, qB, qC, qD, qop)
        if answer is None:
            continue
        return SolveResult(True, combo, answer, scan_steps)

    return SolveResult(False, None, None, scan_steps)


# ---------------------------------------------------------------------------
# Parse a full problem prompt (numeric version)
# ---------------------------------------------------------------------------

def parse_problem_prompt(prompt: str):
    """
    Parse a full equation-type problem prompt.
    Returns (examples_list, question_tuple) or (None, None).

    Key: examples are filtered to ONLY those with the same operator as the
    question, because each operator has its own independent rule.
    """
    lines = [l.strip() for l in prompt.strip().split('\n') if l.strip()]
    all_examples = []
    question = None

    for line in lines:
        # Question line
        if 'determine the result for' in line.lower() or 'result for:' in line.lower():
            m = re.search(r':\s*(\d{2})([^\d\s])(\d{2})\s*$', line)
            if m:
                lhs1, op, lhs2 = m.groups()
                A, B = int(lhs1[0]), int(lhs1[1])
                C, D = int(lhs2[0]), int(lhs2[1])
                question = (A, B, op, C, D)
            continue
        # Example line with =
        parsed = parse_eq_line(line)
        if parsed:
            all_examples.append(parsed)

    if question is None:
        return None, None

    # Filter examples to only those matching the question's operator
    q_op = question[2]
    examples = [e for e in all_examples if e[2] == q_op]

    if len(examples) < 1:
        # Fallback: use all examples (some problems have only 1 example per op)
        examples = all_examples

    if len(examples) < 2:
        return None, None
    return examples, question


# ---------------------------------------------------------------------------
# Cipher-digit: crack the symbol→digit mapping
# ---------------------------------------------------------------------------

def crack_cipher(examples: list[tuple[str, str]]) -> Optional[dict]:
    """
    examples: list of (cipher_input, cipher_output) strings.
    The operator is always at position 2 in the input.
    All digits 0-9 must appear in inputs or outputs to be crackable.

    Returns sym→digit dict or None if contradictory/incomplete.
    """
    mapping: dict[str, int] = {}  # sym -> digit
    rev_mapping: dict[int, str] = {}  # digit -> sym

    def try_assign(sym: str, digit: int) -> bool:
        if sym in mapping:
            return mapping[sym] == digit
        if digit in rev_mapping:
            return False
        mapping[sym] = digit
        rev_mapping[digit] = sym
        return True

    for cipher_input, cipher_output in examples:
        # cipher_input format: "XY<op>ZW", cipher_output: digits encoded
        if len(cipher_input) < 5:
            continue
        sym_A = cipher_input[0]
        sym_B = cipher_input[1]
        # index 2 is the operator symbol (skip for digit mapping)
        sym_C = cipher_input[3]
        sym_D = cipher_input[4]
        # The output is a cipher-encoded number; decode its digits
        # For now we just collect unique symbols; actual digit assignment
        # requires knowing which digit each symbol maps to.
        # We do this by trying to infer from the examples collectively.

    # Multi-pass: for each example, we know AB<op>CD = R
    # But without knowing digits yet, we can't directly assign.
    # Strategy: collect all symbols; try all bijective mappings? Too slow.
    # Better: use the solve loop below (try to find mapping that makes all examples consistent)
    return None  # Placeholder — see solve_cipher_digit below


def solve_cipher_digit(prompt: str) -> SolveResult:
    """
    Solve a cipher-digit problem by:
    1. Collecting all unique symbols and their positions
    2. Trying to crack the cipher by trying digit assignments that make
       the examples consistent under some combo
    3. If found, apply to target

    This is a backtracking search over (cipher_mapping × combo).
    For efficiency, we fix the operator symbol at position 2 and try to
    infer digit assignments from the numeric examples.
    """
    # Parse examples from prompt
    lines = [l.strip() for l in prompt.strip().split('\n') if l.strip()]
    raw_examples = []
    question_raw = None

    for line in lines:
        if 'determine the result for' in line.lower() or 'result for:' in line.lower():
            m = re.search(r':\s*(\S+)\s*$', line)
            if m:
                question_raw = m.group(1)
            continue
        m = re.match(r'^(\S+)\s*=\s*(\S+)$', line)
        if m:
            raw_examples.append((m.group(1), m.group(2)))

    if not raw_examples or question_raw is None:
        return SolveResult(False, None, None, [])

    # The operator symbol is at position 2 in input (always)
    op_sym = raw_examples[0][0][2] if len(raw_examples[0][0]) >= 3 else None
    if op_sym is None:
        return SolveResult(False, None, None, [])

    # Collect all non-operator symbols from inputs and outputs
    all_input_syms = set()
    all_output_syms = set()
    for inp, out in raw_examples:
        for i, c in enumerate(inp):
            if i != 2:
                all_input_syms.add(c)
        all_output_syms.update(set(out))

    # All symbols that could be digits
    digit_syms = all_input_syms | all_output_syms

    # Build partial mapping from examples by assuming structure:
    # We'll use a constraint-propagation approach with backtracking
    # For each example: inp[0],inp[1] are A,B; inp[3],inp[4] are C,D
    # We try to find mapping M: sym→int (0-9) and combo such that
    # apply_combo(combo, M[A], M[B], M[C], M[D]) == decode_output(M, out)

    # For tractability, limit to inputs with exactly 5 chars
    valid_examples = [(inp, out) for inp, out in raw_examples
                      if len(inp) == 5 and inp[2] == op_sym]

    if len(valid_examples) < 2:
        return SolveResult(False, None, None, [])

    # Try to infer mapping using brute force over assignments for
    # the symbols that appear in input positions (A,B,C,D)
    input_pos_syms = []
    for inp, _ in valid_examples:
        for i in [0, 1, 3, 4]:
            if inp[i] not in input_pos_syms:
                input_pos_syms.append(inp[i])

    # This is potentially 10! search space — too large.
    # Use heuristic: try all combos first with a symbolic solver
    # that constrains the mapping.
    #
    # Better approach: for each combo, check if there's a consistent
    # integer assignment for the symbols.

    from itertools import permutations

    unique_syms = list(dict.fromkeys(  # preserve order, deduplicate
        s for inp, out in valid_examples
        for i, s in enumerate(inp + out) if i != 2 or inp == out  # crude
    ))
    # Actually collect cleanly
    all_syms_ordered = []
    seen_s = set()
    for inp, out in valid_examples:
        for i, c in enumerate(inp):
            if i == 2:
                continue
            if c not in seen_s:
                all_syms_ordered.append(c)
                seen_s.add(c)
        for c in out:
            if c not in seen_s:
                all_syms_ordered.append(c)
                seen_s.add(c)
    if len(question_raw) > 0:
        for i, c in enumerate(question_raw):
            if i == 2:
                continue
            if c not in seen_s:
                all_syms_ordered.append(c)
                seen_s.add(c)

    # If too many unique symbols (>10) can't map to digits
    if len(all_syms_ordered) > 10:
        return SolveResult(False, None, None, [])

    # Try each combo; for each combo try to find a consistent digit assignment
    for combo in COMBOS:
        order, op_name, fmt_name = combo
        # Skip operator-prefixed formats for cipher (operator is encrypted too)
        if 'op_' in fmt_name:
            continue

        # For this combo, we need: for each example (A,B,C,D,R),
        # apply_combo(combo, A,B,C,D) == R
        # This defines constraints on the digit assignment.
        # We'll solve via backtracking over assignments for all_syms_ordered.

        # Constraint collection for this combo:
        constraints = []
        for inp, out in valid_examples:
            if len(inp) != 5:
                continue
            sym_A, sym_B, _, sym_C, sym_D = inp[0], inp[1], inp[2], inp[3], inp[4]
            constraints.append((sym_A, sym_B, sym_C, sym_D, out))

        def backtrack(sym_list, mapping, rev_mapping):
            if not sym_list:
                # Check all constraints
                for sym_A, sym_B, sym_C, sym_D, out in constraints:
                    if any(s not in mapping for s in [sym_A, sym_B, sym_C, sym_D]):
                        return None
                    A = mapping[sym_A]; B = mapping[sym_B]
                    C = mapping[sym_C]; D = mapping[sym_D]
                    res = apply_combo(combo, A, B, C, D, op_sym)
                    if res is None:
                        return None
                    # Decode output using mapping
                    decoded_out = ""
                    for c in out:
                        if c not in mapping:
                            return None
                        decoded_out += str(mapping[c])
                    if res != decoded_out:
                        return None
                return dict(mapping)

            sym = sym_list[0]
            rest = sym_list[1:]
            for digit in range(10):
                if digit in rev_mapping:
                    continue
                mapping[sym] = digit
                rev_mapping[digit] = sym
                result = backtrack(rest, mapping, rev_mapping)
                if result is not None:
                    return result
                del mapping[sym]
                del rev_mapping[digit]
            return None

        # Only try if we have ≤8 unique symbols (otherwise too slow)
        if len(all_syms_ordered) <= 8:
            found_mapping = backtrack(list(all_syms_ordered), {}, {})
            if found_mapping:
                # Apply to question
                if len(question_raw) != 5 or question_raw[2] != op_sym:
                    continue
                qA = found_mapping.get(question_raw[0])
                qB = found_mapping.get(question_raw[1])
                qC = found_mapping.get(question_raw[3])
                qD = found_mapping.get(question_raw[4])
                if None in [qA, qB, qC, qD]:
                    continue
                numeric_ans = apply_combo(combo, qA, qB, qC, qD, op_sym)
                if numeric_ans is None:
                    continue
                # Re-encode answer back to cipher symbols
                rev_map = {v: k for k, v in found_mapping.items()}
                cipher_ans = ""
                for digit_char in numeric_ans:
                    d = int(digit_char)
                    if d not in rev_map:
                        cipher_ans = None
                        break
                    cipher_ans += rev_map[d]
                if cipher_ans is None:
                    continue
                return SolveResult(True, combo, cipher_ans, [])

    return SolveResult(False, None, None, [])


# ---------------------------------------------------------------------------
# CoT trace generator (for SFT data)
# ---------------------------------------------------------------------------

def generate_cot_trace(examples: list, question: tuple, result: SolveResult) -> str:
    """Generate a structured chain-of-thought trace in Donald's template style."""
    if not result.solved:
        return ""

    A1, B1, op1, C1, D1, r1 = examples[0]
    A2, B2, op2, C2, D2, r2 = examples[1]
    op_char = op1
    order, op_name, fmt_name = result.combo

    qA, qB, qop, qC, qD = question

    # Build pair labels
    pairs_ex1 = {name: (L, R) for name, L, R in _lrpairs(A1, B1, C1, D1)}
    pairs_ex2 = {name: (L, R) for name, L, R in _lrpairs(A2, B2, C2, D2)}
    pairs_q   = {name: (L, R) for name, L, R in _lrpairs(qA, qB, qC, qD)}

    L1, R1 = pairs_ex1[order]
    L2, R2 = pairs_ex2[order]
    Lq, Rq = pairs_q[order]

    raw1 = _apply_op(op_name, L1, R1)
    raw2 = _apply_op(op_name, L2, R2)
    rawq = _apply_op(op_name, Lq, Rq)
    ans = result.answer

    # Scan lines (first N steps shown)
    scan_lines = []
    for i, (combo, attempted, matched) in enumerate(result.scan_steps[:20], 1):
        cord, cop, cfmt = combo
        cpairs = {name: (L, R) for name, L, R in _lrpairs(A1, B1, C1, D1)}
        cL, cR = cpairs.get(cord, (0, 0))
        raw_c = _apply_op(cop, cL, cR)
        raw_c_str = str(raw_c) if raw_c is not None else "?"
        match_str = "YES" if matched else "NO"
        scan_lines.append(f"#{i}:{cord}|{cop}|{cfmt} L={cL},R={cR} {raw_c_str}->{attempted} vs {r1} {match_str}")
        if matched:
            break

    scan_block = "\n".join(scan_lines)

    trace = f"""S1: PARSE
Operator: '{op_char}'
EX1: {A1}{B1}{op_char}{C1}{D1} = {r1}
EX1 digits: A={A1},B={B1},C={C1},D={D1}

S2: SCAN
{scan_block}
VER EX2: {A2}{B2}{op_char}{C2}{D2} {order}|L={L2},R={R2}|{op_name}={raw2}|{fmt_name}={r2} vs {r2} YES
LOCK: {order}|{op_name}|{fmt_name}

S3: APPLY
Target: {qA}{qB}{qop}{qC}{qD} A={qA},B={qB},C={qC},D={qD}
{order}: L={Lq},R={Rq}
{op_name}({Lq},{Rq})={rawq} {fmt_name}={ans}

S4: ANS={ans}

\\boxed{{{ans}}}"""
    return trace


# ---------------------------------------------------------------------------
# CLI: test on training data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import json
    from pathlib import Path

    PROJECT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT / "data" / "by_category"

    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "test":
        # Test solve rate on equation_numeric training data
        csv_file = DATA_DIR / "equation_numeric.csv"
        total = 0
        solved = 0
        wrong = 0

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                examples, question = parse_problem_prompt(row["prompt"])
                if examples is None:
                    continue
                res = solve_symbol_digit(examples, question)
                if res.solved:
                    if res.answer == row["answer"]:
                        solved += 1
                    else:
                        wrong += 1
                        if wrong <= 3:
                            print(f"WRONG: predicted={res.answer!r} actual={row['answer']!r}")
                            print(f"  combo={res.combo}")
                            print(f"  prompt={row['prompt'][:120]}")

        print(f"\n=== equation_numeric results ===")
        print(f"Total:  {total}")
        print(f"Solved (correct): {solved} ({100*solved/total:.1f}%)")
        print(f"Solved (wrong):   {wrong}")
        print(f"Unsolved:         {total - solved - wrong}")

    elif mode == "test_symbolic":
        # Test on cipher-digit (slower due to backtracking)
        csv_file = DATA_DIR / "equation_symbolic.csv"
        total = 0
        solved = 0
        wrong = 0

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 100:  # limit for speed
                    break
                total += 1
                res = solve_cipher_digit(row["prompt"])
                if res.solved:
                    if res.answer == row["answer"]:
                        solved += 1
                    else:
                        wrong += 1
                        if wrong <= 3:
                            print(f"WRONG: predicted={res.answer!r} actual={row['answer']!r}")

        print(f"\n=== equation_symbolic results (first {total}) ===")
        print(f"Solved (correct): {solved}")
        print(f"Solved (wrong):   {wrong}")
        print(f"Unsolved:         {total - solved - wrong}")
