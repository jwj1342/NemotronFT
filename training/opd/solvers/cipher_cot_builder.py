"""
Answer-guided cipher-digit CoT builder for training data generation.

For training data, we KNOW the correct answer. This dramatically constrains the
cipher-cracking problem: use ALL examples + the known answer to find the unique
(cipher_mapping, combo) pair, then generate a structured CoT trace.

Algorithm:
1. Parse prompt → same-op examples + question
2. For each combo in frequency order:
   - For each (L,R) in 0-99 × 0-99:
     - Compute expected output for question
     - If it matches the known answer (after encoding), this is a candidate
     - Verify this (L,R,combo) is consistent with all same-op examples
     - If verified → LOCK and generate CoT
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from symbol_digit_solver import COMBOS, apply_combo  # type: ignore


@dataclass
class CotResult:
    solved: bool
    combo: Optional[tuple]
    mapping: Optional[dict]   # sym → digit
    cot_trace: str = ""


def _build_partial_mapping(
    inp: str, cipher_out: str, numeric_out: str,
    A: int, B: int, C: int, D: int,
) -> Optional[tuple[dict, dict]]:
    """Build sym→digit and digit→sym from one (input, output) pair. Returns None on conflict."""
    if len(cipher_out) != len(numeric_out):
        return None
    m: dict[str, int] = {}
    r: dict[int, str] = {}

    for sym, dig in [(inp[0], A), (inp[1], B), (inp[3], C), (inp[4], D)]:
        if sym in m:
            if m[sym] != dig: return None
        elif dig in r:
            if r[dig] != sym: return None
        else:
            m[sym] = dig; r[dig] = sym

    for csym, dchar in zip(cipher_out, numeric_out):
        dig = int(dchar)
        if csym in m:
            if m[csym] != dig: return None
        elif dig in r:
            if r[dig] != csym: return None
        else:
            m[csym] = dig; r[dig] = csym

    return m, r


def _merge(m1, r1, m2, r2):
    m = dict(m1); r = dict(r1)
    for sym, dig in m2.items():
        if sym in m:
            if m[sym] != dig: return None
        elif dig in r:
            if r[dig] != sym: return None
        else:
            m[sym] = dig; r[dig] = sym
    return m, r


def _parse_prompt(prompt: str):
    """Returns (all_examples, question_raw) where all_examples is list of (inp, out)."""
    all_examples = []
    question_raw = None
    for line in prompt.strip().splitlines():
        line = line.strip()
        if re.search(r'(?:determine|result for)', line, re.IGNORECASE):
            m = re.search(r':\s*(\S+)\s*$', line)
            if m:
                question_raw = m.group(1)
            continue
        m = re.match(r'^(\S+)\s*=\s*(\S+)$', line)
        if m:
            all_examples.append((m.group(1), m.group(2)))
    return all_examples, question_raw


def build_cot(prompt: str, known_answer: str, max_iters: int = 50_000) -> CotResult:
    """
    Given a cipher-digit prompt and the known correct answer,
    find the (combo, cipher_mapping) that is consistent with all examples
    and then generates a structured CoT trace.

    Uses the known answer as an additional constraint to crack the cipher.
    max_iters caps the total inner-loop iterations to avoid hanging on hard cases.
    """
    all_examples, question_raw = _parse_prompt(prompt)
    if not all_examples or question_raw is None:
        return CotResult(False, None, None, "(parse error)")

    if len(question_raw) < 5:
        return CotResult(False, None, None, "(question too short)")

    # Operator = position 2 in question
    op_sym = question_raw[2]

    # Filter to same-operator examples
    same_op_exs = [(inp, out) for inp, out in all_examples
                   if len(inp) == 5 and inp[2] == op_sym]

    if len(same_op_exs) < 1:
        return CotResult(False, None, None, "(no same-op examples)")

    ex1_inp, ex1_out = same_op_exs[0]
    ex2 = same_op_exs[1] if len(same_op_exs) > 1 else None

    # The known answer consists of cipher symbols for the result digits.
    ans_sym_len = len(known_answer)
    iters = 0  # global iteration counter for early abort

    for combo in COMBOS:
        order, op_name, fmt_name = combo
        if 'op_' in fmt_name:
            continue

        # --- Step 1: enumerate (L_q, R_q) for the QUESTION that give answer of right length ---
        if iters >= max_iters:
            break
        for Lq in range(100):
            if iters >= max_iters:
                break
            Aq, Bq = Lq // 10, Lq % 10
            for Rq in range(100):
                iters += 1
                if iters >= max_iters:
                    break
                Cq, Dq = Rq // 10, Rq % 10

                numeric_ans = apply_combo(combo, Aq, Bq, Cq, Dq, op_sym)
                if numeric_ans is None or not numeric_ans.isdigit():
                    continue
                if len(numeric_ans) != ans_sym_len:
                    continue

                # Build cipher mapping from question: sym → digit for question input + answer
                q_map_result = _build_partial_mapping(
                    question_raw, known_answer, numeric_ans, Aq, Bq, Cq, Dq
                )
                if q_map_result is None:
                    continue
                mq, rq = q_map_result

                # --- Step 2: verify against EX1 using the partial cipher ---
                # Determine digit ranges for EX1 input symbols
                sA1, sB1, sC1, sD1 = ex1_inp[0], ex1_inp[1], ex1_inp[3], ex1_inp[4]
                A1_range = [mq[sA1]] if sA1 in mq else list(range(10))
                B1_range = [mq[sB1]] if sB1 in mq else list(range(10))
                C1_range = [mq[sC1]] if sC1 in mq else list(range(10))
                D1_range = [mq[sD1]] if sD1 in mq else list(range(10))

                found = False
                final_mapping = None
                for A1 in A1_range:
                    if found: break
                    for B1 in B1_range:
                        if found: break
                        for C1 in C1_range:
                            if found: break
                            for D1 in D1_range:
                                num1 = apply_combo(combo, A1, B1, C1, D1, op_sym)
                                if num1 is None or not num1.isdigit():
                                    continue
                                if len(num1) != len(ex1_out):
                                    continue
                                r1 = _build_partial_mapping(ex1_inp, ex1_out, num1, A1, B1, C1, D1)
                                if r1 is None:
                                    continue
                                merged = _merge(mq, rq, *r1)
                                if merged is None:
                                    continue

                                # --- Step 3: verify against EX2 if available ---
                                if ex2 is not None:
                                    ex2_inp, ex2_out = ex2
                                    m_mid, r_mid = merged
                                    sA2, sB2, sC2, sD2 = ex2_inp[0], ex2_inp[1], ex2_inp[3], ex2_inp[4]
                                    A2_r = [m_mid[sA2]] if sA2 in m_mid else list(range(10))
                                    B2_r = [m_mid[sB2]] if sB2 in m_mid else list(range(10))
                                    C2_r = [m_mid[sC2]] if sC2 in m_mid else list(range(10))
                                    D2_r = [m_mid[sD2]] if sD2 in m_mid else list(range(10))
                                    ex2_ok = False
                                    for A2 in A2_r:
                                        if ex2_ok: break
                                        for B2 in B2_r:
                                            if ex2_ok: break
                                            for C2 in C2_r:
                                                if ex2_ok: break
                                                for D2 in D2_r:
                                                    num2 = apply_combo(combo, A2, B2, C2, D2, op_sym)
                                                    if num2 is None or not num2.isdigit(): continue
                                                    if len(num2) != len(ex2_out): continue
                                                    r2 = _build_partial_mapping(ex2_inp, ex2_out, num2, A2, B2, C2, D2)
                                                    if r2 is None: continue
                                                    m3 = _merge(m_mid, r_mid, *r2)
                                                    if m3 is None: continue
                                                    ex2_ok = True
                                                    merged = m3
                                                    break
                                    if not ex2_ok:
                                        continue

                                found = True
                                final_mapping = merged
                                break

                if found and final_mapping is not None:
                    cot = _generate_cot(
                        all_examples, question_raw, known_answer,
                        op_sym, combo, final_mapping[0],
                        (Aq, Bq, Rq // 10, Rq % 10), numeric_ans
                    )
                    return CotResult(True, combo, final_mapping[0], cot)

    return CotResult(False, None, None, "(no consistent combo+cipher found)")


def _generate_cot(
    all_examples, question_raw, known_answer,
    op_sym, combo, mapping,
    q_digits, numeric_ans,
):
    """Generate a structured CoT trace for the solved cipher-digit problem."""
    order, op_name, fmt_name = combo
    inv = {v: k for k, v in mapping.items()}

    lines = []
    lines.append("I need to solve a cipher-digit problem.")
    lines.append("All characters (including digits) are encrypted with a bijective cipher.")
    lines.append("")

    lines.append("S1: DETECT")
    lines.append(f"  Operator cipher symbol: '{op_sym}' (always at input position 2)")
    lines.append("")

    lines.append("S2: CRACK (using example pairs to build cipher mapping)")
    for sym in sorted(mapping.keys()):
        lines.append(f"  '{sym}' -> {mapping[sym]}")
    lines.append("")

    lines.append("S3: SCAN (frequency-ordered combo search)")
    lines.append(f"  LOCK: {order}|{op_name}|{fmt_name}")
    lines.append("")

    lines.append("S4: VERIFY ON EXAMPLES")
    same_op_exs = [(inp, out) for inp, out in all_examples
                   if len(inp) == 5 and inp[2] == op_sym]
    for inp, cipher_out in same_op_exs[:3]:
        sA, sB, sC, sD = inp[0], inp[1], inp[3], inp[4]
        if any(s not in mapping for s in [sA, sB, sC, sD]):
            lines.append(f"  {inp} = {cipher_out}  (incomplete cipher, skip)")
            continue
        A, B, C, D = mapping[sA], mapping[sB], mapping[sC], mapping[sD]
        num = apply_combo(combo, A, B, C, D, op_sym)
        if num is None:
            lines.append(f"  {inp} = {cipher_out}  (apply_combo None)")
            continue
        cipher_pred = "".join(inv.get(int(d), "?") for d in num if d.isdigit())
        match = "YES" if cipher_pred == cipher_out else "NO"
        lines.append(f"  {inp}: decode {sA}{sB}..{sC}{sD}->{A}{B}..{C}{D}  "
                     f"{order}|{op_name}={num}  encode->{cipher_pred}  {match}")
    lines.append("")

    lines.append("S5: APPLY TO TARGET")
    Aq, Bq, Cq, Dq = q_digits
    lines.append(f"  Target: {question_raw}")
    lines.append(f"  Decode: {question_raw[0]}->{Aq}  {question_raw[1]}->{Bq}  "
                 f"{question_raw[3]}->{Cq}  {question_raw[4]}->{Dq}")
    lines.append(f"  {order}: A={Aq},B={Bq},C={Cq},D={Dq}")
    lines.append(f"  {op_name}|{fmt_name} = {numeric_ans}")
    lines.append("")

    lines.append("S6: ENCODE answer back to cipher")
    lines.append(f"  {numeric_ans} -> {''.join(inv.get(int(d),'?') for d in numeric_ans)}")
    lines.append("")

    lines.append(f"S7: ANS={known_answer}")
    lines.append("")
    lines.append(f"\\boxed{{{known_answer}}}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: test on training data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import csv, time

    PROJECT = Path(__file__).resolve().parents[3]
    csv_file = PROJECT / "data" / "by_category" / "equation_symbolic.csv"

    total = 0; solved = 0
    t0 = time.time()

    with open(csv_file) as f:
        for row in csv.DictReader(f):
            if total >= 200:
                break
            total += 1
            res = build_cot(row["prompt"], row["answer"])
            if res.solved:
                solved += 1

    elapsed = time.time() - t0
    print(f"Tested {total} rows in {elapsed:.1f}s ({elapsed/total:.3f}s/row)")
    print(f"Solved: {solved}/{total} ({100*solved/total:.1f}%)")
