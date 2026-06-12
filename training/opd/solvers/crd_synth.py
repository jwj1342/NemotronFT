"""
Synthetic generator for "cryptarithm_deduce" — the inverse of crd_solver.

gen_problem(rng) -> dict | None
    Sample a fresh problem (cipher, op-map, pp, facts, query), render the Alice
    prompt + answer, then VERIFY with crd_solver.solve() that it is well-posed
    (unique forward solution) and that crd_trace.render_trace reaches the answer.
    Returns {id, prompt, answer, cot, n_facts, ...} or None if the draw is not
    a clean well-posed/renderable instance (caller just retries).

Matches the measured real distribution:
  * #facts {3,4,5} ~ {180,241,238}
  * 10-13 distinct symbols (cipher is a 10-digit bijection; alphabet drawn from a
    larger glyph pool so unused glyphs serve as operator/sign symbols)
  * result-length mix biased toward 2-4 (anchor is a 4-digit mul-family result)
  * negatives + sign symbol + swap preprocessing at realistic rates
"""
from __future__ import annotations

import random
import string
from typing import Optional, Dict, List

from crd_solver import (
    solve, _apply_op, _operand_string, _result_string, ALL_OPS, MUL_FAMILY,
)
from crd_trace import render_trace, verify_trace

# Glyph pool used by the real puzzles (printable punctuation, no whitespace, no
# '=' / ':' which are structural in the prompt).
_GLYPH_POOL = [c for c in (string.punctuation + string.ascii_letters)
               if c not in "=: \t\n"]

# operator glyphs in the real set are drawn from the same pool; we just reserve
# a few glyphs as operator symbols distinct from cipher (digit) symbols.

# op frequency weights — anchor must be mul-family often (matches real anchor op
# distribution: mul 212, mul+1 47, mul-1 39 dominate).
_NONANCHOR_OPS = ALL_OPS  # any of the 13 for secondary facts
_NONANCHOR_W = [6, 3, 3, 5, 4, 3, 1, 1, 6, 3, 3, 4, 4]  # add/sub/mul-ish common

_PP_WEIGHTS = {"none": 0.72, "swap": 0.28}  # ~swap rate seen in real well-posed


def _draw_pp(rng) -> str:
    return "swap" if rng.random() < _PP_WEIGHTS["swap"] else "none"


def _encode_operand(val2: int, pp: str, cipher_rev: Dict[int, str]) -> str:
    """2-digit operand value -> 2 cipher symbols (pp applied to the digit STRING)."""
    s = _operand_string(val2, pp)   # e.g. "43" or "34"
    return "".join(cipher_rev[int(c)] for c in s)


def _encode_result(raw: int, pp_r: str, cipher_rev: Dict[int, str],
                   sign_sym: Optional[str]) -> Optional[str]:
    """operation value -> result symbol string at NATURAL minimal width + pp_r,
    with a leading sign symbol when negative (mirrors solver._encode_answer)."""
    av = abs(raw)
    ds = str(av)                 # natural minimal width
    emit = ds[::-1] if pp_r == "swap" else ds
    try:
        body = "".join(cipher_rev[int(c)] for c in emit)
    except KeyError:
        return None
    if raw < 0:
        if sign_sym is None:
            return None
        return sign_sym + body
    return body


_PREAMBLES = [
    "In Alice's Wonderland, a secret set of transformation rules is applied to "
    "equations. Below are a few examples:",
]


def _build_prompt(fact_strs: List[str], query_str: str) -> str:
    head = _PREAMBLES[0]
    body = "\n".join(fact_strs)
    return f"{head}\n{body}\nNow, determine the result for: {query_str}"


def gen_problem(rng: random.Random) -> Optional[dict]:
    # ---- choose structure ----
    n_facts = rng.choices([3, 4, 5], weights=[180, 241, 238])[0]
    # cipher: bijection over 10 digits using 10 distinct glyphs
    n_cipher_syms = 10
    # operator glyphs: 1-3 distinct, plus optionally a sign glyph
    # a negative result needs a >=2 op set so one op glyph can double as the minus
    # sign (real-data convention: the sign char is itself an operator glyph).
    use_sign = rng.random() < 0.45
    n_ops = rng.choices([1, 2, 3], weights=[2, 5, 4] if not use_sign else [0, 5, 4])[0]

    total_glyphs = n_cipher_syms + n_ops
    if total_glyphs > len(_GLYPH_POOL):
        return None
    glyphs = rng.sample(_GLYPH_POOL, total_glyphs)
    cipher_syms = glyphs[:n_cipher_syms]
    op_glyphs = glyphs[n_cipher_syms:n_cipher_syms + n_ops]
    # The sign symbol of a negative result is, in the real puzzles, ALWAYS one of
    # the operator glyphs reused as a leading minus (solver._split_sign requires
    # the leading sign char to be in op_syms). Assigned from op_glyphs below.
    sign_sym = None

    # cipher: digit -> symbol (bijection)
    digits = list(range(10))
    rng.shuffle(digits)
    cipher_rev = {d: cipher_syms[i] for i, d in enumerate(digits)}  # digit->sym

    # global preprocessing
    pp = (_draw_pp(rng), _draw_pp(rng), _draw_pp(rng))
    pp_a, pp_b, pp_r = pp

    # op-map: each op glyph -> an op. Ensure at least one glyph is mul-family so an
    # anchor with a 3-4 digit result exists.
    op_map: Dict[str, str] = {}
    anchor_glyph = op_glyphs[0]
    op_map[anchor_glyph] = rng.choices(MUL_FAMILY, weights=[6, 3, 3, 2, 2])[0]
    diff_glyph = None
    for g in op_glyphs[1:]:
        op_map[g] = rng.choices(_NONANCHOR_OPS, weights=_NONANCHOR_W)[0]
    # To produce negatives, force ONE secondary glyph to a difference-family op AND
    # reuse THAT SAME glyph as the leading minus sign (real-data convention).
    if use_sign and len(op_glyphs) >= 2:
        diff_glyph = op_glyphs[1]
        op_map[diff_glyph] = rng.choice(["sub", "rsub", "neg_abs"])
        sign_sym = diff_glyph

    # ---- sample facts ----
    fact_strs = []
    # guarantee the FIRST fact is the mul-family anchor with a >=3 digit result
    def make_fact(glyph, want_anchor, want_negative=False):
        op = op_map[glyph]
        for _ in range(120):
            a2 = rng.randint(0, 99)
            b2 = rng.randint(0, 99)
            a_val = int(_operand_string(a2, pp_a))
            b_val = int(_operand_string(b2, pp_b))
            raw = _apply_op(op, a_val, b_val)
            if raw is None:
                continue
            if raw < 0 and sign_sym is None:
                continue
            if want_negative and raw >= 0:
                continue
            if want_anchor and len(str(abs(raw))) < 3:
                continue
            a_s = _encode_operand(a2, pp_a, cipher_rev)
            b_s = _encode_operand(b2, pp_b, cipher_rev)
            r_s = _encode_result(raw, pp_r, cipher_rev, sign_sym)
            if r_s is None:
                continue
            lhs = a_s + glyph + b_s
            if len(lhs) != 5:
                continue
            return lhs, r_s
        return None

    fa = make_fact(anchor_glyph, True)
    if fa is None:
        return None
    fact_strs.append(f"{fa[0]} = {fa[1]}")
    used_glyphs = [anchor_glyph]
    neg_demoed = False
    for i in range(1, n_facts):
        # rotate through op glyphs so all get used; bias to reuse
        g = op_glyphs[i % len(op_glyphs)]
        # demonstrate a negative result on the diff glyph at least once when a sign
        # symbol is reserved (so the sign is deducible from the examples).
        want_neg = (g == diff_glyph and not neg_demoed and rng.random() < 0.8)
        ff = make_fact(g, False, want_negative=want_neg)
        if ff is None and want_neg:
            ff = make_fact(g, False)  # fall back to any result
        if ff is None:
            return None
        if ff[1] and ff[1][0] == sign_sym:
            neg_demoed = True
        fact_strs.append(f"{ff[0]} = {ff[1]}")
        used_glyphs.append(g)
    # if we reserved a sign symbol but never demonstrated a negative, the sign is
    # not forward-deducible -> drop this draw.
    if sign_sym is not None and not neg_demoed:
        return None

    # ensure every op glyph appears in at least one fact (so its op is deducible)
    for g in op_glyphs:
        if g not in used_glyphs:
            return None

    # query uses one of the op glyphs; bias toward the diff glyph sometimes so the
    # ANSWER itself is negative at a realistic rate.
    if diff_glyph is not None and neg_demoed and rng.random() < 0.35:
        q_glyph = diff_glyph
    else:
        q_glyph = rng.choice(op_glyphs)
    qa2, qb2 = rng.randint(0, 99), rng.randint(0, 99)
    q_a = _encode_operand(qa2, pp_a, cipher_rev)
    q_b = _encode_operand(qb2, pp_b, cipher_rev)
    query_str = q_a + q_glyph + q_b
    if len(query_str) != 5:
        return None

    prompt = _build_prompt(fact_strs, query_str)

    # ---- compute the TRUE answer from our generative spec ----
    a_val = int(_operand_string(qa2, pp_a))
    b_val = int(_operand_string(qb2, pp_b))
    raw = _apply_op(op_map[q_glyph], a_val, b_val)
    if raw is None:
        return None
    if op_map[q_glyph] in ("concat", "rconcat"):
        # concat copies operand symbols
        a = q_a[::-1] if pp_a == "swap" else q_a
        b = q_b[::-1] if pp_b == "swap" else q_b
        res = (a + b) if op_map[q_glyph] == "concat" else (b + a)
        answer = res[::-1] if pp_r == "swap" else res
    else:
        answer = _encode_result(raw, pp_r, cipher_rev, sign_sym)
    if answer is None:
        return None

    # ---- VERIFY well-posed + trace reaches answer (no leakage / bounded) ----
    sol = solve(prompt, answer)
    if not (sol.get("solved") and sol.get("well_posed") and sol.get("predicted") == answer):
        return None
    tr = render_trace(prompt, answer)
    if tr is None:
        return None
    good, _ = verify_trace(prompt, answer, tr)
    if not good:
        return None

    return {
        "prompt": prompt,
        "answer": answer,
        "cot": tr,
        "n_facts": n_facts,
        "pp": pp,
        "has_negative": (sign_sym is not None and "-" in str(raw)) or (raw < 0),
        "synthetic": True,
    }


def generate(n_target: int, seed: int = 0, max_tries_factor: int = 40) -> List[dict]:
    rng = random.Random(seed)
    out = []
    tries = 0
    cap = n_target * max_tries_factor
    while len(out) < n_target and tries < cap:
        tries += 1
        p = gen_problem(rng)
        if p is not None:
            out.append(p)
    return out


if __name__ == "__main__":
    import statistics
    from collections import Counter
    rng = random.Random(7)
    ok = 0
    tries = 0
    nf = Counter()
    rlen = Counter()
    neg = 0
    swap = 0
    while ok < 200 and tries < 8000:
        tries += 1
        p = gen_problem(rng)
        if p is None:
            continue
        ok += 1
        nf[p["n_facts"]] += 1
        if p["has_negative"]:
            neg += 1
        if p["pp"] != ("none", "none", "none"):
            swap += 1
    print(f"generated {ok} in {tries} tries (yield {ok/tries:.1%})")
    print("n_facts:", dict(nf))
    print(f"negative-result problems: {neg}  swap-pp problems: {swap}")
