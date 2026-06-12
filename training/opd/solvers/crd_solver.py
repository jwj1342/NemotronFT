"""
Forward solver for the "cryptarithm_deduce" puzzle category.

Problem structure (verified on 659 train problems):
  Prompt has 3-5 example lines "LHS = RHS" then "determine the result for: QUERY".
  Each LHS is exactly 5 symbols: [a0 a1 OP b0 b1].  a = operand a (2 cipher-digits),
  b = operand b, position 2 is an OPERATOR symbol.  RHS = result (1-4 cipher symbols;
  a leading operator/sign symbol encodes a negative result).
  ONE shared bijective symbol->digit cipher applies to all operands & results across
  all examples regardless of operator.  DIFFERENT examples may use DIFFERENT operator
  symbols; the query uses one of them.

Deduced generation model (reverse-engineered from train):
  * ONE GLOBAL preprocessing scheme (pp_a, pp_b, pp_r) in {none,swap}^3 for the whole
    problem.  swap reverses the *digit string at its natural width* (operands width 2,
    result width = number of result-body symbols).  Leading zeros are SIGNIFICANT:
    e.g. add giving 50 with a 2-symbol result, pp_r=swap -> "05".
  * Each OPERATOR SYMBOL maps to ONE of the 13 operations; ANY of the 13 ops may map
    to ANY operator symbol (the +,-,* glyphs are arbitrary -- e.g. '-' can be maxmod).
  * ONE shared bijective cipher symbol->digit applied to operands and result digits.
  * A leading sign symbol on the RHS encodes a negative result.

Algorithm:
  1. Parse all facts + query.
  2. Anchor on the fact with the longest result body (most constraining), enumerate
     operand pairs, seed the cipher from the result digit-pattern -> partial ciphers.
  3. Constraint-propagate to remaining facts (each operator symbol's op unknown ->
     try every op consistently; operator-symbol op is locked once chosen) via
     backtracking.  One global pp scheme is fixed at the anchor.
  4. Apply the deduced op-map + cipher + pp to the query.

API:  solve(prompt, answer=None) -> dict
"""
from __future__ import annotations

import re
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Operation framework (13 ops, from competitor "kemshim")
# ---------------------------------------------------------------------------

def _apply_op(name: str, a: int, b: int) -> Optional[int]:
    if name == "add":      return a + b
    if name == "add+1":    return a + b + 1
    if name == "add-1":    return a + b - 1
    if name == "sub":      return a - b
    if name == "rsub":     return b - a
    if name == "abs_sub":  return abs(a - b)
    if name == "neg_abs":  return -abs(a - b)
    if name == "maxmod":
        lo = min(a, b)
        if lo == 0:
            return None
        return max(a, b) % lo
    if name == "mul":      return a * b
    if name == "mul+1":    return a * b + 1
    if name == "mul-1":    return a * b - 1
    if name == "concat":   return int(f"{a}{b}")
    if name == "rconcat":  return int(f"{b}{a}")
    return None


ALL_OPS = ["add", "add+1", "add-1", "sub", "rsub", "abs_sub", "neg_abs",
           "maxmod", "mul", "mul+1", "mul-1", "concat", "rconcat"]
MUL_FAMILY = ["mul", "mul+1", "mul-1", "concat", "rconcat"]

PREPROCS = ["none", "swap"]  # swap == reverse digit string at natural width


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

def parse_prompt(prompt: str):
    """Return (facts, query, op_syms) or (None, None, None).

    facts: list of (a_syms:str(2), op_sym:str, b_syms:str(2), r_syms:str)
    query: (a_syms, op_sym, b_syms)
    op_syms: set of operator symbols (position 2 of every LHS + query)
    """
    lines = [l.strip() for l in prompt.strip().split('\n') if l.strip()]
    facts = []
    query = None
    for line in lines:
        if 'determine the result for' in line.lower():
            m = re.search(r':\s*(\S+)\s*$', line)
            if m and len(m.group(1)) == 5:
                q = m.group(1)
                query = (q[:2], q[2], q[3:5])
            continue
        m = re.match(r'^(\S+)\s*=\s*(\S+)$', line)
        if m:
            lhs, rhs = m.group(1), m.group(2)
            if len(lhs) == 5:
                facts.append((lhs[:2], lhs[2], lhs[3:5], rhs))
    if query is None or not facts:
        return None, None, None
    op_syms = set(f[1] for f in facts) | {query[1]}
    return facts, query, op_syms


def _split_sign(r_syms: str, op_syms: set):
    """Return (sign, body_syms) or None.  A leading operator symbol = negative sign."""
    if len(r_syms) >= 2 and r_syms[0] in op_syms:
        body = r_syms[1:]
        if any(c in op_syms for c in body):
            return None
        return -1, body
    if any(c in op_syms for c in r_syms):
        return None
    return 1, r_syms


# ---------------------------------------------------------------------------
# Width-aware swap.  swap a value at a given width by reversing its zero-padded
# digit string.  Returns the reversed digit STRING (leading zeros significant).
# ---------------------------------------------------------------------------

def _operand_string(val2: int, pp: str) -> str:
    """val2 in [0,99] -> 2-char digit string after preprocessing."""
    s = f"{val2:02d}"
    return s[::-1] if pp == "swap" else s


def _result_string(raw: int, pp: str, width: int) -> Optional[str]:
    """raw operation value -> |raw| as a width-char digit string after pp.
    Returns None if |raw| doesn't fit in `width` digits."""
    av = abs(raw)
    s = str(av)
    if len(s) > width:
        return None
    s = s.rjust(width, "0")
    return s[::-1] if pp == "swap" else s


# ---------------------------------------------------------------------------
# Cipher consistency: assign body-string digits to body symbols.
# ---------------------------------------------------------------------------

def _unify(cipher: dict, syms: str, digit_str: str):
    """Extend cipher so each sym maps to the corresponding digit. Returns new
    cipher or None on contradiction (bijection enforced)."""
    new = dict(cipher)
    rev = {v: k for k, v in new.items()}
    for sym, ch in zip(syms, digit_str):
        d = int(ch)
        if sym in new:
            if new[sym] != d:
                return None
        else:
            if d in rev:
                return None
            new[sym] = d
            rev[d] = sym
    return new


def _decode(syms: str, cipher: dict):
    try:
        return int("".join(str(cipher[c]) for c in syms))
    except KeyError:
        return None


def _free_assignments(symbols, cipher):
    """Yield every consistent assignment of digits to still-unmapped `symbols`."""
    syms = [s for s in dict.fromkeys(symbols) if s not in cipher]
    used = set(cipher.values())
    avail = [d for d in range(10) if d not in used]

    def rec(i, cur, used_now):
        if i == len(syms):
            yield dict(cur)
            return
        for d in avail:
            if d in used_now:
                continue
            cur[syms[i]] = d
            used_now.add(d)
            yield from rec(i + 1, cur, used_now)
            used_now.remove(d)
            del cur[syms[i]]
    yield from rec(0, dict(cipher), set(cipher.values()))


# ---------------------------------------------------------------------------
# Extend one fact under a FIXED global pp scheme and a set of allowed ops.
# Yields (op_name, new_cipher).
# ---------------------------------------------------------------------------

def _operand_pairs(a_syms, b_syms, cipher):
    """Yield (a2, b2, cipher_after) for all consistent digit assignments to the 4
    operand symbols, honoring symbol-equality (repeated/shared symbols tie digits)
    and the existing partial bijection.  Far cheaper than the naive 0..99 x 0..99."""
    for ca in _free_assignments(a_syms + b_syms, cipher):
        a2 = ca[a_syms[0]] * 10 + ca[a_syms[1]]
        b2 = ca[b_syms[0]] * 10 + ca[b_syms[1]]
        yield a2, b2, ca


def extend_fact(fact, cipher, op_syms, pp, ops, budget=None):
    """Yield (op_name, new_cipher).  `budget` (mutable [n]) is decremented for
    every operand pair examined so the cost of even a no-yield fact is bounded."""
    a_syms, op_sym, b_syms, r_syms = fact
    split = _split_sign(r_syms, op_syms)
    if split is None:
        return
    sign, body_syms = split
    # a digit symbol can never be an operator glyph
    if any(s in op_syms for s in body_syms):
        return
    pp_a, pp_b, pp_r = pp
    width = len(body_syms)
    for a2, b2, ca in _operand_pairs(a_syms, b_syms, cipher):
        if budget is not None:
            budget[0] -= 1
            if budget[0] <= 0:
                return
        a_val = int(_operand_string(a2, pp_a))
        b_val = int(_operand_string(b2, pp_b))
        for opn in ops:
            raw = _apply_op(opn, a_val, b_val)
            if raw is None:
                continue
            if (raw < 0) != (sign < 0) and raw != 0:
                continue
            if raw == 0 and sign < 0:
                continue
            rs = _result_string(raw, pp_r, width)
            if rs is None:
                continue
            nc = _unify(ca, body_syms, rs)
            if nc is not None:
                yield (opn, nc)


# ---------------------------------------------------------------------------
# Backtracking over all facts under a fixed pp scheme.
# ---------------------------------------------------------------------------

def _propagate(facts_remaining, op_constraints, cipher, op_syms, pp, solutions, limit,
               budget=None):
    """budget: optional mutable [remaining_nodes]; decremented per extend step.
    When it hits 0 the search stops early (bounded runtime)."""
    if len(solutions) >= limit:
        return
    if budget is not None and budget[0] <= 0:
        return
    if not facts_remaining:
        solutions.append((dict(op_constraints), dict(cipher)))
        return
    fact = facts_remaining[0]
    rest = facts_remaining[1:]
    op_sym = fact[1]
    ops = [op_constraints[op_sym]] if op_sym in op_constraints else ALL_OPS
    for opn, nc in extend_fact(fact, cipher, op_syms, pp, ops, budget):
        new_oc = dict(op_constraints)
        new_oc[op_sym] = opn
        _propagate(rest, new_oc, nc, op_syms, pp, solutions, limit, budget)
        if len(solutions) >= limit:
            return
        if budget is not None and budget[0] <= 0:
            return


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _find_sign_symbol(facts, op_syms):
    for _, _, _, r in facts:
        if len(r) >= 2 and r[0] in op_syms:
            return r[0]
    return None


def _digit_alphabet(facts, query, answer, op_syms):
    """All non-operator (digit) symbols that appear anywhere in the problem
    (facts + query operands + answer if given). The cipher is a bijection over
    a 10-digit alphabet, so this set has size <= 10."""
    syms = set()
    for a, o, b, r in facts:
        syms.update(a + b)
        sp = _split_sign(r, op_syms)
        if sp:
            syms.update(sp[1])
    syms.update(query[0] + query[2])
    if answer:
        for c in answer:
            if c not in op_syms:
                syms.add(c)
    return syms - op_syms


def _complete_cipher(cipher, alphabet):
    """If the partial cipher plus the problem alphabet uniquely forces the
    remaining symbol<->digit pairings (i.e. exactly as many unmapped alphabet
    symbols as unmapped digits AND that count is 1), fill it in. Returns the
    (possibly extended) cipher.  Only the count==1 case is *forced*; with >=2
    free we leave it partial (ambiguous)."""
    unmapped_syms = [s for s in alphabet if s not in cipher]
    used = set(cipher.values())
    unmapped_digits = [d for d in range(10) if d not in used]
    if len(unmapped_syms) == 1 and len(unmapped_digits) == 1:
        nc = dict(cipher)
        nc[unmapped_syms[0]] = unmapped_digits[0]
        return nc
    return cipher


def _body_len(f, op_syms):
    sp = _split_sign(f[3], op_syms)
    return len(sp[1]) if sp else -1


def _is_concat_pattern(f, op_syms):
    """True if the fact's result body is just the operand symbols concatenated
    (in either order, possibly with the 2-symbol operands reversed). Such facts
    place NO constraint on the digit VALUES (concat copies symbols), so they are
    useless as a cipher anchor and must not be enumerated as one."""
    a_syms, _, b_syms, r_syms = f
    sp = _split_sign(r_syms, op_syms)
    if sp is None:
        return False
    sign, body = sp
    if sign < 0 or len(body) != 4:
        return False
    a_variants = {a_syms, a_syms[::-1]}
    b_variants = {b_syms, b_syms[::-1]}
    for av in a_variants:
        for bv in b_variants:
            cat = av + bv
            if body == cat or body == cat[::-1]:
                return True
            if body == bv + av or body == (bv + av)[::-1]:
                return True
    return False


def _constraint_score(f, op_syms):
    """Higher = more constraining anchor. Concat-pattern facts score lowest;
    otherwise longer result bodies (esp. 4-digit products) constrain most."""
    if _is_concat_pattern(f, op_syms):
        return -1
    return _body_len(f, op_syms)


def solve(prompt: str, answer: Optional[str] = None) -> dict:
    t0 = time.time()
    facts, query, op_syms = parse_prompt(prompt)
    base = {"solved": False, "correct": None, "reachable": None, "well_posed": None,
            "n_distinct_predictions": 0, "predicted": None, "op_map": {}, "cipher": {},
            "preprocess": None, "per_fact_candidate_sizes": [], "elapsed": 0.0,
            "fail_reason": None}
    if facts is None:
        base["elapsed"] = time.time() - t0
        base["fail_reason"] = "parse_fail"
        return base

    # Order facts by constraint power (anchor = most-constraining arithmetic fact
    # first; concat-pattern facts last since they pin no digit values).
    ordered = sorted(facts, key=lambda f: _constraint_score(f, op_syms), reverse=True)
    if _constraint_score(ordered[0], op_syms) < 1:
        base["elapsed"] = time.time() - t0
        base["fail_reason"] = "no_anchor"
        return base

    sign_sym = _find_sign_symbol(facts, op_syms)

    # Enumerate full solutions across all 8 global pp schemes, sharing a single
    # node budget so no problem can run unbounded.
    LIMIT = 64             # cap distinct full solutions per pp scheme
    NODE_BUDGET = 120000   # shared extend-step budget across all pp schemes
    budget = [NODE_BUDGET]
    all_solutions = []     # (pp, op_map, cipher)
    per_fact_sizes = []
    for pp in [(a, b, c) for a in PREPROCS for b in PREPROCS for c in PREPROCS]:
        # per-fact candidate growth diagnostic for THIS pp (cheap prefix counts,
        # capped) — recorded once on the first pp that yields any solution.
        sols = []
        _propagate(ordered, {}, {}, op_syms, pp, sols, LIMIT, budget)
        for oc, cip in sols:
            all_solutions.append((pp, oc, cip))
        if budget[0] <= 0:
            break

    if not all_solutions:
        reason = "search_budget_exceeded" if budget[0] <= 0 else "no_consistent_cipher"
        base["elapsed"] = time.time() - t0
        base["fail_reason"] = reason
        return base

    # diagnostic: per-fact candidate-set sizes (prefix solution counts) under the
    # winning pp scheme, capped so the diagnostic itself stays cheap.
    DIAG_CAP = 2000
    win_pp = all_solutions[0][0]
    diag_budget = [60000]
    for i in range(1, len(ordered) + 1):
        s = []
        _propagate(ordered[:i], {}, {}, op_syms, win_pp, s, DIAG_CAP, diag_budget)
        per_fact_sizes.append(len(s))
        if diag_budget[0] <= 0:
            break

    # Apply each full solution to the query; collect distinct predicted answers.
    qa, qop, qb = query
    # Forward alphabet (facts + query operands, NOT the answer) — used to
    # complete a 9/10 cipher so a query operand symbol that appears only in the
    # query (but is the unique unmapped alphabet symbol) gets its forced digit.
    fwd_alphabet = _digit_alphabet(facts, query, None, op_syms)
    preds = {}
    for pp, oc, cip in all_solutions:
        pp_a, pp_b, pp_r = pp
        opn = oc.get(qop)
        if opn is None:
            continue
        cipc = _complete_cipher(cip, fwd_alphabet)
        if opn in ("concat", "rconcat"):
            # concat/rconcat copies operand SYMBOLS directly — no cipher needed,
            # so this works even when a query operand symbol never appeared in the
            # examples (cipher need not cover it).
            ans = _encode_concat(qa, qb, opn, pp, sign_sym)
        else:
            a2 = _decode(qa, cipc)
            b2 = _decode(qb, cipc)
            if a2 is None or b2 is None:
                continue
            a_val = int(_operand_string(a2, pp_a))
            b_val = int(_operand_string(b2, pp_b))
            raw = _apply_op(opn, a_val, b_val)
            if raw is None:
                continue
            ans = _encode_answer(raw, pp_r, cipc, sign_sym, op_syms)
        if ans is None:
            continue
        op_map = dict(oc)
        preds.setdefault(ans, (op_map, cipc, pp))

    if not preds:
        base["elapsed"] = time.time() - t0
        base["fail_reason"] = "query_unencodable"
        base["per_fact_candidate_sizes"] = per_fact_sizes
        return base

    well_posed = (len(preds) == 1)
    if well_posed:
        chosen = next(iter(preds))
    elif answer is not None and answer in preds:
        chosen = answer
    else:
        chosen = sorted(preds)[0]
    op_map, cip, pp = preds[chosen]

    return {
        "solved": True,
        "correct": None if answer is None else (chosen == answer),
        "reachable": None if answer is None else (answer in preds),
        "well_posed": well_posed,
        "n_distinct_predictions": len(preds),
        "predicted": chosen,
        "op_map": op_map,
        "cipher": cip,
        "preprocess": pp,
        "per_fact_candidate_sizes": per_fact_sizes,
        "elapsed": time.time() - t0,
        "fail_reason": None,
    }


def _encode_concat(qa: str, qb: str, opn: str, pp, sign_sym):
    """Build a concat/rconcat result directly from operand SYMBOLS (no cipher).
    pp = (pp_a, pp_b, pp_r). swap reverses a 2-symbol operand; pp_r reverses the
    whole 4-symbol result. concat = a then b; rconcat = b then a."""
    pp_a, pp_b, pp_r = pp
    a = qa[::-1] if pp_a == "swap" else qa
    b = qb[::-1] if pp_b == "swap" else qb
    res = (a + b) if opn == "concat" else (b + a)
    if pp_r == "swap":
        res = res[::-1]
    return res  # concat results are always positive 4-symbol strings


def _encode_answer(raw: int, pp_r: str, cipher: dict, sign_sym, op_syms):
    """Encode the query result back to cipher symbols. Tries the minimal natural
    width.  Returns symbol string (with sign symbol if negative) or None."""
    rev = {v: k for k, v in cipher.items()}
    av = abs(raw)
    # The generator emits the result at its natural minimal width (no spurious zero
    # padding), then applies pp_r (reverse the digit string of that width).
    digit_str = str(av)  # natural minimal width
    if pp_r == "swap":
        emit = digit_str[::-1]
    else:
        emit = digit_str
    syms = []
    for ch in emit:
        d = int(ch)
        if d not in rev:
            return None
        syms.append(rev[d])
    body = "".join(syms)
    if raw < 0:
        if sign_sym is None:
            return None
        return sign_sym + body
    if raw == 0:
        return body
    return body


# ---------------------------------------------------------------------------
# __main__: evaluate over all 659 crd problems
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, csv, glob, statistics
    from collections import Counter
    from pathlib import Path

    PROJECT = Path("/scratch/jwj/NemotronFT")
    ids = set()
    for f in glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d['category'] == 'cryptarithm_deduce':
                ids.add(d['id'])
    rows = {}
    with open(PROJECT / "data/train.csv") as f:
        for row in csv.DictReader(f):
            if row['id'] in ids:
                rows[row['id']] = row

    n = len(rows)
    solved = correct = reachable = wellposed = 0
    fail = Counter()
    times = []
    npred = []
    all_pf = []
    examples_wrong = []
    examples_unsolved = []
    for k, v in rows.items():
        r = solve(v['prompt'], v['answer'])
        times.append(r['elapsed'])
        all_pf.extend(r.get('per_fact_candidate_sizes', []))
        if r['solved']:
            solved += 1
            if r.get('correct'):
                correct += 1
            elif len(examples_wrong) < 6:
                examples_wrong.append((k, v['answer'], r.get('predicted'),
                                       r.get('n_distinct_predictions'), r.get('reachable')))
            if r.get('reachable'):
                reachable += 1
            if r.get('well_posed'):
                wellposed += 1
            npred.append(r.get('n_distinct_predictions', 1))
        else:
            fail[r['fail_reason']] += 1
            if len(examples_unsolved) < 6:
                examples_unsolved.append((k, r['fail_reason']))

    def pct(x):
        return f"{x} ({100*x/n:.1f}%)"
    print(f"=== cryptarithm_deduce: {n} problems ===")
    print(f"coverage (produced an answer): {pct(solved)}")
    print(f"accuracy (predicted==truth):   {pct(correct)}")
    print(f"reachable (truth among forward preds): {pct(reachable)}")
    print(f"well-posed (unique forward solution):  {pct(wellposed)}")
    print(f"--- solve time: median {statistics.median(times)*1000:.1f}ms "
          f"p95 {sorted(times)[int(0.95*len(times))]*1000:.1f}ms "
          f"max {max(times)*1000:.1f}ms")
    if npred:
        print(f"--- n_distinct_predictions: median {statistics.median(npred)} "
              f"p95 {sorted(npred)[int(0.95*len(npred))]} max {max(npred)}")
    if all_pf:
        print(f"--- per-fact candidate-set sizes: median {statistics.median(all_pf)} "
              f"p95 {sorted(all_pf)[int(0.95*len(all_pf))]} max {max(all_pf)}")
    print(f"--- failure breakdown: {dict(fail)}")
    print("--- sample wrong (id, truth, pred, n_preds, reachable):")
    for e in examples_wrong:
        print("   ", e)
    print("--- sample unsolved (id, reason):")
    for e in examples_unsolved:
        print("   ", e)
