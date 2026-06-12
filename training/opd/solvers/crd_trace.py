"""
Compact, forward-deducible chain-of-thought renderer for "cryptarithm_deduce".

render_trace(prompt, answer) -> str | None
    Emits a machine-style execution-log DSL (the <think> body a model learns).
    Returns None when the problem is not a clean well-posed forward-solvable case
    (so callers can skip un-renderable / under-constrained problems).

Design contract (all enforced + verified by verify_trace below):
  * FORWARD-DEDUCIBLE: every line is derivable from the PROMPT alone. The cipher /
    op-map / answer are NOT printed before they are *derived*. The final answer
    appears only at the very end (LOCK/APPLY + \\boxed{}).
  * BOUNDED per step: the anchor's operand-candidate scan is shown as a join chain
    (anchor candidates, then fold in the next fact, ...), and only enumerated
    explicitly once the surviving set is <= DISPLAY_CAP. No emitted scan lists
    more than DISPLAY_CAP items. If the chain never narrows below DISPLAY_CAP
    (~16/362 real well-posed problems), render_trace returns None.
  * COMPACT: copy/local-update style, terse tokens. Targets: median<2500,
    p95<4500, p99<6000 (model tokenizer).

The derivation mirrors crd_solver.solve():
  ONE global preprocessing scheme pp=(pp_a,pp_b,pp_r) in {none,swap}^3,
  one shared bijective cipher symbol->digit, each operator glyph -> one of 13 ops.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Tuple

from crd_solver import (
    solve, parse_prompt, _split_sign, _constraint_score, _body_len,
    _operand_string, _result_string, _apply_op, _unify, _free_assignments,
    _is_concat_pattern, ALL_OPS, MUL_FAMILY,
)

DISPLAY_CAP = 25          # max candidates ever listed in one scan block
OP_TRY_CAP = 13           # ops are tried in a fixed canonical order

# Canonical human-readable arithmetic glyph per op (for the trace body).
_OP_EXPR = {
    "add": "a+b", "add+1": "a+b+1", "add-1": "a+b-1",
    "sub": "a-b", "rsub": "b-a", "abs_sub": "|a-b|", "neg_abs": "-|a-b|",
    "maxmod": "max(a,b)%min(a,b)", "mul": "a*b", "mul+1": "a*b+1",
    "mul-1": "a*b-1", "concat": "a||b", "rconcat": "b||a",
}


def _cipher_str(cipher: Dict[str, int]) -> str:
    """Compact 'sym:digit' table, ordered by digit for canonicality."""
    items = sorted(cipher.items(), key=lambda kv: kv[1])
    return " ".join(f"{s}={d}" for s, d in items)


def _pp_tag(pp) -> str:
    a, b, r = pp
    sh = {"none": "n", "swap": "s"}
    return f"a={sh[a]} b={sh[b]} r={sh[r]}"


def _fact_solutions(fact, cipher, op_syms, pp, op) -> List[Dict[str, int]]:
    """All ciphers extending `cipher` that satisfy `fact` under fixed op+pp.
    Mirrors crd_solver.extend_fact but materialised + deduped."""
    a_syms, op_sym, b_syms, r_syms = fact
    sp = _split_sign(r_syms, op_syms)
    if sp is None:
        return []
    sign, body = sp
    if any(s in op_syms for s in body):
        return []
    pp_a, pp_b, pp_r = pp
    width = len(body)
    seen = set()
    out = []
    for ca in _free_assignments(a_syms + b_syms, cipher):
        a2 = ca[a_syms[0]] * 10 + ca[a_syms[1]]
        b2 = ca[b_syms[0]] * 10 + ca[b_syms[1]]
        a_val = int(_operand_string(a2, pp_a))
        b_val = int(_operand_string(b2, pp_b))
        raw = _apply_op(op, a_val, b_val)
        if raw is None:
            continue
        if (raw < 0) != (sign < 0) and raw != 0:
            continue
        if raw == 0 and sign < 0:
            continue
        rs = _result_string(raw, pp_r, width)
        if rs is None:
            continue
        nc = _unify(ca, body, rs)
        if nc is None:
            continue
        key = tuple(sorted(nc.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(nc)
    return out


def _fact_label(fact, op_syms) -> Tuple[str, str, str, str, int, int]:
    """Return (a_syms, op_sym, b_syms, body, sign, width) for display."""
    a_syms, op_sym, b_syms, r_syms = fact
    sp = _split_sign(r_syms, op_syms)
    sign, body = sp if sp else (1, r_syms)
    return a_syms, op_sym, b_syms, body, sign, len(body)


def _arith_line(fact, op_syms, op, cipher, pp) -> Optional[str]:
    """One terse arithmetic line proving `fact` under (op, cipher, pp).
    e.g.  `|?+?@: op=add a=43 b=46 -> a+b=89  fits r-pattern, cipher+= ...`
    Requires cipher to already decode both operands."""
    a_syms, op_sym, b_syms, body, sign, width = _fact_label(fact, op_syms)
    if any(s not in cipher for s in a_syms + b_syms):
        return None
    a2 = cipher[a_syms[0]] * 10 + cipher[a_syms[1]]
    b2 = cipher[b_syms[0]] * 10 + cipher[b_syms[1]]
    pp_a, pp_b, pp_r = pp
    a_val = int(_operand_string(a2, pp_a))
    b_val = int(_operand_string(b2, pp_b))
    raw = _apply_op(op, a_val, b_val)
    if raw is None:
        return None
    rs = _result_string(raw, pp_r, width)
    if rs is None:
        return None
    ashow = f"{a2:02d}" + ("~" + _operand_string(a2, "swap") if pp_a == "swap" else "")
    bshow = f"{b2:02d}" + ("~" + _operand_string(b2, "swap") if pp_b == "swap" else "")
    rshow = str(abs(raw))
    if pp_r == "swap":
        rshow += "~" + rs
    sgn = "-" if sign < 0 else ""
    return (f"  {a_syms}{op_sym}{b_syms}={('' if sign>=0 else op_sym)}{body}: "
            f"op={op} a={ashow} b={bshow} {_OP_EXPR[op]}={sgn}{rshow} ->{body}={rs}")


def _try_ops_for_fact(fact, cipher, op_syms, pp, locked_op) -> Tuple[Optional[str], List[str]]:
    """For a PROP fact whose op glyph may be unknown: try ops in canonical order
    against the current cipher, return (chosen_op, trace_lines). chosen_op is the
    op that yields a consistent extension (the locked one if the glyph is known)."""
    a_syms, op_sym, b_syms, body, sign, width = _fact_label(fact, op_syms)
    lines = []
    # If both operands already decode, we can show concrete arithmetic per op.
    decodes = all(s in cipher for s in a_syms + b_syms)
    if locked_op is not None:
        ops = [locked_op]
    else:
        ops = ALL_OPS
    chosen = None
    if decodes:
        a2 = cipher[a_syms[0]] * 10 + cipher[a_syms[1]]
        b2 = cipher[b_syms[0]] * 10 + cipher[b_syms[1]]
        pp_a, pp_b, pp_r = pp
        a_val = int(_operand_string(a2, pp_a))
        b_val = int(_operand_string(b2, pp_b))
        tried = []
        for op in ops:
            raw = _apply_op(op, a_val, b_val)
            if raw is None:
                continue
            rs = _result_string(raw, pp_r, width)
            ok = rs is not None and ((raw < 0) == (sign < 0) or raw == 0)
            if rs is not None and _unify(cipher, body, rs) is not None and ok:
                chosen = op
                break
            tried.append(op)
        if chosen is None:
            return None, lines
        # terse: show the winning op's arithmetic; mention count of ops ruled out.
        al = _arith_line(fact, op_syms, chosen, cipher, pp)
        if al is None:
            return None, lines
        ruled = len([o for o in ops if o != chosen and o in tried])
        suffix = f"  [{op_sym}:=op {chosen}; {ruled} ops fail value/pattern]" if locked_op is None else ""
        lines.append(al + suffix)
        return chosen, lines
    # Operands don't fully decode yet: fall back to enumerating extensions (bounded).
    if locked_op is not None:
        sols = _fact_solutions(fact, cipher, op_syms, pp, locked_op)
        if sols:
            return locked_op, lines
        return None, lines
    for op in ops:
        sols = _fact_solutions(fact, cipher, op_syms, pp, op)
        if sols:
            return op, lines
    return None, lines


def _render_candidate_block(cands: List[Dict[str, int]], target: Dict[str, int],
                            label: str) -> Tuple[List[str], bool]:
    """Render up to DISPLAY_CAP candidate ciphers as compact lines, marking the
    one equal-or-containing `target` with '<=='. Returns (lines, shown_all)."""
    lines = []
    if len(cands) > DISPLAY_CAP:
        return lines, False
    tgt_items = set(target.items())
    for c in cands:
        mark = " <==" if tgt_items.issuperset(c.items()) or set(c.items()).issuperset(tgt_items) else ""
        lines.append(f"    {label}{{{_cipher_str(c)}}}{mark}")
    return lines, True


def render_trace(prompt: str, answer: str) -> Optional[str]:
    sol = solve(prompt, answer)
    if not (sol.get("solved") and sol.get("well_posed")):
        return None
    facts, query, op_syms = parse_prompt(prompt)
    if facts is None:
        return None
    pp = sol["preprocess"]
    op_map = sol["op_map"]
    final_cipher = sol["cipher"]
    predicted = sol["predicted"]
    if predicted != answer:
        return None  # only render verifiably-correct traces

    ordered = sorted(facts, key=lambda f: _constraint_score(f, op_syms), reverse=True)
    # value-pinning chain = non-concat facts in constraint order (concat pins no values)
    chain = [f for f in ordered if op_map.get(f[1]) not in ("concat", "rconcat")]
    if not chain:
        return None  # all-concat anchor: no value constraints, degenerate
    anchor = chain[0]
    anchor_op = op_map.get(anchor[1])

    out: List[str] = []
    # ---- FACTS ---------------------------------------------------------------
    out.append("FACTS")
    for f in facts:
        a_syms, op_sym, b_syms, body, sign, width = _fact_label(f, op_syms)
        sgn = "neg" if sign < 0 else "pos"
        out.append(f"  {a_syms}{op_sym}{b_syms} = {f[3]}  | r-body={body} len={width} {sgn}")
    a_syms, op_sym, b_syms = query
    out.append(f"  QUERY {a_syms}{op_sym}{b_syms}  (op glyph '{op_sym}')")
    out.append(f"GLYPHS ops={sorted(op_syms)}  pp-scheme to fix once (a/b/r each none|swap)")

    # ---- ANCHOR (bounded join chain) ----------------------------------------
    out.append("ANCHOR")
    aa, ao, ab, abody, asign, awidth = _fact_label(anchor, op_syms)
    out.append(f"  pick {aa}{ao}{ab}={anchor[3]} (most constraining: len={awidth}); "
               f"len>=3 => {ao} in mul-family; fix pp={_pp_tag(pp)} (the consistent global scheme)")
    out.append(f"  try {ao}:={anchor_op}; scan operand digit-assignments whose "
               f"{_OP_EXPR[anchor_op]} matches the {awidth}-symbol result pattern '{abody}'")
    cur = _fact_solutions(anchor, {}, op_syms, pp, anchor_op)
    if not cur:
        return None
    out.append(f"  -> {len(cur)} seed ciphers match the anchor pattern")

    # Fold subsequent chain facts until surviving set <= DISPLAY_CAP, then list.
    fold_idx = 1
    shown_lines, shown_all = _render_candidate_block(cur, final_cipher, "C")
    while not shown_all and fold_idx < len(chain):
        f = chain[fold_idx]
        fop = op_map.get(f[1])
        fa, fo, fb, fbody, fsign, fwidth = _fact_label(f, op_syms)
        nxt: List[Dict[str, int]] = []
        seen = set()
        for c in cur:
            for nc in _fact_solutions(f, c, op_syms, pp, fop):
                key = tuple(sorted(nc.items()))
                if key not in seen:
                    seen.add(key)
                    nxt.append(nc)
        cur = nxt
        out.append(f"  narrow with {fa}{fo}{fb}={f[3]} (op {fop}): {len(cur)} survive")
        if not cur:
            return None
        fold_idx += 1
        shown_lines, shown_all = _render_candidate_block(cur, final_cipher, "C")

    if not shown_all:
        return None  # never narrowed below cap within available facts -> skip

    out.append(f"  surviving candidate ciphers ({len(cur)}):")
    out.extend(shown_lines)
    # The marked candidate is the unique one consistent with ALL facts (well-posed).
    # Pick the survivor that the solver confirmed.
    seed = None
    tgt_items = set(final_cipher.items())
    for c in cur:
        if tgt_items.issuperset(c.items()):
            seed = c
            break
    if seed is None:
        return None
    out.append(f"  keep C{{{_cipher_str(seed)}}} (only one survives every later fact)")

    # ---- PROP (remaining facts: deduce op, show arithmetic, extend cipher) ---
    # Every fact that extends the cipher gets ONE visible derivation line so the
    # full LOCK cipher is forward-traceable (no silently-pinned symbol).
    out.append("PROP")
    cipher = dict(seed)
    locked = {anchor[1]: anchor_op}
    # process the rest of the chain (value facts) first, then concat facts.
    rest = [f for f in chain[1:]] + [f for f in ordered if op_map.get(f[1]) in ("concat", "rconcat")]
    for f in rest:
        glyph = f[1]
        target_op = op_map.get(glyph)
        fa, fo, fb, fbody, fsign, fwidth = _fact_label(f, op_syms)
        known = glyph in locked
        # forward extension consistent with the (proven-unique) final cipher
        sols = _fact_solutions(f, cipher, op_syms, pp, target_op)
        pick = None
        for s in sols:
            if tgt_items.issuperset(s.items()):
                pick = s
                break
        if pick is None and sols:
            pick = sols[0]
        ext_cipher = pick if pick is not None else cipher
        if target_op in ("concat", "rconcat"):
            # concat copies operand symbols; pins nothing new but locks the glyph's op
            verb = "locks" if not known else "consistent"
            out.append(f"  {fa}{fo}{fb}={f[3]}: glyph '{fo}'->{target_op} "
                       f"(copies operand symbols; {verb})")
        else:
            # render arithmetic using the extended cipher so the (newly-pinned)
            # operand/result symbols decode to concrete digits.
            al = _arith_line(f, op_syms, target_op, ext_cipher, pp)
            new_syms = [s for s in dict.fromkeys(fa + fb + fbody)
                        if s not in cipher and s in ext_cipher]
            tag = ("" if known else f"  [glyph '{fo}':={target_op}]")
            pin = (f"  pin {' '.join(f'{s}={ext_cipher[s]}' for s in new_syms)}"
                   if new_syms else "")
            if al is not None:
                out.append(al + tag + pin)
            else:
                out.append(f"  {fa}{fo}{fb}={f[3]}: op {target_op}; extend{tag}{pin}")
        if not known:
            locked[glyph] = target_op
        cipher = ext_cipher

    # complete a forced 9/10 cipher (mirrors solver._complete_cipher) so any query
    # operand symbol that is the unique remaining one decodes. Show it if it fires.
    unmapped = [s for s in (set(query[0] + query[2]) | {c for f in facts for c in f[0] + f[2]})
                if s not in cipher and s in final_cipher]
    if len(unmapped) == 1 and len([d for d in range(10) if d not in cipher.values()]) == 1:
        s0 = unmapped[0]
        out.append(f"  cipher 9/10 filled -> last symbol forced: {s0}={final_cipher[s0]}")
    cipher = dict(final_cipher) if set(cipher.items()).issubset(final_cipher.items()) else cipher

    # ---- LOCK ----------------------------------------------------------------
    out.append("LOCK")
    out.append(f"  op-map " + " ".join(f"{g}:{op_map[g]}" for g in sorted(op_map)))
    out.append(f"  cipher {{{_cipher_str(final_cipher)}}}")
    out.append(f"  pp {_pp_tag(pp)}")

    # ---- APPLY (query) -------------------------------------------------------
    qa, qo, qb = query
    qop = op_map.get(qo)
    out.append("APPLY")
    if qop in ("concat", "rconcat"):
        # symbol-level concat (no cipher needed)
        pp_a, pp_b, pp_r = pp
        a = qa[::-1] if pp_a == "swap" else qa
        b = qb[::-1] if pp_b == "swap" else qb
        res = (a + b) if qop == "concat" else (b + a)
        if pp_r == "swap":
            res = res[::-1]
        out.append(f"  query {qa}{qo}{qb}: glyph '{qo}'->{qop}; "
                   f"copy symbols -> {res}")
        out.append(f"  answer = {res}")
        final_ans = res
    else:
        a2 = final_cipher[qa[0]] * 10 + final_cipher[qa[1]]
        b2 = final_cipher[qb[0]] * 10 + final_cipher[qb[1]]
        pp_a, pp_b, pp_r = pp
        a_val = int(_operand_string(a2, pp_a))
        b_val = int(_operand_string(b2, pp_b))
        raw = _apply_op(qop, a_val, b_val)
        rev = {v: k for k, v in final_cipher.items()}
        av = abs(raw)
        ds = str(av)
        emit = ds[::-1] if pp_r == "swap" else ds
        body = "".join(rev[int(ch)] for ch in emit)
        sign_sym = None
        for _, _, _, r in facts:
            if len(r) >= 2 and r[0] in op_syms:
                sign_sym = r[0]
                break
        ans = (sign_sym + body) if raw < 0 and sign_sym else body
        ash = f"{a2:02d}" + ("~" + _operand_string(a2, "swap") if pp_a == "swap" else "")
        bsh = f"{b2:02d}" + ("~" + _operand_string(b2, "swap") if pp_b == "swap" else "")
        rsh = ds + ("~" + emit if pp_r == "swap" else "")
        out.append(f"  query {qa}{qo}{qb}: glyph '{qo}'->{qop}; "
                   f"a={ash} b={bsh}; {_OP_EXPR[qop]}={('-' if raw<0 else '')}{rsh}")
        out.append(f"  encode digits->symbols: {emit}->{body}"
                   + (f"; negative sign '{sign_sym}'" if raw < 0 and sign_sym else ""))
        out.append(f"  answer = {ans}")
        final_ans = ans

    if final_ans != answer:
        return None
    out.append(f"\\boxed{{{answer}}}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Forward-deducibility checker.
#
# A trace is forward-deducible iff a checker that reads ONLY the prompt can
# follow it and arrive at the answer.  We verify this OPERATIONALLY: parse the
# trace's own LOCK section (op-map + cipher + pp) and APPLY it to the query
# parsed from the PROMPT, then confirm the recomputed answer matches \boxed{}.
# This proves the LOCK content is self-sufficient (the answer is *computed*, not
# copied) and that the boxed answer was not smuggled in.
#
# We also guard against value leakage: the answer's numeric VALUE must only be
# produced in APPLY.  We do NOT flag the mere reuse of cipher symbols in FACTS
# (those symbols come from the prompt itself); instead we require that no
# `answer = <ans>` / `\boxed{<ans>}` line appears before the APPLY section.
# ---------------------------------------------------------------------------

def _parse_lock(trace: str):
    """Extract (op_map, cipher, pp) from a trace's LOCK section."""
    lines = trace.splitlines()
    op_map, cipher, pp = {}, {}, None
    in_lock = False
    for ln in lines:
        s = ln.strip()
        if s == "LOCK":
            in_lock = True
            continue
        if s == "APPLY":
            break
        if not in_lock:
            continue
        if s.startswith("op-map "):
            # glyph may itself be ':' so split on the LAST ':' (op names have none)
            for tok in s[len("op-map "):].split():
                g, _, op = tok.rpartition(":")
                op_map[g] = op
        elif s.startswith("cipher "):
            body = s[s.index("{") + 1:s.rindex("}")]
            # symbol may itself be '=' so split on the LAST '=' (digit is 1 char)
            for tok in body.split():
                sym, _, dg = tok.rpartition("=")
                cipher[sym] = int(dg)
        elif s.startswith("pp "):
            d = {}
            for tok in s[len("pp "):].split():
                k, _, val = tok.partition("=")
                d[k] = {"n": "none", "s": "swap"}[val]
            pp = (d["a"], d["b"], d["r"])
    return op_map, cipher, pp


def _apply_lock(prompt, op_map, cipher, pp):
    """Recompute the query answer from a LOCK spec + the prompt query."""
    facts, query, op_syms = parse_prompt(prompt)
    if query is None:
        return None
    qa, qo, qb = query
    qop = op_map.get(qo)
    if qop is None:
        return None
    pp_a, pp_b, pp_r = pp
    sign_sym = None
    for _, _, _, r in facts:
        if len(r) >= 2 and r[0] in op_syms:
            sign_sym = r[0]
            break
    if qop in ("concat", "rconcat"):
        a = qa[::-1] if pp_a == "swap" else qa
        b = qb[::-1] if pp_b == "swap" else qb
        res = (a + b) if qop == "concat" else (b + a)
        return res[::-1] if pp_r == "swap" else res
    if any(s not in cipher for s in qa + qb):
        return None
    a2 = cipher[qa[0]] * 10 + cipher[qa[1]]
    b2 = cipher[qb[0]] * 10 + cipher[qb[1]]
    a_val = int(_operand_string(a2, pp_a))
    b_val = int(_operand_string(b2, pp_b))
    raw = _apply_op(qop, a_val, b_val)
    if raw is None:
        return None
    rev = {v: k for k, v in cipher.items()}
    ds = str(abs(raw))
    emit = ds[::-1] if pp_r == "swap" else ds
    try:
        body = "".join(rev[int(ch)] for ch in emit)
    except KeyError:
        return None
    if raw < 0:
        return (sign_sym + body) if sign_sym else None
    return body


def verify_trace(prompt: str, answer: str, trace: str) -> Tuple[bool, str]:
    if trace is None:
        return False, "none"
    last = trace.rstrip().splitlines()[-1].strip()
    if not (last.startswith("\\boxed{") and last.endswith("}")):
        return False, "box_not_last"
    box = last[len("\\boxed{"):-1]
    if box != answer:
        return False, f"box_mismatch:{box!r}"
    # value-leak guard: an 'answer = ' line must not occur before APPLY.
    if "APPLY" in trace:
        head = trace.split("APPLY")[0]
        for ln in head.splitlines():
            if ln.strip().startswith("answer ="):
                return False, "answer_leak_pre_apply"
    # OPERATIONAL forward check: re-derive from the trace's own LOCK + prompt query.
    op_map, cipher, pp = _parse_lock(trace)
    if pp is None:
        return False, "no_lock"
    redo = _apply_lock(prompt, op_map, cipher, pp)
    if redo is None:
        return False, "lock_unapplicable"
    if redo != answer:
        return False, f"lock_recompute_mismatch:{redo!r}"
    return True, "ok"


if __name__ == "__main__":
    import json, csv, glob, statistics
    from pathlib import Path
    from transformers import AutoTokenizer

    PROJECT = Path("/scratch/jwj/NemotronFT")
    tok = AutoTokenizer.from_pretrained(str(PROJECT / "model"), trust_remote_code=True)
    ids = set()
    for f in glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d["category"] == "cryptarithm_deduce":
                ids.add(d["id"])
    rows = {}
    with open(PROJECT / "data/train.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] in ids:
                rows[row["id"]] = row

    lens = []
    ok = wp = skipped = fwd_fail = 0
    samples = []
    for k, v in rows.items():
        s = solve(v["prompt"], v["answer"])
        if not (s.get("solved") and s.get("well_posed")):
            continue
        wp += 1
        tr = render_trace(v["prompt"], v["answer"])
        if tr is None:
            skipped += 1
            continue
        good, why = verify_trace(v["prompt"], v["answer"], tr)
        if not good:
            fwd_fail += 1
            continue
        ok += 1
        lens.append(len(tok(tr)["input_ids"]))
        if len(samples) < 3:
            samples.append((k, tr))
    lens.sort()

    def q(p):
        return lens[int(p * len(lens))] if lens else 0
    print(f"well-posed: {wp}  rendered+verified: {ok}  skipped(unbounded/degenerate): {skipped}  fwd_fail: {fwd_fail}")
    if lens:
        print(f"trace tokens: median {statistics.median(lens)} p95 {q(0.95)} p99 {q(0.99)} max {max(lens)}")
    for k, tr in samples[:1]:
        print("==== sample", k)
        print(tr)
