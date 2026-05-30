"""Fork of resource/winning-solution/investigators/cryptarithm_deduce.py with:

  1. Extended OPS list (19 ops vs Huikang's 5)
  2. Configurable base (10, 9, 11, ...)
  3. Optional post-reverse of result digits

Same algorithm: DFS over (digit-mapping, op-assignment) jointly, with bijective
digit-char mapping. Each op symbol in the puzzle is independently assigned to
one of the OPS, learned from the example(s) using that symbol.

This is the CORRECT mental model: a puzzle has multiple op symbols, each with
its own (a, b) → result rule.  Huikang's `op_assign` dict does this; we keep
it and just extend the OPS catalog.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional


# -------- Extended operations (a, b are integers in [0, base^2)) -----------


def _to_2digit(n: int, base: int) -> tuple[int, int]:
    return (n // base, n % base)


def num_to_digits(n: int, base: int) -> tuple[int, ...]:
    if n < 0:
        return ()
    if n == 0:
        return (0,)
    out: list[int] = []
    while n > 0:
        out.append(n % base)
        n //= base
    return tuple(reversed(out))


def _op_add(a, b, base):       return a + b
def _op_sub_ab(a, b, base):    return a - b if a >= b else None
def _op_sub_ba(a, b, base):    return b - a if b >= a else None
def _op_abs_diff(a, b, base):  return abs(a - b)
def _op_mul(a, b, base):       return a * b
def _op_concat(a, b, base):    return a * (base * base) + b
def _op_rev_concat(a, b, base): return b * (base * base) + a
def _op_div_ab(a, b, base):    return a // b if b > 0 and a % b == 0 else None
def _op_div_ba(a, b, base):    return b // a if a > 0 and b % a == 0 else None
def _op_mod_ab(a, b, base):    return a % b if b > 0 else None
def _op_mod_ba(a, b, base):    return b % a if a > 0 else None


def _op_digit_add(a, b, base):
    a0, a1 = _to_2digit(a, base)
    b0, b1 = _to_2digit(b, base)
    return ((a0 + b0) % base) * base + ((a1 + b1) % base)


def _op_digit_sub(a, b, base):
    a0, a1 = _to_2digit(a, base)
    b0, b1 = _to_2digit(b, base)
    return ((a0 - b0) % base) * base + ((a1 - b1) % base)


def _op_digit_mul(a, b, base):
    a0, a1 = _to_2digit(a, base)
    b0, b1 = _to_2digit(b, base)
    return ((a0 * b0) % base) * base + ((a1 * b1) % base)


def _op_digit_xor(a, b, base):
    if base != 10 and base != 8 and base != 16:
        return None
    a0, a1 = _to_2digit(a, base)
    b0, b1 = _to_2digit(b, base)
    if a0 >= base or a1 >= base or b0 >= base or b1 >= base:
        return None
    return (a0 ^ b0) * base + (a1 ^ b1)


def _op_digit_max(a, b, base):
    a0, a1 = _to_2digit(a, base)
    b0, b1 = _to_2digit(b, base)
    return max(a0, b0) * base + max(a1, b1)


def _op_digit_min(a, b, base):
    a0, a1 = _to_2digit(a, base)
    b0, b1 = _to_2digit(b, base)
    return min(a0, b0) * base + min(a1, b1)


def _op_add_plus_1(a, b, base):     return a + b + 1
def _op_add_minus_1(a, b, base):    return a + b - 1
def _op_mul_plus_1(a, b, base):     return a * b + 1
def _op_mul_minus_1(a, b, base):    return a * b - 1


OPS_BY_NAME = {
    "add":            _op_add,
    "sub_ab":         _op_sub_ab,
    "sub_ba":         _op_sub_ba,
    "abs_diff":       _op_abs_diff,
    "mul":            _op_mul,
    "concat":         _op_concat,
    "rev_concat":     _op_rev_concat,
    "div_ab":         _op_div_ab,
    "div_ba":         _op_div_ba,
    "mod_ab":         _op_mod_ab,
    "mod_ba":         _op_mod_ba,
    "digit_add":      _op_digit_add,
    "digit_sub":      _op_digit_sub,
    "digit_mul":      _op_digit_mul,
    "digit_xor":      _op_digit_xor,
    "digit_max":      _op_digit_max,
    "digit_min":      _op_digit_min,
    "add_plus_1":     _op_add_plus_1,
    "add_minus_1":    _op_add_minus_1,
    "mul_plus_1":     _op_mul_plus_1,
    "mul_minus_1":    _op_mul_minus_1,
}
OP_NAMES = list(OPS_BY_NAME.keys())


# -------- Solver (forked from Huikang) -------------------------------------


def is_trivial_concat(ex, base: int) -> Optional[str]:
    """If the example is pure forward / reverse concat at the symbol level."""
    s0, s1, _, s3, s4, rsyms = ex
    if rsyms == (s0, s1, s3, s4):
        return "concat"
    if rsyms == (s3, s4, s0, s1):
        return "rev_concat"
    return None


class ExtendedSolver:
    def __init__(self, examples, query, base: int = 10, post_reverse: bool = False, unique: bool = True):
        self.examples = examples
        self.query = query
        self.base = base
        self.post_reverse = post_reverse
        self.unique = unique
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.op_assign: dict[str, str] = {}
        self.answers: Counter = Counter()
        self.answer_info: dict[str, tuple[dict, dict]] = {}
        self.max_solutions = 200
        self.branch_count = 0
        self.branch_cap = 1_500_000

    def solve(self):
        try:
            self._process(0)
        except _BranchCap:
            pass
        if self.answers:
            best, best_count = self.answers.most_common(1)[0]
            total = sum(self.answers.values())
            if not self.unique and total > 1 and best_count < total * 0.3:
                return None, ({}, {})
            return best, self.answer_info.get(best, ({}, {}))
        return None, ({}, {})

    def _process(self, idx: int):
        self.branch_count += 1
        if self.branch_count > self.branch_cap:
            raise _BranchCap()
        if len(self.answers) >= self.max_solutions:
            return
        if idx == len(self.examples):
            self._compute_query()
            return

        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)
        base = self.base

        # Filter feasible ops by output length
        feasible_ops = []
        for name in OP_NAMES:
            # rough feasibility filter on max result for 2-digit base-N operands
            max_op = base * base - 1  # 99 for base 10
            if name in ("add", "add_plus_1", "add_minus_1"):
                max_result = 2 * max_op + 1
            elif name in ("sub_ab", "sub_ba", "abs_diff"):
                max_result = max_op
            elif name in ("mul", "mul_plus_1", "mul_minus_1"):
                max_result = max_op * max_op + 1
            elif name in ("concat", "rev_concat"):
                if rlen != 4:
                    continue
                feasible_ops.append(name)
                continue
            elif name in ("div_ab", "div_ba", "mod_ab", "mod_ba"):
                max_result = max_op
            elif name in ("digit_add", "digit_sub", "digit_mul", "digit_xor",
                          "digit_max", "digit_min"):
                # digit-wise produces 2-digit result
                if rlen != 2:
                    continue
                feasible_ops.append(name)
                continue
            else:
                max_result = max_op * max_op
            min_len = 1
            max_len = len(num_to_digits(max_result, base))
            if min_len <= rlen <= max_len:
                feasible_ops.append(name)

        if not feasible_ops:
            return

        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * base + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * base + d4

                        if op_sym in self.op_assign:
                            ops_to_try = [self.op_assign[op_sym]]
                        else:
                            ops_to_try = feasible_ops

                        for op_name in ops_to_try:
                            result_val = OPS_BY_NAME[op_name](lv, rv, base)
                            if result_val is None or result_val < 0:
                                continue
                            if op_name in ("concat", "rev_concat"):
                                # pad to 4 digits in base
                                if result_val >= base ** 4:
                                    continue
                                rd = (result_val // (base ** 3),
                                      (result_val // (base ** 2)) % base,
                                      (result_val // base) % base,
                                      result_val % base)
                            elif op_name in ("digit_add", "digit_sub", "digit_mul",
                                             "digit_xor", "digit_max", "digit_min"):
                                rd = (result_val // base, result_val % base)
                            else:
                                rd = num_to_digits(result_val, base)
                            if self.post_reverse:
                                rd = tuple(reversed(rd))
                            if len(rd) != rlen:
                                continue

                            assigns = []
                            ok = True
                            for rs, rdig in zip(rsyms, rd):
                                ns = self._assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                assigns.append((rs, ns))

                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new:
                                    self.op_assign[op_sym] = op_name
                                self._process(idx + 1)
                                if op_new:
                                    del self.op_assign[op_sym]

                            for rs, ns in reversed(assigns):
                                self._undo(rs, ns)

                            if len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4); self._undo(s3, n3)
                                self._undo(s1, n1); self._undo(s0, n0)
                                return

                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(self.base) if d not in self.used)
        return tuple(range(self.base))

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        if dig >= self.base:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym, was_new):
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self):
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return

        ql = self.mapping[qs0] * self.base + self.mapping[qs1]
        qr = self.mapping[qs3] * self.base + self.mapping[qs4]

        if qop in self.op_assign:
            op_candidates = [self.op_assign[qop]]
        else:
            op_candidates = OP_NAMES

        d2s: dict[int, str] = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s

        for op_name in op_candidates:
            result_val = OPS_BY_NAME[op_name](ql, qr, self.base)
            if result_val is None or result_val < 0:
                continue
            if op_name in ("concat", "rev_concat"):
                if result_val >= self.base ** 4:
                    continue
                rd = (result_val // (self.base ** 3),
                      (result_val // (self.base ** 2)) % self.base,
                      (result_val // self.base) % self.base,
                      result_val % self.base)
            elif op_name in ("digit_add", "digit_sub", "digit_mul",
                             "digit_xor", "digit_max", "digit_min"):
                rd = (result_val // self.base, result_val % self.base)
            else:
                rd = num_to_digits(result_val, self.base)
            if self.post_reverse:
                rd = tuple(reversed(rd))

            parts = []
            ok = True
            for d in rd:
                if d not in d2s:
                    ok = False; break
                parts.append(d2s[d])
            if not ok:
                continue

            ans = "".join(parts)
            self.answers[ans] += 1
            if ans not in self.answer_info:
                op_info = dict(self.op_assign)
                op_info[qop] = op_name
                self.answer_info[ans] = (dict(self.mapping), op_info)


class _BranchCap(Exception):
    pass


def solve_problem(data, base: int = 10, post_reverse: bool = False):
    """Top-level entry. `data` matches Huikang's format.

    Logic mirrors `resource/winning-solution/investigators/cryptarithm_deduce.py`
    exactly, with only the OPS list and base extended. Critical: do NOT seed
    op_assign — let the solver discover concat naturally inside its DSL.
    """
    examples = []
    for e in data["examples"]:
        inp = e["input_value"]
        out = e["output_value"]
        if len(inp) != 5:
            continue
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))

    q = data["question"]
    if len(q) != 5:
        return None, ({}, {})
    query = (q[0], q[1], q[2], q[3], q[4])

    # Identify pure concat operators (results == concat of operands, by symbol)
    concat_ops: dict[str, str] = {}  # op_sym -> "concat" | "rev_concat"
    nonconcat_ops: set[str] = set()
    for ex in examples:
        kind = is_trivial_concat(ex, base)
        if kind is not None:
            # Only record if not already overridden by a non-concat example
            concat_ops.setdefault(ex[2], kind)
        else:
            nonconcat_ops.add(ex[2])
    # Remove any op that ALSO has a non-concat example — those need real solving
    concat_only_ops = {op: kind for op, kind in concat_ops.items() if op not in nonconcat_ops}

    q_op = query[2]

    # If query op is purely concat (no non-concat example with it), return directly
    if q_op in concat_only_ops:
        kind = concat_only_ops[q_op]
        if kind == "concat":
            return query[0] + query[1] + query[3] + query[4], ({}, {q_op: "concat"})
        else:
            return query[3] + query[4] + query[0] + query[1], ({}, {q_op: "rev_concat"})

    # Solve over arith examples (concat-only examples are unconstraining)
    arith_examples = [ex for ex in examples if ex[2] not in concat_only_ops]
    example_op_syms = {ex[2] for ex in examples}

    if arith_examples:
        solver = ExtendedSolver(arith_examples, query, base=base, post_reverse=post_reverse, unique=True)
        ans, info = solver.solve()
        if ans is not None:
            return ans, info

        solver2 = ExtendedSolver(arith_examples, query, base=base, post_reverse=post_reverse, unique=False)
        ans2, info2 = solver2.solve()
        if ans2 is not None:
            return ans2, info2

    # Last resort: if query op never appears in any example, default to concat
    # ("keep both chunks" — Huikang's neutral fallback for cryptarithm_guess).
    # We only do this AFTER the solver fails, so we don't block any solver hits.
    if q_op not in example_op_syms:
        return query[0] + query[1] + query[3] + query[4], ({}, {q_op: "concat_no_op"})

    return None, ({}, {})
