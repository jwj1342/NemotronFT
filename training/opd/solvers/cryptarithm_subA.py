"""Cryptarithm subtype-A solver.

Subtype A = the cryptarithm_guess (or cryptarithm_deduce) puzzles where every
example LHS has the form `AB <op> CD = result` with the operator at position 2
being one of `+-*/`, and the query LHS has the same shape.

Each non-operator character represents a distinct decimal digit 0..9 (so all
LHS/RHS digit-characters are drawn from a single bijective `char -> digit`
mapping for that puzzle).  We solve by constrained backtracking: assign
characters to digits in an order chosen to fail-fast (process the example
whose chars are most determined first; check examples as soon as all of their
chars are assigned).

After finding the unique consistent assignment, compute the query LHS and
encode the integer result back to characters via the inverse mapping.  If the
result contains a digit whose char is not known from the examples, return
`undecodable` rather than guess.

The solver returns a structured trace so a verbalizer can turn it into a CoT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

OPS = set("+-*/")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class Example:
    raw: str            # original line e.g. "'(-^} = }`"
    lhs: str            # 5-char left side
    rhs: str            # variable-length right side
    a: str              # first 2-char operand (chars)
    op: str             # operator
    b: str              # second 2-char operand
    chars: frozenset    # set of digit-chars in this example


@dataclass
class Puzzle:
    examples: list[Example]
    query_lhs: str       # the unsolved 5-char LHS, e.g. "''+`!"
    query_a: str
    query_op: str
    query_b: str


def parse_examples(example_lines: list[str]) -> list[Example]:
    """Note: only position 2 of LHS is the operator; the same symbol at any
    other position (or in RHS) is a digit-character."""
    out = []
    for line in example_lines:
        lhs, rhs = line.split(" = ")
        if len(lhs) != 5 or lhs[2] not in OPS:
            raise ValueError(f"non-subtype-A example: {line!r}")
        a, op, b = lhs[:2], lhs[2], lhs[3:]
        chars = frozenset(list(a) + list(b) + list(rhs))
        out.append(Example(raw=line, lhs=lhs, rhs=rhs, a=a, op=op, b=b, chars=chars))
    return out


def parse_puzzle(prompt: str) -> Puzzle:
    body = prompt.split("Below are a few examples:\n", 1)[1]
    ex_section, query = body.split("\nNow, determine the result for: ", 1)
    examples = parse_examples([ln for ln in ex_section.strip().splitlines() if ln.strip()])
    q = query.strip()
    if len(q) != 5 or q[2] not in OPS:
        raise ValueError(f"non-subtype-A query: {q!r}")
    return Puzzle(examples=examples, query_lhs=q, query_a=q[:2], query_op=q[2], query_b=q[3:])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _to_num(s: str, assign: dict[str, int]) -> int:
    n = 0
    for c in s:
        n = n * 10 + assign[c]
    return n


def _eval_example(ex: Example, assign: dict[str, int]) -> bool:
    a = _to_num(ex.a, assign)
    b = _to_num(ex.b, assign)
    if ex.op == "+":
        res = a + b
    elif ex.op == "-":
        res = a - b
    elif ex.op == "*":
        res = a * b
    elif ex.op == "/":
        if b == 0 or a % b != 0:
            return False
        res = a // b
    else:
        return False
    if res < 0:
        return False
    return _to_num(ex.rhs, assign) == res


@dataclass
class SolveResult:
    ok: bool
    mapping: Optional[dict[str, int]] = None       # char -> digit
    inv: Optional[dict[int, str]] = None           # digit -> char (only for assigned digits)
    num_solutions: int = 0
    reason: str = ""
    # backtracking trace (per-step) for CoT verbalization
    order: list[str] = field(default_factory=list)         # char-assignment order
    branches: list[tuple[str, int, str, str]] = field(default_factory=list)
    # ^ list of (char, digit_tried, outcome, comment)
    # outcome ∈ {"taken", "consistent", "contradict:<example.raw>", "dead-end"}


def _all_assigned(s: str, assign: dict[str, int]) -> bool:
    return all(c in assign for c in s)


def solve(puzzle: Puzzle, record_trace: bool = True, max_solutions: int = 2) -> SolveResult:
    # Collect characters
    all_chars: set[str] = set()
    for ex in puzzle.examples:
        all_chars |= set(ex.chars)
    all_chars |= set(list(puzzle.query_a) + list(puzzle.query_b))
    if len(all_chars) > 10:
        return SolveResult(ok=False, reason=f"too many distinct chars: {len(all_chars)}")

    # Variable ordering: greedy "close the easiest example next".
    # Repeatedly pick the example with the fewest unassigned chars and add
    # those chars (in left-to-right order within the example) to the order.
    chars_order: list[str] = []
    seen: set[str] = set()
    remaining_examples = list(puzzle.examples)
    while remaining_examples:
        # pick example with fewest unassigned chars
        def num_unassigned(ex: Example) -> int:
            return len([c for c in set(ex.a + ex.b + ex.rhs) if c not in seen])
        remaining_examples.sort(key=num_unassigned)
        ex = remaining_examples.pop(0)
        for c in ex.a + ex.b + ex.rhs:
            if c not in seen:
                seen.add(c)
                chars_order.append(c)
    # Add query chars last if not yet ordered
    for c in puzzle.query_a + puzzle.query_b:
        if c not in seen:
            seen.add(c)
            chars_order.append(c)

    chars = chars_order

    solutions: list[dict[str, int]] = []
    used = [False] * 10
    assign: dict[str, int] = {}
    trace_order: list[str] = []
    trace_branches: list[tuple[str, int, str, str]] = []

    # For each example, precompute which char in chars_order is the LAST one
    # to be assigned — that's when we should check this example.
    char_idx = {c: i for i, c in enumerate(chars)}
    example_last_idx: dict[int, list[int]] = {}  # idx -> list of example indices to check after that char
    for ei, ex in enumerate(puzzle.examples):
        ex_chars = set(ex.a + ex.b + ex.rhs)
        last_i = max(char_idx[c] for c in ex_chars)
        example_last_idx.setdefault(last_i, []).append(ei)

    def backtrack(i: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if i == len(chars):
            solutions.append(dict(assign))
            return
        c = chars[i]
        if record_trace and c not in trace_order:
            trace_order.append(c)
        for d in range(10):
            if used[d]:
                continue
            assign[c] = d
            used[d] = True
            # Check any examples that just got fully determined
            fail_ex = None
            for ei in example_last_idx.get(i, ()):
                if not _eval_example(puzzle.examples[ei], assign):
                    fail_ex = puzzle.examples[ei].raw
                    break
            if fail_ex is None:
                if record_trace and len(trace_branches) < 2000:
                    trace_branches.append((c, d, "consistent", ""))
                backtrack(i + 1)
            else:
                if record_trace and len(trace_branches) < 2000:
                    trace_branches.append((c, d, "contradict", fail_ex))
            used[d] = False
            del assign[c]

    backtrack(0)

    if len(solutions) == 0:
        return SolveResult(ok=False, reason="no consistent assignment", order=trace_order, branches=trace_branches)
    if len(solutions) > 1:
        return SolveResult(
            ok=False,
            reason=f"ambiguous: ≥{len(solutions)} assignments",
            num_solutions=len(solutions),
            order=trace_order,
            branches=trace_branches,
        )

    mapping = solutions[0]
    inv = {d: c for c, d in mapping.items()}
    return SolveResult(ok=True, mapping=mapping, inv=inv, num_solutions=1,
                       order=trace_order, branches=trace_branches)


# ---------------------------------------------------------------------------
# Apply to query
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    ok: bool
    a_val: Optional[int] = None
    b_val: Optional[int] = None
    op: str = ""
    result_int: Optional[int] = None
    result_digits: list[int] = field(default_factory=list)
    encoded: Optional[str] = None
    reason: str = ""


def apply_to_query(puzzle: Puzzle, mapping: dict[str, int], inv: dict[int, str]) -> QueryResult:
    try:
        a = _to_num(puzzle.query_a, mapping)
        b = _to_num(puzzle.query_b, mapping)
    except KeyError as e:
        return QueryResult(ok=False, reason=f"query char unmapped: {e}")

    op = puzzle.query_op
    if op == "+":
        r = a + b
    elif op == "-":
        r = a - b
    elif op == "*":
        r = a * b
    elif op == "/":
        if b == 0 or a % b != 0:
            return QueryResult(ok=False, reason="non-exact division", a_val=a, b_val=b, op=op)
        r = a // b
    else:
        return QueryResult(ok=False, reason=f"unknown op {op}")

    if r < 0:
        return QueryResult(ok=False, reason="negative result", a_val=a, b_val=b, op=op, result_int=r)

    digits = [int(d) for d in str(r)]
    encoded_chars = []
    for d in digits:
        if d not in inv:
            return QueryResult(
                ok=False,
                reason=f"result digit {d} has no char in mapping",
                a_val=a, b_val=b, op=op, result_int=r, result_digits=digits,
            )
        encoded_chars.append(inv[d])
    return QueryResult(
        ok=True,
        a_val=a, b_val=b, op=op,
        result_int=r,
        result_digits=digits,
        encoded="".join(encoded_chars),
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def solve_prompt(prompt: str) -> tuple[Optional[str], SolveResult, Optional[QueryResult], Optional[Puzzle]]:
    """Top-level: returns (predicted_answer_str_or_None, solve_result, query_result, puzzle).

    None for predicted_answer means we cannot confidently solve this puzzle.
    """
    try:
        puzzle = parse_puzzle(prompt)
    except ValueError as e:
        return None, SolveResult(ok=False, reason=f"parse: {e}"), None, None

    sr = solve(puzzle, record_trace=True)
    if not sr.ok:
        return None, sr, None, puzzle

    qr = apply_to_query(puzzle, sr.mapping, sr.inv)
    if not qr.ok:
        return None, sr, qr, puzzle
    return qr.encoded, sr, qr, puzzle
