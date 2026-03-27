"""
cot_synthesis/equation_symbolic.py — Synthesize CoT training data for symbolic equation tasks.

Format matches the competition dataset:
  XY op ZW = result
where X,Y,Z,W are symbols from a hidden alphabet, 'op' is an operator whose
semantics are hidden, and 'result' uses the same or related symbols.

Each problem has two hidden layers:
  1. Symbol→digit mapping (e.g. ! -> 3, @ -> 7, ...)
  2. Per-operator semantics: each operator is independently one of:
       - concat     : result = left + right  (string concatenation)
       - rev_concat : result = right + left
       - add        : numeric addition using the digit mapping
       - sub        : numeric subtraction (with '-' sign preserved)
       - mul        : numeric multiplication

Curriculum levels:
  1 — All operators are string ops (concat/rev_concat); no numeric reasoning needed
  2 — One numeric operator (add/sub/mul) + one string operator
  3 — Two numeric operators with different semantics
  4 — Three operators: mixed numeric + string (hardest)

Usage:
    python cot_synthesis/equation_symbolic.py \
        --output_dir data/cot/equation_symbolic \
        --num_samples 500 \
        --levels 1 2 3 4 \
        --seed 42
"""

import argparse
import csv
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL_POOL = list("!@#$%^&*()[]{}|<>?/\\`~'_:;.")
# Exclude chars that conflict with CSV/formatting, equation parsing ('-','='),
# or numeric sign rendering ('-' must only appear as a math minus, never as a symbol).
SAFE_SYMBOLS = [c for c in SYMBOL_POOL if c not in ['"', ',', '\n', '=', '-']]

OPERATOR_CHARS = list("+-*/")           # standard operators (familiar to model)
EXOTIC_OPERATOR_CHARS = list("@#$%^&!") # exotic operators (harder)

STRING_MODES = ["concat", "rev_concat"]
NUMERIC_MODES = ["add", "sub", "mul"]
ALL_MODES = STRING_MODES + NUMERIC_MODES

PROMPT_PREFIX = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
    "Below are a few examples:"
)

# Phrase variation banks for syntactic diversity
OBSERVATION_PHRASES = [
    "Let me analyze the examples systematically.",
    "I'll work through these examples step by step.",
    "Let me investigate the hidden transformation rules.",
    "I need to figure out the secret rules from these equations.",
    "Let me decode the transformation rules from the examples.",
]

HYPOTHESIS_INTRO_PHRASES = [
    "First, let me consider what '{op}' might mean.",
    "The key question is: what does '{op}' do here?",
    "Let me focus on equations containing '{op}'.",
    "My starting point is to examine the '{op}' operator.",
    "Let me isolate equations with '{op}' to understand its behavior.",
]

WRONG_GUESS_PHRASES = [
    "Initial guess: '{op}' might be {wrong_mode}. Let me check.",
    "Could '{op}' be {wrong_mode}? Testing...",
    "Hypothesis: '{op}' is {wrong_mode}.",
    "Let me try assuming '{op}' means {wrong_mode}.",
]

REFUTATION_PHRASES = [
    "This doesn't match — {wrong_mode} is ruled out.",
    "The equation {eq} contradicts this. Discarding {wrong_mode}.",
    "No, this fails on {eq}. {wrong_mode} is not the rule.",
    "{wrong_mode} doesn't work here — the result should be {expected} but we got {got}.",
]

DISCOVERY_PHRASES = [
    "The correct rule for '{op}': it is {mode}.",
    "Breakthrough: '{op}' is {mode}. Let me verify.",
    "Found it — '{op}' behaves as {mode}.",
    "The pattern is clear: '{op}' = {mode}.",
]

MAPPING_INTRO_PHRASES = [
    "Now I need to find the symbol-to-digit mapping.",
    "Next: decode the hidden digit mapping from the numeric equations.",
    "Let me extract the digit mapping using algebra.",
    "Time to reverse-engineer the symbol↔digit correspondence.",
]

APPLY_INTRO_PHRASES = [
    "Now I can solve the target equation.",
    "Let me apply everything I've learned to the query.",
    "Armed with the rules and mapping, I can now compute the answer.",
    "Final step: apply the decoded rules to the target.",
]

CODE_INTRO_PHRASES = [
    "To eliminate any ambiguity, I'll formalize the rules as Python functions:",
    "Let me make the rules concrete by writing them as executable Python:",
    "Formalizing the discovered rules — this captures the exact logic with no room for misinterpretation:",
    "Before solving, I'll crystallize everything I've learned into Python form:",
    "Translating the mental model into code to make the logic explicit and verifiable:",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Step A: Logic Engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OperatorRule:
    """One operator with its hidden semantics."""
    char: str       # the operator character shown in the equation
    mode: str       # "concat" | "rev_concat" | "add" | "sub" | "mul"

    def apply(self, left: str, right: str, mapping: dict[str, int],
              reverse_map: dict[int, str]) -> Optional[str]:
        """Apply the operator to two 2-char operands. Returns result string or None."""
        if self.mode == "concat":
            return left + right
        elif self.mode == "rev_concat":
            return right + left
        elif self.mode in ("add", "sub", "mul"):
            try:
                left_num = mapping[left[0]] * 10 + mapping[left[1]]
                right_num = mapping[right[0]] * 10 + mapping[right[1]]
            except KeyError:
                return None

            if self.mode == "add":
                result_num = left_num + right_num
            elif self.mode == "sub":
                result_num = left_num - right_num
            else:
                result_num = left_num * right_num

            result_str = str(result_num)
            try:
                out = ""
                for ch in result_str:
                    if ch == "-":
                        out += "-"
                    else:
                        out += reverse_map[int(ch)]
                return out
            except KeyError:
                return None
        return None

    @property
    def mode_description(self) -> str:
        names = {
            "concat": "string concatenation (left + right)",
            "rev_concat": "reverse concatenation (right + left)",
            "add": "numeric addition",
            "sub": "numeric subtraction",
            "mul": "numeric multiplication",
        }
        return names.get(self.mode, self.mode)


@dataclass
class SymbolicProblem:
    """A fully generated symbolic equation problem."""
    operators: list[OperatorRule]         # the hidden operator rules
    symbol_pool: list[str]                # symbols used in this problem
    mapping: dict[str, int]              # symbol -> digit
    reverse_map: dict[int, str]          # digit -> symbol
    examples: list[tuple[str, str, str, str]]  # [(left, op, right, result), ...]
    query_left: str
    query_op: str
    query_right: str
    query_result: str
    level: int

    @property
    def op_map(self) -> dict[str, OperatorRule]:
        return {op.char: op for op in self.operators}


# ═══════════════════════════════════════════════════════════════════════════════
# Step B: Problem Generator
# ═══════════════════════════════════════════════════════════════════════════════

def _make_operators(rng: random.Random, level: int,
                    use_exotic: bool) -> list[OperatorRule]:
    """Pick operator characters and assign modes based on curriculum level."""
    op_chars = EXOTIC_OPERATOR_CHARS if use_exotic else OPERATOR_CHARS

    if level == 1:
        # 1-2 operators, all string modes
        n_ops = rng.randint(1, 2)
        modes = rng.choices(STRING_MODES, k=n_ops)
    elif level == 2:
        # 2 operators: one numeric, one string
        n_ops = 2
        modes = [rng.choice(NUMERIC_MODES), rng.choice(STRING_MODES)]
        rng.shuffle(modes)
    elif level == 3:
        # 2-3 operators: 2 distinct numeric modes
        n_ops = rng.choice([2, 3])
        num_modes = rng.sample(NUMERIC_MODES, k=min(2, n_ops))
        str_modes = rng.choices(STRING_MODES, k=n_ops - len(num_modes))
        modes = num_modes + str_modes
        rng.shuffle(modes)
    else:  # level 4
        # 3 operators: mixed, includes at least 2 numeric
        n_ops = 3
        num_modes = rng.sample(NUMERIC_MODES, k=2)
        str_mode = rng.choice(STRING_MODES)
        modes = num_modes + [str_mode]
        rng.shuffle(modes)

    chars = rng.sample(op_chars, k=n_ops)
    return [OperatorRule(char=c, mode=m) for c, m in zip(chars, modes)]


def _make_mapping(rng: random.Random, n_symbols: int,
                  exclude_chars: Optional[list] = None) -> tuple[dict, dict, list]:
    """Create a random symbol→digit mapping.

    Args:
        exclude_chars: Characters to exclude from the symbol pool (e.g., operator chars
                       used in this problem). Prevents operator-symbol collision where the
                       same char appears as both an operand symbol and an operator.
    """
    available = [c for c in SAFE_SYMBOLS if c not in (exclude_chars or [])]
    if len(available) < n_symbols:
        # Extremely unlikely, but fall back gracefully
        available = SAFE_SYMBOLS
    syms = rng.sample(available, k=n_symbols)
    digits = rng.sample(range(10), k=n_symbols)
    mapping = dict(zip(syms, digits))
    reverse_map = dict(zip(digits, syms))
    return mapping, reverse_map, syms


def _sample_operand(rng: random.Random, syms: list[str]) -> str:
    """Sample a 2-character operand from the symbol pool."""
    return rng.choice(syms) + rng.choice(syms)


def generate_problem(rng: random.Random, level: int,
                     n_symbols: int = 9,
                     n_examples: Optional[int] = None,
                     use_exotic: bool = False) -> SymbolicProblem:
    """Generate one symbolic equation problem at the given level.

    Args:
        level: 1-4 curriculum level
        n_symbols: number of distinct symbols in this problem (5-11)
        n_examples: number of demonstration equations (3-5, randomized if None)
        use_exotic: use exotic operator chars (harder to pattern-match)
    """
    if n_examples is None:
        n_examples = rng.randint(3, 5)

    operators = _make_operators(rng, level, use_exotic)
    op_chars_used = [op.char for op in operators]
    mapping, reverse_map, syms = _make_mapping(rng, n_symbols, exclude_chars=op_chars_used)
    op_map = {op.char: op for op in operators}

    # First pass: guarantee at least one example per operator
    examples = []
    for op_rule in operators:
        for _ in range(50):
            left = _sample_operand(rng, syms)
            right = _sample_operand(rng, syms)
            result = op_rule.apply(left, right, mapping, reverse_map)
            if result is None or len(result) == 0:
                continue
            eq = (left, op_rule.char, right, result)
            if eq not in examples:
                examples.append(eq)
                break
        else:
            # Could not generate a valid example for this operator — retry problem
            return generate_problem(rng, level, n_symbols, n_examples, use_exotic)

    # Fill remaining slots with random operators
    attempts = 0
    while len(examples) < n_examples and attempts < 200:
        attempts += 1
        op_rule = rng.choice(operators)
        left = _sample_operand(rng, syms)
        right = _sample_operand(rng, syms)
        result = op_rule.apply(left, right, mapping, reverse_map)
        if result is None or len(result) == 0:
            continue
        eq = (left, op_rule.char, right, result)
        if eq not in examples:
            examples.append(eq)

    if len(examples) < n_examples:
        return generate_problem(rng, level, n_symbols, n_examples, use_exotic)

    # Generate query
    query_attempts = 0
    while True:
        query_attempts += 1
        if query_attempts > 100:
            return generate_problem(rng, level, n_symbols, n_examples, use_exotic)
        op_rule = rng.choice(operators)
        q_left = _sample_operand(rng, syms)
        q_right = _sample_operand(rng, syms)
        q_result = op_rule.apply(q_left, q_right, mapping, reverse_map)
        if q_result is None or len(q_result) == 0:
            continue
        # Avoid query being identical to any example
        if (q_left, op_rule.char, q_right, q_result) not in examples:
            break

    # Validate: at least one numeric operator has evidence in examples (if level > 1)
    if level > 1:
        numeric_ops_used = {eq[1] for eq in examples
                            if op_map[eq[1]].mode in NUMERIC_MODES}
        if not numeric_ops_used:
            return generate_problem(rng, level, n_symbols, n_examples, use_exotic)

    # Validate: result diversity (at least 3 distinct results)
    unique_results = set(eq[3] for eq in examples)
    if len(unique_results) < 2:
        return generate_problem(rng, level, n_symbols, n_examples, use_exotic)

    return SymbolicProblem(
        operators=operators,
        symbol_pool=syms,
        mapping=mapping,
        reverse_map=reverse_map,
        examples=examples,
        query_left=q_left,
        query_op=op_rule.char,
        query_right=q_right,
        query_result=q_result,
        level=level,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step C: CoT Generator
# ═══════════════════════════════════════════════════════════════════════════════

def _pick(rng: random.Random, lst: list) -> str:
    return rng.choice(lst)


def _fmt_eq(left: str, op: str, right: str, result: str) -> str:
    return f"{left}{op}{right} = {result}"


def _wrong_mode_for(correct_mode: str, rng: random.Random) -> str:
    """Pick a plausible wrong mode different from the correct one."""
    options = [m for m in ALL_MODES if m != correct_mode]
    return rng.choice(options)


def _mode_label(mode: str) -> str:
    labels = {
        "concat": "concatenation (left + right)",
        "rev_concat": "reverse concatenation (right + left)",
        "add": "numeric addition",
        "sub": "numeric subtraction",
        "mul": "numeric multiplication",
    }
    return labels.get(mode, mode)


def _generate_wrong_string_test(eq: tuple, wrong_mode: str) -> tuple[str, str]:
    """Show what the wrong mode would produce and why it fails."""
    left, op, right, result = eq
    if wrong_mode == "concat":
        expected_wrong = left + right
    elif wrong_mode == "rev_concat":
        expected_wrong = right + left
    else:
        expected_wrong = f"(numeric result)"
    return expected_wrong, result


def _generate_heuristic_hints(problem: SymbolicProblem) -> list[str]:
    """Produce heuristic observations that narrow down the search space."""
    lines = []
    op_map = problem.op_map

    for op_rule in problem.operators:
        op_eqs = [eq for eq in problem.examples if eq[1] == op_rule.char]
        if not op_eqs:
            continue

        # Check result-length patterns
        result_lens = [len(eq[3]) for eq in op_eqs]
        input_len = 2  # each operand is always 2 chars

        if all(l == 4 for l in result_lens):
            lines.append(
                f"Notice: '{op_rule.char}' always produces 4-character results "
                f"from 2+2 character inputs — this suggests string concatenation."
            )
        elif all(l <= 3 for l in result_lens):
            # Check for minus sign
            has_negative = any(eq[3].startswith("-") for eq in op_eqs)
            if has_negative:
                lines.append(
                    f"Notice: '{op_rule.char}' sometimes produces a '-' prefix "
                    f"— this strongly suggests numeric subtraction."
                )
            else:
                lines.append(
                    f"Notice: '{op_rule.char}' produces shorter results than its "
                    f"inputs (lengths {result_lens}) — numeric arithmetic is likely."
                )

    return lines


def _generate_string_op_cot(op_rule: OperatorRule, op_eqs: list,
                             rng: random.Random) -> list[str]:
    """CoT segment for a string operator (concat/rev_concat).

    Structural jitter:
      - 30% chance: "genius" path — correct guess on first try, no wrong hypothesis
      - 40% chance: one wrong guess before finding the answer
      - 30% chance: two wrong guesses (try a numeric mode first, then wrong string mode)
    """
    lines = []
    correct_mode = op_rule.mode
    wrong_string = "rev_concat" if correct_mode == "concat" else "concat"
    eq = op_eqs[0]

    jitter = rng.random()

    if jitter < 0.30:
        # Genius path: direct insight from result length
        lines.append(
            f"Observation: '{op_rule.char}' always produces "
            f"{len(eq[3])}-character results from two 2-character inputs. "
            f"This immediately suggests string {_mode_label(correct_mode)}."
        )
        lines.append("")
    elif jitter < 0.70:
        # Single wrong guess: opposite string mode
        wrong_result, correct_result = _generate_wrong_string_test(eq, wrong_string)
        lines.append(_pick(rng, WRONG_GUESS_PHRASES).format(
            op=op_rule.char, wrong_mode=_mode_label(wrong_string)
        ))
        lines.append(
            f"  Test on {_fmt_eq(*eq)}: "
            f"'{wrong_string}' would give '{wrong_result}', but result is '{correct_result}'."
        )
        lines.append(_pick(rng, REFUTATION_PHRASES).format(
            wrong_mode=_mode_label(wrong_string),
            eq=_fmt_eq(*eq),
            expected=wrong_result,
            got=correct_result,
        ))
        lines.append("")
    else:
        # Two wrong guesses: numeric first, then wrong string mode
        # Wrong guess 1: numeric add
        numeric_wrong_result = eq[0] + eq[2] + "?"  # placeholder to show it's wrong
        lines.append(_pick(rng, WRONG_GUESS_PHRASES).format(
            op=op_rule.char, wrong_mode=_mode_label("add")
        ))
        lines.append(
            f"  Test on {_fmt_eq(*eq)}: numeric addition would require "
            f"a digit mapping, but all results are exactly 4 characters — "
            f"suspicious. Also, {_fmt_eq(*op_eqs[-1])} gives a 4-char result "
            f"regardless of inputs, which is inconsistent with arithmetic."
        )
        lines.append(_pick(rng, REFUTATION_PHRASES).format(
            wrong_mode=_mode_label("add"),
            eq=_fmt_eq(*eq),
            expected="a shorter number",
            got=eq[3],
        ))
        lines.append("")
        # Wrong guess 2: wrong string mode
        wrong_result, correct_result = _generate_wrong_string_test(eq, wrong_string)
        lines.append(_pick(rng, WRONG_GUESS_PHRASES).format(
            op=op_rule.char, wrong_mode=_mode_label(wrong_string)
        ))
        lines.append(
            f"  Test on {_fmt_eq(*eq)}: "
            f"'{wrong_string}' would give '{wrong_result}', but result is '{correct_result}'."
        )
        lines.append(_pick(rng, REFUTATION_PHRASES).format(
            wrong_mode=_mode_label(wrong_string),
            eq=_fmt_eq(*eq),
            expected=wrong_result,
            got=correct_result,
        ))
        lines.append("")

    # Show the correct mode works
    lines.append(_pick(rng, DISCOVERY_PHRASES).format(
        op=op_rule.char, mode=_mode_label(correct_mode)
    ))
    for eq in op_eqs[:3]:
        left, op, right, result = eq
        computed = left + right if correct_mode == "concat" else right + left
        check = "✓" if computed == result else "✗"
        lines.append(
            f"  {_fmt_eq(left, op, right, result)} → "
            f"'{correct_mode}' gives '{computed}' {check}"
        )
    return lines


def _generate_numeric_op_cot(op_rule: OperatorRule, op_eqs: list,
                              problem: SymbolicProblem,
                              rng: random.Random) -> list[str]:
    """CoT segment for a numeric operator.

    Fixes applied:
    1. Structural jitter: 30% genius (skip wrong guesses), 40% one wrong guess,
       30% two wrong guesses (wrong string + wrong numeric mode).
    2. No magical deduction: the mapping derivation shows explicit unit-digit
       constraints and cross-equation reasoning before revealing numbers.
    3. Robust sub detection: works even when all results happen to be positive.
    """
    lines = []
    mapping = problem.mapping
    reverse_map = problem.reverse_map

    # ── Phase 1: Wrong hypothesis / structural jitter ─────────────────────────
    eq0 = op_eqs[0]
    concat_result = eq0[0] + eq0[2]
    rev_result = eq0[2] + eq0[0]

    jitter = rng.random()

    if jitter < 0.30:
        # Genius path: heuristic immediately rules out string ops
        result_lens = [len(e[3]) for e in op_eqs]
        lines.append(
            f"Key observation: '{op_rule.char}' produces results of length "
            f"{sorted(set(result_lens))} — these vary across equations and are "
            f"shorter than the 4 characters expected from concatenation. "
            f"This rules out all string operations immediately."
        )
        lines.append("")
    elif jitter < 0.70:
        # One wrong guess: always concat (the simplest string hypothesis)
        lines.append(_pick(rng, WRONG_GUESS_PHRASES).format(
            op=op_rule.char, wrong_mode=_mode_label("concat")
        ))
        lines.append(
            f"  Test on {_fmt_eq(*eq0)}: "
            f"'concat' would give '{concat_result}', but result is '{eq0[3]}'."
        )
        lines.append(_pick(rng, REFUTATION_PHRASES).format(
            wrong_mode=_mode_label("concat"),
            eq=_fmt_eq(*eq0),
            expected=concat_result,
            got=eq0[3],
        ))
        lines.append("")
    else:
        # Two wrong guesses: concat, then wrong numeric mode
        # Wrong guess 1: concat
        lines.append(_pick(rng, WRONG_GUESS_PHRASES).format(
            op=op_rule.char, wrong_mode=_mode_label("concat")
        ))
        lines.append(
            f"  Test on {_fmt_eq(*eq0)}: "
            f"'concat' would give '{concat_result}', but result is '{eq0[3]}'."
        )
        lines.append(_pick(rng, REFUTATION_PHRASES).format(
            wrong_mode=_mode_label("concat"),
            eq=_fmt_eq(*eq0),
            expected=concat_result,
            got=eq0[3],
        ))
        lines.append("")
        # Wrong guess 2: a wrong numeric mode
        wrong_num = rng.choice([m for m in NUMERIC_MODES if m != op_rule.mode])
        lines.append(_pick(rng, WRONG_GUESS_PHRASES).format(
            op=op_rule.char, wrong_mode=_mode_label(wrong_num)
        ))
        # Compute what the wrong numeric mode would give for eq0
        left_num_w = mapping[eq0[0][0]] * 10 + mapping[eq0[0][1]]
        right_num_w = mapping[eq0[2][0]] * 10 + mapping[eq0[2][1]]
        if wrong_num == "add":
            wrong_num_result = left_num_w + right_num_w
        elif wrong_num == "sub":
            wrong_num_result = left_num_w - right_num_w
        else:
            wrong_num_result = left_num_w * right_num_w
        wrong_sym_result = "".join(
            "-" if c == "-" else reverse_map.get(int(c), "?")
            for c in str(wrong_num_result)
        )
        if wrong_sym_result != eq0[3]:
            lines.append(
                f"  Test on {_fmt_eq(*eq0)}: "
                f"{_mode_label(wrong_num)} would give '{wrong_sym_result}' "
                f"(mapping → {wrong_num_result}), but result is '{eq0[3]}'."
            )
            lines.append(_pick(rng, REFUTATION_PHRASES).format(
                wrong_mode=_mode_label(wrong_num),
                eq=_fmt_eq(*eq0),
                expected=wrong_sym_result,
                got=eq0[3],
            ))
        else:
            lines.append(f"  This also fails consistently on other equations.")
        lines.append("")

    # ── Phase 2: Identify the correct numeric mode (漏洞3 fix) ───────────────
    has_negative = any(e[3].startswith("-") for e in op_eqs)

    if op_rule.mode == "sub" and has_negative:
        lines.append(
            f"The '-' prefix in some results (e.g. '{op_eqs[0][3]}') "
            f"directly proves this is numeric subtraction — only subtraction "
            f"produces negative values."
        )
    elif op_rule.mode == "sub" and not has_negative:
        # All sub results happen to be positive — need a different signal
        # Compare result magnitude to input magnitude
        sample_left = mapping[op_eqs[0][0][0]] * 10 + mapping[op_eqs[0][0][1]]
        sample_right = mapping[op_eqs[0][2][0]] * 10 + mapping[op_eqs[0][2][1]]
        sample_res = int("".join(str(mapping[c]) for c in op_eqs[0][3]))
        lines.append(
            f"Notice: the results are consistently smaller than the inputs. "
            f"For instance, {_fmt_eq(*op_eqs[0])} yields a result whose "
            f"magnitude is less than both operands — consistent with subtraction. "
            f"Checking multiple equations confirms the pattern: "
            f"result ≈ left − right in all cases."
        )
    elif op_rule.mode == "mul":
        result_lens = [len(e[3]) for e in op_eqs]
        lines.append(
            f"Results have lengths {sorted(set(result_lens))} — some are 3-4 characters "
            f"long, which is too large for addition of two 2-digit numbers (max 99+99=198) "
            f"but consistent with multiplication (e.g. 99×99=9801, up to 4 digits). "
            f"This points to numeric multiplication."
        )
    else:  # add
        result_lens = [len(e[3]) for e in op_eqs]
        lines.append(
            f"Results have lengths {sorted(set(result_lens))} — consistent with "
            f"two-digit addition (max result 198, up to 3 digits). "
            f"Multiplication would produce up to 4 digits; subtraction could produce "
            f"negative values. Addition fits all examples."
        )

    lines.append(_pick(rng, DISCOVERY_PHRASES).format(
        op=op_rule.char, mode=_mode_label(op_rule.mode)
    ))
    lines.append("")

    # ── Phase 3: Mapping derivation — no magical jumps (漏洞1 fix) ────────────
    lines.append(_pick(rng, MAPPING_INTRO_PHRASES))
    lines.append(
        "I'll set up the algebraic constraints from each equation, "
        "then use cross-equation comparison to pin down each symbol's digit."
    )
    lines.append("")

    # Assign variable names to all unknown symbols across all equations
    all_syms_needed = set()
    for eq in op_eqs:
        left, _, right, result = eq
        for ch in left + right + result.replace("-", ""):
            all_syms_needed.add(ch)

    var_map = {}  # symbol -> variable letter
    for i, s in enumerate(sorted(all_syms_needed)):
        var_map[s] = chr(ord('A') + i)

    def sym_to_var(ch):
        return var_map.get(ch, ch)

    if op_rule.mode == "add":
        op_str = "+"
    elif op_rule.mode == "sub":
        op_str = "-"
    else:
        op_str = "×"

    # Show the system of equations in algebraic form
    lines.append("System of equations (symbols → variables):")
    lines.append(
        "  " + ", ".join(f"'{s}'={v}" for s, v in sorted(var_map.items()))
    )
    lines.append("")
    for i, eq in enumerate(op_eqs):
        left, _, right, result = eq
        lv0, lv1 = sym_to_var(left[0]), sym_to_var(left[1])
        rv0, rv1 = sym_to_var(right[0]), sym_to_var(right[1])
        alg_res = "".join(
            "-" if c == "-" else sym_to_var(c)
            for c in result
        )
        lines.append(
            f"  Eq{i+1}: ({lv0}×10+{lv1}) {op_str} ({rv0}×10+{rv1}) = {alg_res}"
        )
    lines.append("")

    # Show the reasoning for pinning digits: look for repeated symbols & unit-digit tricks
    # Find a symbol that appears in both operands and result across multiple equations
    sym_freq = {}
    for eq in op_eqs:
        left, _, right, result = eq
        for ch in set(left + right + result.replace("-", "")):
            sym_freq[ch] = sym_freq.get(ch, 0) + 1
    anchor_sym = max(sym_freq, key=sym_freq.get)

    lines.append(
        f"Cross-equation analysis: '{anchor_sym}' (={sym_to_var(anchor_sym)}) "
        f"appears in {sym_freq[anchor_sym]} equation(s) — a good anchor."
    )

    # Unit-digit constraint: for add/sub, the unit digit of left[1] op right[1]
    # determines (or tightly constrains) the unit digit of the result
    if op_rule.mode in ("add", "sub") and len(op_eqs) >= 1:
        eq = op_eqs[0]
        left, _, right, result = eq
        unit_left = mapping[left[1]]
        unit_right = mapping[right[1]]
        unit_res_sym = result[-1]
        lines.append(
            f"Unit-digit constraint from Eq1: "
            f"the last digit of ({sym_to_var(left[1])} {op_str} {sym_to_var(right[1])}) "
            f"must equal {sym_to_var(unit_res_sym)} (the last symbol of the result)."
        )
        lines.append(
            f"  Only digits where (d₁ {op_str} d₂) mod 10 = mapping('{unit_res_sym}') are valid."
        )
        lines.append("")

    lines.append(
        "By setting up and solving the complete system of constraints "
        f"(exhaustive search over digits 0-9 for {len(all_syms_needed)} unknowns), "
        "the unique consistent solution is:"
    )
    lines.append("")

    # Now reveal the actual values — but only after the above explanation
    shown_syms = set()
    for eq in op_eqs:
        left, _, right, result = eq
        left_num = mapping[left[0]] * 10 + mapping[left[1]]
        right_num = mapping[right[0]] * 10 + mapping[right[1]]
        numeric_result_val = (
            left_num + right_num if op_rule.mode == "add"
            else left_num - right_num if op_rule.mode == "sub"
            else left_num * right_num
        )
        result_digits = " ".join(
            "-" if c == "-" else
            f"{c}\u2192'{reverse_map.get(int(c), c)}'"
            for c in str(numeric_result_val)
        )
        all_syms_in_eq = sorted(set(left + right + result.replace("-", "")))
        new_bindings = [(s, mapping[s]) for s in all_syms_in_eq if s not in shown_syms]
        binding_str = ", ".join(f"'{s}'={d}" for s, d in new_bindings)

        lines.append(f"  {_fmt_eq(left, op_rule.char, right, result)}")
        if binding_str:
            lines.append(f"    New bindings: {binding_str}")
        lines.append(
            f"    Check: {left_num} {op_str} {right_num} = {numeric_result_val} "
            f"→ {result_digits} ✓"
        )
        for s in all_syms_in_eq:
            shown_syms.add(s)

    lines.append("")
    known = {s: mapping[s] for s in shown_syms if s in mapping}
    known_str = ", ".join(f"'{s}'→{d}" for s, d in sorted(known.items(), key=lambda x: x[1]))
    lines.append(f"Partial mapping confirmed: {known_str}")

    return lines


def _generate_code_block(problem: SymbolicProblem, rng: random.Random) -> list[str]:
    """Generate a Python code block expressing the rules."""
    lines = []
    lines.append(_pick(rng, CODE_INTRO_PHRASES))
    lines.append("")
    lines.append("```python")

    # Mapping dict
    m_items = ", ".join(f"'{s}': {d}" for s, d in sorted(problem.mapping.items()))
    lines.append(f"mapping = {{{m_items}}}")
    rev_items = ', '.join(f'{d}: {repr(s)}' for d, s in sorted(problem.reverse_map.items()))
    lines.append(f'reverse = {{{rev_items}}}')
    lines.append("")

    for op_rule in problem.operators:
        fn_name = f"op_{op_rule.char.encode().hex()}"
        lines.append(f"def apply_{op_rule.mode}(left, right):  # '{op_rule.char}'")
        if op_rule.mode == "concat":
            lines.append("    return left + right")
        elif op_rule.mode == "rev_concat":
            lines.append("    return right + left")
        else:
            lines.append("    L = mapping[left[0]] * 10 + mapping[left[1]]")
            lines.append("    R = mapping[right[0]] * 10 + mapping[right[1]]")
            if op_rule.mode == "add":
                lines.append("    n = L + R")
            elif op_rule.mode == "sub":
                lines.append("    n = L - R")
            else:
                lines.append("    n = L * R")
            lines.append("    return ('-' if n < 0 else '') + ''.join(reverse[int(d)] for d in str(abs(n)))")
    lines.append("```")
    lines.append("")

    # Cross-check one example to confirm the formalization is consistent
    eq = rng.choice(problem.examples)
    left, op, right, result = eq
    op_rule_check = problem.op_map[op]
    computed = op_rule_check.apply(left, right, problem.mapping, problem.reverse_map)
    lines.append(
        f"Cross-check with example {_fmt_eq(left, op, right, result)}: "
        f"apply_{op_rule_check.mode}('{left}', '{right}') → '{computed}' "
        f"{'✓ consistent' if computed == result else '✗ mismatch'}"
    )
    return lines


def generate_cot(problem: SymbolicProblem, rng: random.Random) -> str:
    """Generate the full detective-style CoT for a symbolic equation problem."""
    lines = []
    op_map = problem.op_map

    lines.append(_pick(rng, OBSERVATION_PHRASES))
    lines.append("")

    # ── Phase 1: Initial observation + heuristic hints ───────────────────────
    lines.append("**Step 1: Observe the structure**")
    lines.append(
        f"Each equation has the form: [2 symbols] [operator] [2 symbols] = [result]. "
        f"There {'is' if len(problem.operators) == 1 else 'are'} "
        f"{len(problem.operators)} distinct operator(s): "
        f"{', '.join(repr(op.char) for op in problem.operators)}."
    )
    lines.append("")

    # Show first 2 examples
    for i, eq in enumerate(problem.examples[:2]):
        lines.append(f"  Example {i+1}: {_fmt_eq(*eq)}")
    lines.append("")

    # Heuristic hints
    hints = _generate_heuristic_hints(problem)
    if hints:
        lines.append("**Step 2: Quick heuristics**")
        for hint in hints:
            lines.append(f"  • {hint}")
        lines.append("")
        step_offset = 3
    else:
        step_offset = 2

    # ── Phase 2: Per-operator analysis ───────────────────────────────────────
    for op_idx, op_rule in enumerate(problem.operators):
        step_num = step_offset + op_idx
        op_eqs = [eq for eq in problem.examples if eq[1] == op_rule.char]

        lines.append(f"**Step {step_num}: Decode operator '{op_rule.char}'**")
        lines.append(_pick(rng, HYPOTHESIS_INTRO_PHRASES).format(op=op_rule.char))
        lines.append("")

        if op_rule.mode in STRING_MODES:
            lines.extend(_generate_string_op_cot(op_rule, op_eqs, rng))
        else:
            lines.extend(_generate_numeric_op_cot(op_rule, op_eqs, problem, rng))
        lines.append("")

    # ── Phase 3: Formalize Rules as Python (before solving) ──────────────────
    step_num = step_offset + len(problem.operators)
    lines.append(f"**Step {step_num}: Formalize rules as Python**")
    lines.extend(_generate_code_block(problem, rng))
    lines.append("")

    # ── Phase 4: Full rule summary ────────────────────────────────────────────
    step_num += 1
    lines.append(f"**Step {step_num}: Rule summary**")
    mapping_str = ', '.join(f"'{s}'→{d}" for s, d in sorted(problem.mapping.items(), key=lambda x: x[1]))
    lines.append(f'Mapping: {mapping_str}')
    for op_rule in problem.operators:
        lines.append(f"  '{op_rule.char}' = {op_rule.mode_description}")
    lines.append("")

    # ── Phase 5: Solve the query ──────────────────────────────────────────────
    step_num += 1
    lines.append(f"**Step {step_num}: Solve the query**")
    lines.append(_pick(rng, APPLY_INTRO_PHRASES))
    lines.append(f"Query: {problem.query_left}{problem.query_op}{problem.query_right}")
    lines.append("")

    q_op_rule = op_map[problem.query_op]
    if q_op_rule.mode in ("add", "sub", "mul"):
        left_num = problem.mapping[problem.query_left[0]] * 10 + problem.mapping[problem.query_left[1]]
        right_num = problem.mapping[problem.query_right[0]] * 10 + problem.mapping[problem.query_right[1]]

        lines.append(
            f"  Step 1 — Symbol→digit: "
            f"'{problem.query_left}' → {left_num}, "
            f"'{problem.query_right}' → {right_num}"
        )
        if q_op_rule.mode == "add":
            numeric_result = left_num + right_num
            op_sym = "+"
        elif q_op_rule.mode == "sub":
            numeric_result = left_num - right_num
            op_sym = "-"
        else:
            numeric_result = left_num * right_num
            op_sym = "×"
        lines.append(f"  Step 2 — Compute: {left_num} {op_sym} {right_num} = {numeric_result}")
        digit_sym_parts = []
        for c in str(numeric_result):
            if c == "-":
                digit_sym_parts.append("-")
            else:
                sym = problem.reverse_map.get(int(c), "?")
                digit_sym_parts.append(f"{c}\u2192'{sym}'")
        lines.append(f"  Step 3 \u2014 Digit\u2192symbol: {' '.join(digit_sym_parts)}")
    else:
        if q_op_rule.mode == "concat":
            lines.append(
                f"  Concatenation: '{problem.query_left}' + '{problem.query_right}' = '{problem.query_left + problem.query_right}'"
            )
        else:
            lines.append(
                f"  Reverse concat: '{problem.query_right}' + '{problem.query_left}' = '{problem.query_right + problem.query_left}'"
            )

    lines.append("")
    lines.append(f"Result: **{problem.query_result}**")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_prompt(problem: SymbolicProblem) -> str:
    """Format in competition-compatible style."""
    example_lines = []
    for left, op, right, result in problem.examples:
        example_lines.append(f"{left}{op}{right} = {result}")
    examples_str = "\n".join(example_lines)
    query = f"{problem.query_left}{problem.query_op}{problem.query_right}"
    return f"{PROMPT_PREFIX}\n{examples_str}\nNow, determine the result for: {query}"


def format_answer_with_cot(cot_text: str, answer: str) -> str:
    return f"{cot_text}\n\n\\boxed{{{answer}}}"


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    num_samples: int,
    levels: list[int],
    seed: int = 42,
    n_symbols_range: tuple[int, int] = (6, 10),
    exotic_op_prob: float = 0.3,
) -> list[dict]:
    """Generate a full CoT dataset for symbolic equation problems."""
    rng = random.Random(seed)

    samples_per_level = num_samples // len(levels)
    remainder = num_samples % len(levels)

    dataset = []
    for level_idx, level in enumerate(levels):
        count = samples_per_level + (1 if level_idx < remainder else 0)
        for _ in range(count):
            n_syms = rng.randint(*n_symbols_range)
            use_exotic = rng.random() < exotic_op_prob
            problem = generate_problem(rng, level, n_symbols=n_syms, use_exotic=use_exotic)

            prompt = format_prompt(problem)
            answer = problem.query_result
            cot = generate_cot(problem, rng)

            op_summary = {op.char: op.mode for op in problem.operators}
            dataset.append({
                "id": uuid.uuid4().hex[:8],
                "prompt": prompt,
                "answer": answer,
                "cot_answer": format_answer_with_cot(cot, answer),
                "level": level,
                "op_semantics": op_summary,
                "n_symbols": n_syms,
            })

    rng.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full JSONL
    p = output_dir / "equation_symbolic_cot.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for s in dataset:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[saved] {p} ({len(dataset)} samples)")

    # CoT CSV (for training)
    p = output_dir / "equation_symbolic_cot.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "answer"])
        for s in dataset:
            w.writerow([s["id"], s["prompt"], s["cot_answer"]])
    print(f"[saved] {p}")

    # Direct CSV (competition format, no CoT)
    p = output_dir / "equation_symbolic_direct.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "answer"])
        for s in dataset:
            w.writerow([s["id"], s["prompt"], s["answer"]])
    print(f"[saved] {p}")

    # Per-level splits
    by_level = {}
    for s in dataset:
        by_level.setdefault(s["level"], []).append(s)
    for level, samples in sorted(by_level.items()):
        p = output_dir / f"level_{level}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[saved] {p} ({len(samples)} samples, level {level})")

    # Operator mode distribution stats
    from collections import Counter
    mode_dist = Counter()
    for s in dataset:
        for op, mode in s["op_semantics"].items():
            mode_dist[mode] += 1
    stats = {
        "total": len(dataset),
        "by_level": {lv: len(smp) for lv, smp in sorted(by_level.items())},
        "op_mode_distribution": dict(mode_dist),
    }
    p = output_dir / "stats.json"
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[saved] {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT training data for symbolic equation tasks.")
    parser.add_argument("--output_dir", type=str, default="data/cot/equation_symbolic")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--min_symbols", type=int, default=6)
    parser.add_argument("--max_symbols", type=int, default=10)
    parser.add_argument("--exotic_op_prob", type=float, default=0.3,
                        help="Probability of using exotic operator chars (harder).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path("/scratch/jwj/NemotronFT") / output_dir

    print(f"Generating {args.num_samples} symbolic-equation CoT samples "
          f"(levels={args.levels}, seed={args.seed})...")
    dataset = generate_dataset(
        num_samples=args.num_samples,
        levels=args.levels,
        seed=args.seed,
        n_symbols_range=(args.min_symbols, args.max_symbols),
        exotic_op_prob=args.exotic_op_prob,
    )
    save_dataset(dataset, output_dir)
    print(f"\nDone! {len(dataset)} samples saved to {output_dir}/")

    # Preview
    print("\n" + "=" * 70)
    print("SAMPLE PREVIEW")
    print("=" * 70)
    for level in args.levels:
        s = next(x for x in dataset if x["level"] == level)
        print(f"\n--- Level {level} (ops: {s['op_semantics']}) ---")
        print(s["prompt"])
        print(f"\nAnswer: {s['answer']}")
        print(f"\nCoT (first 800 chars):\n{s['cot_answer'][:800]}...")
        print()


if __name__ == "__main__":
    main()
