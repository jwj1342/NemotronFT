"""
cot_synthesis/equation_numeric.py — Synthesize CoT training data for equation_numeric tasks.

Generates "Alice's Wonderland" equation problems where operator symbols are
reassigned to different arithmetic/string operations. Each sample includes:
  - A prompt (matching the competition format)
  - A mathematical-reasoning Chain-of-Thought
  - The final answer

Supports curriculum learning:
  - Level 1: Simple ops (add, sub, concat, swap_concat, abs_diff) with 2 operators, 3-5 examples
  - Level 2: + reversed ops (rev_add, rev_sub, rev_mul) with 2-3 operators, 3-5 examples
  - Level 3: All ops + operator-as-sign display, 2-3 operators, 2-4 examples (less evidence)

Usage:
    python cot_synthesis/equation_numeric.py \
        --output_dir data/cot/equation_numeric \
        --num_samples 500 \
        --levels 1 2 3 \
        --seed 42
"""

import argparse
import csv
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Step A: Operation Definitions & Utilities
# ═══════════════════════════════════════════════════════════════════════════════

# Pool of operator symbols (matching competition style)
OPERATOR_SYMBOLS = [
    "#", "@", "$", "%", "^", "&", "!", "?", "~",
    "{", "}", "[", "]", "(", ")", "<", ">", "|",
    ":", ";", "'", '"', "`", "\\",
    "+", "-", "*", "/",
]


def _rev_str(n: int) -> str:
    """Reverse the string representation of an integer.

    Negative numbers: reverse the absolute value, prepend '-'.
    """
    if n < 0:
        return "-" + str(-n)[::-1]
    return str(n)[::-1]


def _rev_int(s: str) -> int:
    """Reverse the digits of a 2-digit string operand and return as int.

    Preserves leading zeros: '06' -> '60' -> 60.
    """
    return int(s[::-1])


# ── Operation Functions ─────────────────────────────────────────────────────
# Each function takes (a_str, b_str) and returns result_str.
# a_str/b_str are zero-padded 2-digit strings (e.g. "06", "73").

def op_add(a_str: str, b_str: str, _sym: str = "") -> str:
    return str(int(a_str) + int(b_str))


def op_sub(a_str: str, b_str: str, sym: str = "-") -> str:
    """Subtraction with operator-as-sign: negative results use sym instead of '-'."""
    val = int(a_str) - int(b_str)
    if val < 0:
        return f"{sym}{-val}"
    return str(val)


def op_mul(a_str: str, b_str: str, _sym: str = "") -> str:
    return str(int(a_str) * int(b_str))


def op_abs_diff(a_str: str, b_str: str, _sym: str = "") -> str:
    return str(abs(int(a_str) - int(b_str)))


def op_concat(a_str: str, b_str: str, _sym: str = "") -> str:
    """Concatenate: a_str || b_str."""
    return a_str + b_str


def op_swap_concat(a_str: str, b_str: str, _sym: str = "") -> str:
    """Swap-concatenate: b_str || a_str."""
    return b_str + a_str


def op_rev_add(a_str: str, b_str: str, _sym: str = "") -> str:
    """Reverse digits of each operand, add, reverse the result."""
    ra, rb = _rev_int(a_str), _rev_int(b_str)
    return _rev_str(ra + rb)


def op_rev_sub(a_str: str, b_str: str, sym: str = "-") -> str:
    """Reverse digits of each operand, subtract, reverse the result.

    Negative results use operator symbol as sign.
    """
    ra, rb = _rev_int(a_str), _rev_int(b_str)
    val = ra - rb
    rev = _rev_str(val)
    # Replace '-' with operator symbol for negative results
    if rev.startswith("-"):
        return sym + rev[1:]
    return rev


def op_rev_mul(a_str: str, b_str: str, _sym: str = "") -> str:
    """Reverse digits of each operand, multiply, reverse the result."""
    ra, rb = _rev_int(a_str), _rev_int(b_str)
    return _rev_str(ra * rb)


# ── Operation Registry ──────────────────────────────────────────────────────

@dataclass
class OpInfo:
    """Metadata for an operation type."""
    name: str           # human-readable name (e.g. "addition")
    key: str            # short key (e.g. "add")
    func: callable      # (a_str, b_str, sym) -> result_str
    description: str    # CoT description
    uses_sign: bool = False  # whether negative results use operator-as-sign


ALL_OPERATIONS = [
    OpInfo("addition", "add", op_add,
           "standard addition: a + b"),
    OpInfo("subtraction", "sub", op_sub,
           "subtraction: a − b (negative shown with operator symbol)",
           uses_sign=True),
    OpInfo("multiplication", "mul", op_mul,
           "standard multiplication: a × b"),
    OpInfo("absolute difference", "abs_diff", op_abs_diff,
           "absolute difference: |a − b|"),
    OpInfo("concatenation", "concat", op_concat,
           "concatenation: digits of a followed by digits of b"),
    OpInfo("swap-concatenation", "swap_concat", op_swap_concat,
           "swap-concatenation: digits of b followed by digits of a"),
    OpInfo("reverse-add", "rev_add", op_rev_add,
           "reverse digits of each operand, add, then reverse the result"),
    OpInfo("reverse-subtract", "rev_sub", op_rev_sub,
           "reverse digits of each operand, subtract, then reverse the result",
           uses_sign=True),
    OpInfo("reverse-multiply", "rev_mul", op_rev_mul,
           "reverse digits of each operand, multiply, then reverse the result"),
]

# Lookup by key
_OP_BY_KEY = {op.key: op for op in ALL_OPERATIONS}

# Level-specific operation pools
_LEVEL_OPS = {
    1: ["add", "sub", "mul", "abs_diff", "concat", "swap_concat"],
    2: ["add", "sub", "mul", "abs_diff", "concat", "swap_concat",
        "rev_add", "rev_sub", "rev_mul"],
    3: ["add", "sub", "mul", "abs_diff", "concat", "swap_concat",
        "rev_add", "rev_sub", "rev_mul"],
}

# Number of distinct operators per problem, per level
_NUM_OPS = {
    1: (2, 2),    # always 2 operators
    2: (2, 3),    # 2-3 operators
    3: (2, 3),    # 2-3 operators
}

# Number of example equations per problem, per level
_NUM_EXAMPLES = {
    1: (3, 5),
    2: (3, 5),
    3: (2, 4),    # fewer examples = harder
}


# ═══════════════════════════════════════════════════════════════════════════════
# Step B: Problem Generation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EquationProblem:
    """A fully generated equation_numeric problem."""
    operator_map: dict              # symbol -> OpInfo
    examples: list[tuple[str, str, str, str]]  # (a_str, sym, b_str, result_str)
    query_a: str                    # query operand a
    query_sym: str                  # query operator symbol
    query_b: str                    # query operand b
    query_result: str               # ground truth result
    level: int                      # curriculum level

    @property
    def prompt_equations(self) -> list[str]:
        """Format example equations as competition-style lines."""
        return [f"{a}{sym}{b} = {r}" for a, sym, b, r in self.examples]

    @property
    def query_equation(self) -> str:
        """Format the query equation (without result)."""
        return f"{self.query_a}{self.query_sym}{self.query_b}"


def _random_operand(rng: random.Random) -> str:
    """Generate a random 2-digit operand (00-99), zero-padded."""
    return f"{rng.randint(0, 99):02d}"


def _ensure_no_trivial_collision(op_keys: list[str], operands: list[tuple[str, str]],
                                  sym_map: dict) -> bool:
    """Check that different operators produce different results for the same operands.

    This ensures the problem is solvable — if two operators gave identical results
    on all examples, the model couldn't distinguish them.
    """
    # Simple heuristic: just check that the operations themselves are different
    return len(set(op_keys)) == len(op_keys)


def generate_problem(rng: random.Random, level: int) -> EquationProblem:
    """Generate a random equation_numeric problem at the given curriculum level."""
    num_ops_range = _NUM_OPS[level]
    num_examples_range = _NUM_EXAMPLES[level]
    available_ops = _LEVEL_OPS[level]

    # Pick number of distinct operators
    num_ops = rng.randint(*num_ops_range)

    # Choose operator symbols (no repeats)
    symbols = rng.sample(OPERATOR_SYMBOLS, num_ops)

    # Choose operations (no repeats)
    op_keys = rng.sample(available_ops, num_ops)

    # Build symbol -> OpInfo mapping
    operator_map = {}
    for sym, key in zip(symbols, op_keys):
        operator_map[sym] = _OP_BY_KEY[key]

    # Generate examples: distribute among operators (each gets at least 1)
    n_examples = rng.randint(*num_examples_range)
    n_examples = max(n_examples, num_ops)  # at least one per operator

    # Assign examples to operators: first round-robin, then random
    example_assignments = list(range(num_ops))
    for _ in range(n_examples - num_ops):
        example_assignments.append(rng.randint(0, num_ops - 1))
    rng.shuffle(example_assignments)

    examples = []
    for op_idx in example_assignments:
        sym = symbols[op_idx]
        op_info = operator_map[sym]

        # Generate operands (avoid degenerate cases for some ops)
        for _attempt in range(50):
            a_str = _random_operand(rng)
            b_str = _random_operand(rng)
            # Avoid 00*00=0 (too trivial), and division by zero
            if op_info.key == "mul" and int(a_str) == 0 and int(b_str) == 0:
                continue
            # Avoid identical operands for abs_diff (always 0)
            if op_info.key == "abs_diff" and a_str == b_str:
                continue
            break

        result = op_info.func(a_str, b_str, sym)
        examples.append((a_str, sym, b_str, result))

    # Generate query: pick a random operator
    query_sym = rng.choice(symbols)
    query_op = operator_map[query_sym]

    for _attempt in range(50):
        qa = _random_operand(rng)
        qb = _random_operand(rng)
        if query_op.key == "abs_diff" and qa == qb:
            continue
        break

    query_result = query_op.func(qa, qb, query_sym)

    return EquationProblem(
        operator_map=operator_map,
        examples=examples,
        query_a=qa,
        query_sym=query_sym,
        query_b=qb,
        query_result=query_result,
        level=level,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step C: Chain-of-Thought Generation (Mathematical Reasoning Style)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Phrasing Diversity ────────────────────────────────────────────────────────

_OPENINGS = [
    "Let me analyze the equation examples to discover the transformation rules.",
    "I need to figure out what each operator does from the given examples.",
    "Let me work through these equations to find the hidden operations.",
    "Looking at the examples to determine what each symbol represents.",
    "I'll examine the examples to figure out the secret transformation rules.",
]

_OP_DISCOVERIES = {
    "add": [
        "'{sym}' performs **addition**: {a} + {b} = {result}.",
        "The '{sym}' operator is **standard addition**.",
        "'{sym}' simply adds the two numbers together.",
    ],
    "sub": [
        "'{sym}' performs **subtraction**: {a} − {b} = {result}.",
        "The '{sym}' operator is **subtraction** (a minus b).",
        "'{sym}' subtracts the right operand from the left.",
    ],
    "mul": [
        "'{sym}' performs **multiplication**: {a} × {b} = {result}.",
        "The '{sym}' operator is **multiplication**.",
        "'{sym}' multiplies the two numbers together.",
    ],
    "abs_diff": [
        "'{sym}' computes the **absolute difference**: |{a} − {b}| = {result}.",
        "The '{sym}' operator gives the **absolute difference** of the two numbers.",
        "'{sym}' takes the unsigned difference between the operands.",
    ],
    "concat": [
        "'{sym}' performs **concatenation**: \"{a}\" + \"{b}\" = \"{result}\".",
        "The '{sym}' operator **concatenates** the digits: a followed by b.",
        "'{sym}' joins the two numbers as strings.",
    ],
    "swap_concat": [
        "'{sym}' performs **swap-concatenation**: \"{b}\" + \"{a}\" = \"{result}\".",
        "The '{sym}' operator **swaps then concatenates**: b followed by a.",
        "'{sym}' reverses the order and concatenates: right operand first.",
    ],
    "rev_add": [
        "'{sym}' performs **reverse-add**: reverse each operand, add, reverse the result.",
        "The '{sym}' operator reverses digits, adds, then reverses the sum.",
        "'{sym}' is a **reverse-digit addition** operation.",
    ],
    "rev_sub": [
        "'{sym}' performs **reverse-subtract**: reverse each operand, subtract, reverse result.",
        "The '{sym}' operator reverses digits, subtracts, then reverses.",
        "'{sym}' is a **reverse-digit subtraction** operation.",
    ],
    "rev_mul": [
        "'{sym}' performs **reverse-multiply**: reverse each operand, multiply, reverse result.",
        "The '{sym}' operator reverses digits, multiplies, then reverses the product.",
        "'{sym}' is a **reverse-digit multiplication** operation.",
    ],
}


# ── Full CoT Generation ──────────────────────────────────────────────────────

def generate_cot(problem: EquationProblem, rng: random.Random) -> str:
    """Generate a full mathematical-reasoning CoT trace.

    Structure:
    1. Observation — list all examples
    2. Group by operator
    3. For each operator: test hypotheses, discover operation
    [Optional] Detour: wrong initial guess
    4. Verify on all examples
    5. Apply to query
    """
    lines = []
    step = 1

    lines.append(rng.choice(_OPENINGS))
    lines.append("")

    # ── Observation ──────────────────────────────────────────────────────────
    lines.append(f"**Step {step}: List the examples**")
    step += 1
    for a, sym, b, r in problem.examples:
        lines.append(f"  {a}{sym}{b} = {r}")
    lines.append("")

    # ── Group by operator ────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Group equations by operator**")
    step += 1

    by_sym = {}
    for a, sym, b, r in problem.examples:
        by_sym.setdefault(sym, []).append((a, b, r))

    for sym, eqs in by_sym.items():
        eq_strs = [f"{a}{sym}{b} = {r}" for a, b, r in eqs]
        lines.append(f"  Operator '{sym}': {', '.join(eq_strs)}")
    lines.append("")

    # ── Analyze each operator ────────────────────────────────────────────────
    lines.append(f"**Step {step}: Analyze each operator**")
    step += 1

    for sym, eqs in by_sym.items():
        op_info = problem.operator_map[sym]
        lines.append(f"\nFor operator '{sym}':")

        # Show hypothesis testing
        a0, b0, r0 = eqs[0]
        ai, bi = int(a0), int(b0)

        # Try standard operations as hypotheses
        hypotheses = [
            ("addition", str(ai + bi)),
            ("subtraction", str(ai - bi)),
            ("multiplication", str(ai * bi)),
            ("absolute difference", str(abs(ai - bi))),
            ("concatenation", a0 + b0),
            ("swap-concatenation", b0 + a0),
        ]

        # Add reversed hypotheses for Level 2+
        if problem.level >= 2:
            ra, rb = int(a0[::-1]), int(b0[::-1])
            hypotheses.extend([
                ("reverse-add", _rev_str(ra + rb)),
                ("reverse-subtract", _rev_str(ra - rb)),
                ("reverse-multiply", _rev_str(ra * rb)),
            ])

        # ── [Optional] Wrong hypothesis detour (~35%) ────────────────────
        if rng.random() < 0.35:
            # Pick a wrong hypothesis
            wrong_candidates = [
                (name, val) for name, val in hypotheses
                if val != r0 and name != op_info.name
            ]
            if wrong_candidates:
                wrong_name, wrong_val = rng.choice(wrong_candidates)
                lines.append(f"  Hypothesis: {wrong_name}?")
                lines.append(f"    {a0} → expected {wrong_val}, but got {r0}. ✗")
                lines.append(f"  That's not {wrong_name}. Let me try other operations.")
                lines.append("")

        # Show the correct hypothesis
        found = False
        for hyp_name, hyp_val in hypotheses:
            if hyp_val == r0:
                # Verify on other examples for this operator
                all_match = True
                for a, b, r in eqs[1:]:
                    test_result = op_info.func(a, b, sym)
                    if test_result != r:
                        all_match = False
                        break

                if all_match:
                    lines.append(f"  Hypothesis: {hyp_name}?")
                    for a, b, r in eqs:
                        test_result = op_info.func(a, b, sym)
                        lines.append(f"    {a}{sym}{b}: {hyp_name} gives {test_result}"
                                     f" = {r} ✓")

                    # Discovery phrasing
                    templates = _OP_DISCOVERIES.get(op_info.key, [
                        f"'{{sym}}' performs **{op_info.name}**: {op_info.description}."
                    ])
                    discovery = rng.choice(templates).format(
                        sym=sym, a=a0, b=b0, result=r0)
                    lines.append(f"  → {discovery}")
                    found = True
                    break

        if not found:
            # Fallback (shouldn't happen with our operations)
            lines.append(f"  → '{sym}' performs {op_info.name}: {op_info.description}.")

        lines.append("")

    # ── Verify all examples ──────────────────────────────────────────────────
    lines.append(f"**Step {step}: Verify on all examples**")
    step += 1

    for a, sym, b, r in problem.examples:
        op_info = problem.operator_map[sym]
        computed = op_info.func(a, b, sym)
        match = computed == r
        lines.append(f"  {a}{sym}{b} = {computed} "
                     f"(expected {r}) {'✓' if match else '✗'}")
    lines.append("")
    lines.append("All examples verified successfully!")
    lines.append("")

    # ── Compute query answer ─────────────────────────────────────────────────
    lines.append(f"**Step {step}: Solve the query**")
    step += 1

    qa, qsym, qb = problem.query_a, problem.query_sym, problem.query_b
    qop = problem.operator_map[qsym]
    lines.append(f"Query: {qa}{qsym}{qb}")
    lines.append(f"Operator '{qsym}' = {qop.name}")

    # Show computation steps based on operation type
    lines.extend(_show_computation(qa, qb, qsym, qop))
    lines.append(f"Result: {problem.query_result}")

    return "\n".join(lines)


def _show_computation(a_str: str, b_str: str, sym: str, op: OpInfo) -> list[str]:
    """Show step-by-step computation for the query."""
    lines = []
    a, b = int(a_str), int(b_str)

    if op.key == "add":
        lines.append(f"  {a} + {b} = {a + b}")
    elif op.key == "sub":
        val = a - b
        if val < 0:
            lines.append(f"  {a} − {b} = {val}")
            lines.append(f"  Negative result displayed with operator symbol: {sym}{-val}")
        else:
            lines.append(f"  {a} − {b} = {val}")
    elif op.key == "mul":
        lines.append(f"  {a} × {b} = {a * b}")
    elif op.key == "abs_diff":
        lines.append(f"  |{a} − {b}| = {abs(a - b)}")
    elif op.key == "concat":
        lines.append(f"  Concatenate: \"{a_str}\" + \"{b_str}\" = \"{a_str}{b_str}\"")
    elif op.key == "swap_concat":
        lines.append(f"  Swap and concatenate: \"{b_str}\" + \"{a_str}\" = \"{b_str}{a_str}\"")
    elif op.key == "rev_add":
        ra, rb = _rev_int(a_str), _rev_int(b_str)
        s = ra + rb
        lines.append(f"  Reverse operands: {a_str}→{ra}, {b_str}→{rb}")
        lines.append(f"  Add: {ra} + {rb} = {s}")
        lines.append(f"  Reverse result: {s}→{_rev_str(s)}")
    elif op.key == "rev_sub":
        ra, rb = _rev_int(a_str), _rev_int(b_str)
        s = ra - rb
        rev = _rev_str(s)
        lines.append(f"  Reverse operands: {a_str}→{ra}, {b_str}→{rb}")
        lines.append(f"  Subtract: {ra} − {rb} = {s}")
        lines.append(f"  Reverse result: {s}→{rev}")
        if s < 0:
            lines.append(f"  Negative displayed with operator symbol: "
                         f"{sym}{rev[1:] if rev.startswith('-') else rev}")
    elif op.key == "rev_mul":
        ra, rb = _rev_int(a_str), _rev_int(b_str)
        s = ra * rb
        lines.append(f"  Reverse operands: {a_str}→{ra}, {b_str}→{rb}")
        lines.append(f"  Multiply: {ra} × {rb} = {s}")
        lines.append(f"  Reverse result: {s}→{_rev_str(s)}")

    return lines


# ── Light CoT Generation ─────────────────────────────────────────────────────

def generate_light_cot(problem: EquationProblem, rng: random.Random) -> str:
    """Generate abbreviated CoT: 3-5 sentences capturing key insights.

    No step numbers, no detailed hypothesis testing. Just:
    observation → discovered operations → apply to query → answer.
    """
    lines = []

    # Group by operator
    by_sym = {}
    for a, sym, b, r in problem.examples:
        by_sym.setdefault(sym, []).append((a, b, r))

    # Observation
    lines.append(f"Examining {len(problem.examples)} example equations "
                 f"with {len(by_sym)} distinct operator(s).")
    lines.append("")

    # For each operator, state the discovered operation concisely
    for sym, eqs in by_sym.items():
        op = problem.operator_map[sym]
        a0, b0, r0 = eqs[0]
        lines.append(f"Operator '{sym}': testing {a0}{sym}{b0}={r0} → "
                     f"**{op.name}** ({op.description}).")

        if len(eqs) > 1:
            # Quick verification on another example
            a1, b1, r1 = eqs[1]
            lines.append(f"  Verified: {a1}{sym}{b1} = {r1} ✓")
    lines.append("")

    # Apply to query
    qa, qsym, qb = problem.query_a, problem.query_sym, problem.query_b
    qop = problem.operator_map[qsym]

    lines.append(f"Query: {qa}{qsym}{qb}, using '{qsym}' = {qop.name}.")

    # One-line computation
    result = problem.query_result
    if qop.key in ("rev_add", "rev_sub", "rev_mul"):
        ra, rb = _rev_int(qa), _rev_int(qb)
        lines.append(f"  Reverse: {qa}→{ra}, {qb}→{rb}, "
                     f"compute → reverse → {result}")
    else:
        lines.append(f"  = {result}")

    return "\n".join(lines)


# ── Detour Generators ────────────────────────────────────────────────────────

def _generate_wrong_op_detour(sym: str, eqs: list[tuple[str, str, str]],
                               correct_op: OpInfo, rng: random.Random) -> list[str]:
    """Generate a wrong-operation hypothesis that fails.

    The model guesses a wrong operation, checks, gets a mismatch,
    and then course-corrects.
    """
    a0, b0, r0 = eqs[0]
    ai, bi = int(a0), int(b0)

    # Build wrong candidates
    wrong_tests = {
        "addition": str(ai + bi),
        "subtraction": str(ai - bi),
        "multiplication": str(ai * bi),
        "concatenation": a0 + b0,
    }

    candidates = [(name, val) for name, val in wrong_tests.items()
                  if val != r0 and name != correct_op.name]
    if not candidates:
        return []

    wrong_name, wrong_val = rng.choice(candidates)

    lines = []
    lines.append(f"  Let me first try {wrong_name} for '{sym}':")
    lines.append(f"    {a0}{sym}{b0}: {wrong_name} gives {wrong_val}, "
                 f"but expected {r0}. ✗")
    lines.append(f"  Not {wrong_name}. Need to look more carefully.")
    lines.append("")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:\n"
    "{examples}\n"
    "Now, determine the result for: {query}"
)


def format_prompt(problem: EquationProblem) -> str:
    """Format the problem into the competition-style prompt."""
    return PROMPT_TEMPLATE.format(
        examples="\n".join(problem.prompt_equations),
        query=problem.query_equation,
    )


def format_answer_with_cot(cot_text: str, answer: str) -> str:
    r"""Format the full answer: CoT reasoning followed by \boxed{answer}."""
    return f"{cot_text}\n\n\\boxed{{{answer}}}"


def format_answer_direct(answer: str) -> str:
    r"""Format direct answer (no CoT): just \boxed{answer}."""
    return f"\\boxed{{{answer}}}"


def format_answer_think(cot_text: str, answer: str, depth: str) -> str:
    r"""Format answer for <think> training mode.

    - depth="direct":  </think>\n\boxed{answer}
    - depth="light":   light_reasoning\n</think>\n\boxed{answer}
    - depth="full":    full_reasoning\n</think>\n\boxed{answer}
    """
    if depth == "direct":
        return f"</think>\n\\boxed{{{answer}}}"
    return f"{cot_text}\n</think>\n\\boxed{{{answer}}}"


# ═══════════════════════════════════════════════════════════════════════════════
# CoT Depth Assignment (direct / light / full per difficulty level)
# ═══════════════════════════════════════════════════════════════════════════════

_DEPTH_RATIOS = {
    #          direct  light   full
    1:        (0.40,   0.35,   0.25),   # Simple ops: often straightforward
    2:        (0.25,   0.30,   0.45),   # Reversed ops: more reasoning needed
    3:        (0.15,   0.25,   0.60),   # Few examples + complex ops: need deep reasoning
}


def _assign_depth(level: int, rng: random.Random) -> str:
    """Assign CoT depth (direct/light/full) based on level and randomness."""
    ratios = _DEPTH_RATIOS.get(level, (0.30, 0.30, 0.40))
    r = rng.random()
    if r < ratios[0]:
        return "direct"
    elif r < ratios[0] + ratios[1]:
        return "light"
    return "full"


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Generate Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def _describe_operators(problem: EquationProblem) -> str:
    """Human-readable description of the operator mapping."""
    parts = []
    for sym, op in problem.operator_map.items():
        parts.append(f"'{sym}'={op.name}")
    return ", ".join(parts)


def generate_dataset(
    num_samples: int,
    levels: list[int],
    seed: int = 42,
) -> list[dict]:
    """Generate a CoT dataset for equation_numeric tasks with layered depth.

    Returns list of dicts with keys: id, prompt, answer, cot_answer,
    think_answer, cot_depth, level, operator_description.
    """
    rng = random.Random(seed)

    # Distribute samples across levels
    samples_per_level = num_samples // len(levels)
    remainder = num_samples % len(levels)

    dataset = []
    for level_idx, level in enumerate(levels):
        count = samples_per_level + (1 if level_idx < remainder else 0)

        for _ in range(count):
            problem = generate_problem(rng, level)

            prompt = format_prompt(problem)
            answer = problem.query_result
            depth = _assign_depth(level, rng)

            # Generate reasoning at the assigned depth
            if depth == "full":
                cot = generate_cot(problem, rng)
            elif depth == "light":
                cot = generate_light_cot(problem, rng)
            else:
                cot = ""

            sample_id = uuid.uuid4().hex[:8]
            dataset.append({
                "id": sample_id,
                "prompt": prompt,
                "answer": answer,
                "cot_answer": format_answer_with_cot(cot, answer)
                              if cot else format_answer_direct(answer),
                "think_answer": format_answer_think(cot, answer, depth),
                "cot_depth": depth,
                "level": level,
                "operator_description": _describe_operators(problem),
            })

    rng.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], output_dir: Path):
    """Save dataset in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "equation_numeric"

    # 1. Full JSONL (all fields)
    jsonl_path = output_dir / f"{prefix}_cot.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[saved] {jsonl_path} ({len(dataset)} samples)")

    # 2. Competition-format CSV — direct answer only
    csv_direct_path = output_dir / f"{prefix}_direct.csv"
    with open(csv_direct_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"],
                             sample["answer"]])
    print(f"[saved] {csv_direct_path}")

    # 3. CoT-format CSV — answer includes CoT reasoning
    csv_cot_path = output_dir / f"{prefix}_cot.csv"
    with open(csv_cot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"],
                             sample["cot_answer"]])
    print(f"[saved] {csv_cot_path}")

    # 4. <think> format CSV — for train.py --enable_thinking mode
    csv_think_path = output_dir / f"{prefix}_think.csv"
    with open(csv_think_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"],
                             sample["think_answer"]])
    print(f"[saved] {csv_think_path}")

    # 5. Per-level splits
    by_level = {}
    for sample in dataset:
        by_level.setdefault(sample["level"], []).append(sample)

    for level, samples in sorted(by_level.items()):
        level_path = output_dir / f"level_{level}.jsonl"
        with open(level_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"[saved] {level_path} ({len(samples)} samples, level {level})")

    # 6. Stats
    by_depth = {}
    for sample in dataset:
        by_depth.setdefault(sample["cot_depth"], []).append(sample)

    stats = {
        "total_samples": len(dataset),
        "by_level": {
            str(level): len(samples)
            for level, samples in sorted(by_level.items())
        },
        "by_depth": {
            depth: len(samples)
            for depth, samples in sorted(by_depth.items())
        },
        "depth_by_level": {
            str(level): {
                depth: sum(1 for s in samples if s["cot_depth"] == depth)
                for depth in ["direct", "light", "full"]
            }
            for level, samples in sorted(by_level.items())
        },
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[saved] {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT training data for equation_numeric tasks.")
    parser.add_argument("--output_dir", type=str,
                        default="data/cot/equation_numeric",
                        help="Output directory for generated data.")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Total number of samples to generate.")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
                        help="Curriculum levels to include.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Generating {args.num_samples} equation_numeric CoT samples "
          f"(levels={args.levels}, seed={args.seed})...")
    dataset = generate_dataset(
        num_samples=args.num_samples,
        levels=args.levels,
        seed=args.seed,
    )
    save_dataset(dataset, output_dir)
    print(f"\nDone! {len(dataset)} samples saved to {output_dir}/")

    # Print a sample preview for each depth
    for depth in ["direct", "light", "full"]:
        samples = [s for s in dataset if s["cot_depth"] == depth]
        if samples:
            s = samples[0]
            print(f"\n{'=' * 70}")
            print(f"SAMPLE PREVIEW — depth={depth}, level={s['level']}, "
                  f"ops={s['operator_description']}")
            print("=" * 70)
            print(f"Prompt: {s['prompt'][:200]}...")
            think = s["think_answer"]
            print(f"\n--- Think Answer ---\n{think[:800]}"
                  f"{'...' if len(think) > 800 else ''}")


if __name__ == "__main__":
    main()
