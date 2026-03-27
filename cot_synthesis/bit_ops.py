"""
cot_synthesis/bit_ops.py — Synthesize CoT training data for bit manipulation tasks.

Generates problems where a hidden rule (1-3 atomic operations) transforms 8-bit
binary numbers. Each sample includes:
  - A prompt (matching the competition format)
  - A detective-style Chain-of-Thought with hypothesis testing
  - The final answer

Supports curriculum learning:
  - Level 1: single atomic operation
  - Level 2: two-operation composition
  - Level 3: three-operation composition

Usage:
    python cot_synthesis/bit_ops.py \
        --output_dir data/cot/bit_ops \
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
# Step A: Atomic Operations (Rule Pool)
# ═══════════════════════════════════════════════════════════════════════════════

def shift_left(x: int, n: int) -> int:
    """Logical left shift by n, keep lower 8 bits."""
    return (x << n) & 0xFF


def shift_right(x: int, n: int) -> int:
    """Logical right shift by n."""
    return (x >> n) & 0xFF


def rotate_left(x: int, n: int) -> int:
    """Circular left rotation by n on 8 bits."""
    n = n % 8
    return ((x << n) | (x >> (8 - n))) & 0xFF


def rotate_right(x: int, n: int) -> int:
    """Circular right rotation by n on 8 bits."""
    n = n % 8
    return ((x >> n) | (x << (8 - n))) & 0xFF


def bitwise_not(x: int) -> int:
    """Bitwise NOT (complement) on 8 bits."""
    return (~x) & 0xFF


def bitwise_xor(x: int, mask: int) -> int:
    """XOR with a fixed 8-bit mask."""
    return (x ^ mask) & 0xFF


def bitwise_and(x: int, mask: int) -> int:
    """AND with a fixed 8-bit mask."""
    return (x & mask) & 0xFF


def bitwise_or(x: int, mask: int) -> int:
    """OR with a fixed 8-bit mask."""
    return (x | mask) & 0xFF


def reverse_bits(x: int) -> int:
    """Reverse the bit order of an 8-bit number."""
    result = 0
    for _ in range(8):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


def swap_nibbles(x: int) -> int:
    """Swap the upper 4 bits and lower 4 bits."""
    return ((x & 0x0F) << 4) | ((x >> 4) & 0x0F)


def swap_adjacent_pairs(x: int) -> int:
    """Swap adjacent bit pairs: bits (0,1)<->(2,3), (4,5)<->(6,7)."""
    return ((x & 0x33) << 2) | ((x >> 2) & 0x33)


def swap_adjacent_bits(x: int) -> int:
    """Swap adjacent single bits: bit0<->bit1, bit2<->bit3, etc."""
    return ((x & 0x55) << 1) | ((x >> 1) & 0x55)


# ── Operation Registry ───────────────────────────────────────────────────────

@dataclass
class AtomicOp:
    """Describes one atomic bit operation with its parameters."""
    name: str               # human-readable name
    func: callable          # function(x, ...) -> int
    params: dict = field(default_factory=dict)   # fixed params (excluding x)
    description: str = ""   # natural language description for CoT

    def apply(self, x: int) -> int:
        return self.func(x, **self.params)


def _fmt_bin(mask: int) -> str:
    """Format an 8-bit integer as a spaced binary string."""
    return format(mask, '08b')


def make_op_pool(rng: random.Random) -> list[AtomicOp]:
    """Generate a diverse pool of atomic operations with random parameters."""
    pool = []

    # Shifts (1-3 positions)
    for n in range(1, 4):
        pool.append(AtomicOp(
            f"shift_left_{n}", shift_left, {"n": n},
            f"logical left shift by {n} bit{'s' if n > 1 else ''}",
        ))
        pool.append(AtomicOp(
            f"shift_right_{n}", shift_right, {"n": n},
            f"logical right shift by {n} bit{'s' if n > 1 else ''}",
        ))

    # Rotations (1-3 positions)
    for n in range(1, 4):
        pool.append(AtomicOp(
            f"rotate_left_{n}", rotate_left, {"n": n},
            f"circular left rotation by {n} bit{'s' if n > 1 else ''}",
        ))
        pool.append(AtomicOp(
            f"rotate_right_{n}", rotate_right, {"n": n},
            f"circular right rotation by {n} bit{'s' if n > 1 else ''}",
        ))

    # NOT
    pool.append(AtomicOp("bitwise_not", bitwise_not, {}, "bitwise NOT (complement)"))

    # XOR with random masks (generate a few diverse masks)
    xor_masks = _sample_diverse_masks(rng, count=5)
    for mask in xor_masks:
        pool.append(AtomicOp(
            f"xor_{_fmt_bin(mask)}", bitwise_xor, {"mask": mask},
            f"XOR with mask {_fmt_bin(mask)}",
        ))

    # AND with random masks
    and_masks = _sample_diverse_masks(rng, count=3, avoid_trivial=True)
    for mask in and_masks:
        pool.append(AtomicOp(
            f"and_{_fmt_bin(mask)}", bitwise_and, {"mask": mask},
            f"AND with mask {_fmt_bin(mask)}",
        ))

    # OR with random masks
    or_masks = _sample_diverse_masks(rng, count=3, avoid_trivial=True)
    for mask in or_masks:
        pool.append(AtomicOp(
            f"or_{_fmt_bin(mask)}", bitwise_or, {"mask": mask},
            f"OR with mask {_fmt_bin(mask)}",
        ))

    # Bit reversal
    pool.append(AtomicOp("reverse_bits", reverse_bits, {}, "reverse bit order"))

    # Nibble swap
    pool.append(AtomicOp("swap_nibbles", swap_nibbles, {}, "swap upper and lower 4-bit nibbles"))

    # Adjacent pair swap
    pool.append(AtomicOp("swap_adjacent_pairs", swap_adjacent_pairs, {},
                         "swap adjacent 2-bit pairs"))

    # Adjacent bit swap
    pool.append(AtomicOp("swap_adjacent_bits", swap_adjacent_bits, {},
                         "swap adjacent single bits"))

    return pool


def _sample_diverse_masks(rng: random.Random, count: int = 5,
                          avoid_trivial: bool = False) -> list[int]:
    """Sample diverse 8-bit masks that aren't all 0s or all 1s."""
    masks = set()
    while len(masks) < count:
        m = rng.randint(1, 254)  # avoid 0x00 and 0xFF
        if avoid_trivial and m in (0x0F, 0xF0):
            continue
        masks.add(m)
    return sorted(masks)


# ═══════════════════════════════════════════════════════════════════════════════
# Step B: Problem Generation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BitOpsProblem:
    """A fully generated bit-ops problem with rule, examples, and query."""
    rule_ops: list[AtomicOp]       # ordered list of operations (the secret rule)
    examples: list[tuple[int, int]]  # (input, output) demonstration pairs
    query_input: int                # the input to predict
    query_output: int               # ground truth output
    level: int                      # curriculum level (1/2/3)

    def apply_rule(self, x: int) -> int:
        for op in self.rule_ops:
            x = op.apply(x)
        return x

    @property
    def rule_description(self) -> str:
        if len(self.rule_ops) == 1:
            return self.rule_ops[0].description
        steps = [f"step {i+1}: {op.description}" for i, op in enumerate(self.rule_ops)]
        return ", then ".join([op.description for op in self.rule_ops])


def generate_problem(rng: random.Random, op_pool: list[AtomicOp],
                     level: int, num_examples: tuple[int, int] = (6, 10)
                     ) -> BitOpsProblem:
    """Generate a random bit-ops problem at the given curriculum level.

    Args:
        level: 1 = single op, 2 = two ops, 3 = three ops.
        num_examples: (min, max) range for demonstration pairs per problem.
            Randomized to prevent the model from overfitting to a fixed format.
    """
    if isinstance(num_examples, int):
        n_ex = num_examples
    else:
        n_ex = rng.randint(num_examples[0], num_examples[1])

    # Pick operations (avoid duplicates that cancel out)
    ops = _sample_compatible_ops(rng, op_pool, num_ops=level)

    # Generate diverse inputs (avoid duplicates, ensure spread)
    inputs = _sample_diverse_inputs(rng, count=n_ex + 1)

    # Compute outputs
    def apply_chain(x):
        for op in ops:
            x = op.apply(x)
        return x

    examples = [(inp, apply_chain(inp)) for inp in inputs[:n_ex]]
    query_input = inputs[n_ex]
    query_output = apply_chain(query_input)

    # Validate: rule should not be trivially identity on all examples
    if all(i == o for i, o in examples) and query_input == query_output:
        return generate_problem(rng, op_pool, level, num_examples)

    # Validate: rule should produce at least 3 distinct outputs (non-degenerate)
    unique_outputs = set(o for _, o in examples)
    if len(unique_outputs) < 3:
        return generate_problem(rng, op_pool, level, num_examples)

    return BitOpsProblem(
        rule_ops=ops,
        examples=examples,
        query_input=query_input,
        query_output=query_output,
        level=level,
    )


def _op_family(name: str) -> str:
    """Get the operation family to avoid combining inverse/redundant ops.

    Opposite directions are merged into one family so that e.g.
    rotate_left_2 + rotate_right_2 (= identity) can never co-occur.
    """
    if name.startswith("shift_left") or name.startswith("shift_right"):
        return "shift"
    if name.startswith("rotate_left") or name.startswith("rotate_right"):
        return "rotate"
    if name.startswith("xor_"):
        return "xor"
    if name.startswith("and_"):
        return "and"
    if name.startswith("or_"):
        return "or"
    return name


def _sample_compatible_ops(rng: random.Random, pool: list[AtomicOp],
                           num_ops: int) -> list[AtomicOp]:
    """Sample num_ops operations from different families.

    Avoids degenerate combos like shift_left_1 + shift_left_3 (= shift_left_4),
    or NOT + NOT (= identity).
    """
    chosen = []
    used_families = set()
    available = list(pool)
    for _ in range(num_ops):
        op = rng.choice(available)
        chosen.append(op)
        family = _op_family(op.name)
        used_families.add(family)
        # Remove all ops from the same family
        available = [o for o in available if _op_family(o.name) not in used_families]
        if not available:
            # Fallback: reset but still avoid exact duplicates
            available = [o for o in pool if o.name not in {c.name for c in chosen}]
        if not available:
            available = list(pool)
    return chosen


def _sample_diverse_inputs(rng: random.Random, count: int) -> list[int]:
    """Sample diverse 8-bit inputs with good bit pattern coverage."""
    inputs = set()
    # Seed with some structured values
    structured = [0b00000000, 0b11111111, 0b10101010, 0b01010101,
                  0b11110000, 0b00001111, 0b10000001, 0b01111110]
    rng.shuffle(structured)
    for v in structured[:min(3, count)]:
        inputs.add(v)

    # Fill rest randomly
    while len(inputs) < count:
        inputs.add(rng.randint(0, 255))

    result = list(inputs)
    rng.shuffle(result)
    return result[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Step C: Chain-of-Thought Generation (Hypothesis Testing Style)
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(x: int) -> str:
    """Format 8-bit int as spaced binary string (tokenizer-friendly)."""
    return " ".join(format(x, '08b'))


def fmt_dense(x: int) -> str:
    """Format 8-bit int as dense binary string (for prompt compatibility)."""
    return format(x, '08b')


def _popcount(x: int) -> int:
    """Count the number of 1-bits in an 8-bit integer."""
    return bin(x).count('1')


def _generate_heuristic_analysis(problem: BitOpsProblem) -> list[str]:
    """Generate heuristic feature analysis to narrow down operation families.

    Analyzes hamming weight preservation, parity, and bit-count changes
    to logically prune the search space before hypothesis testing.
    """
    ex = problem.examples
    lines = []

    # ── Hamming weight (popcount) analysis ───────────────────────────────────
    weights_preserved = all(_popcount(inp) == _popcount(out) for inp, out in ex)
    weight_diffs = [_popcount(out) - _popcount(inp) for inp, out in ex]
    constant_diff = len(set(weight_diffs)) == 1

    lines.append("Let me count the number of '1' bits (Hamming weight) in each "
                 "input and output to narrow down the operation type.")
    lines.append("")

    for i in range(min(4, len(ex))):
        inp, out = ex[i]
        hw_in, hw_out = _popcount(inp), _popcount(out)
        lines.append(f"  Example {i+1}: Input has {hw_in} ones, "
                     f"Output has {hw_out} ones"
                     f"{' (same)' if hw_in == hw_out else ''}")
    lines.append("")

    if weights_preserved:
        lines.append("Observation: The number of '1' bits is PRESERVED across all examples.")
        lines.append("This rules out AND, OR, XOR with non-zero mask, and NOT — "
                     "these operations change the bit count.")
        lines.append("The rule must be a permutation: rotations, shifts (only if no "
                     "bits fall off), bit reversal, or swaps.")
    elif constant_diff and weight_diffs[0] != 0:
        d = weight_diffs[0]
        direction = "increases" if d > 0 else "decreases"
        lines.append(f"Observation: The bit count consistently {direction} by "
                     f"{abs(d)}. This suggests an OR or AND mask operation "
                     f"is involved.")
    else:
        lines.append("Observation: The bit count changes are not constant — "
                     "the rule likely involves XOR, NOT, or a combination of "
                     "operations that affect bits non-uniformly.")

    lines.append("")

    # ── Parity analysis ──────────────────────────────────────────────────────
    parity_preserved = all(
        _popcount(inp) % 2 == _popcount(out) % 2 for inp, out in ex
    )
    if parity_preserved and not weights_preserved:
        lines.append("Additionally, the parity (odd/even count of 1s) is preserved, "
                     "which is consistent with XOR by an even-weight mask or "
                     "double-NOT-like operations.")
        lines.append("")
    elif not parity_preserved:
        lines.append("The parity of 1-bits flips in some examples, which means "
                     "NOT or XOR with an odd-weight mask is likely involved.")
        lines.append("")

    return lines


def _generate_code_verification(problem: BitOpsProblem) -> list[str]:
    """Generate a Python code block that verifies the discovered rule.

    Code-as-Thought: expressing the rule as executable code improves
    the model's structured reasoning for discrete math tasks.
    """
    ops = problem.rule_ops
    ex = problem.examples
    lines = []

    lines.append("To be absolutely certain, let me express this rule as code:")
    lines.append("")
    lines.append("```python")
    lines.append("def apply_rule(x):")

    for i, op in enumerate(ops):
        comment = f"# Step {i+1}: {op.description}"
        name = op.name

        if name.startswith("shift_left_"):
            n = op.params["n"]
            lines.append(f"    x = (x << {n}) & 0xFF  {comment}")
        elif name.startswith("shift_right_"):
            n = op.params["n"]
            lines.append(f"    x = (x >> {n}) & 0xFF  {comment}")
        elif name.startswith("rotate_left_"):
            n = op.params["n"]
            lines.append(f"    x = ((x << {n}) | (x >> {8 - n})) & 0xFF  {comment}")
        elif name.startswith("rotate_right_"):
            n = op.params["n"]
            lines.append(f"    x = ((x >> {n}) | (x << {8 - n})) & 0xFF  {comment}")
        elif name == "bitwise_not":
            lines.append(f"    x = (~x) & 0xFF  {comment}")
        elif name.startswith("xor_"):
            mask = op.params["mask"]
            lines.append(f"    x = x ^ 0b{format(mask, '08b')}  {comment}")
        elif name.startswith("and_"):
            mask = op.params["mask"]
            lines.append(f"    x = x & 0b{format(mask, '08b')}  {comment}")
        elif name.startswith("or_"):
            mask = op.params["mask"]
            lines.append(f"    x = x | 0b{format(mask, '08b')}  {comment}")
        elif name == "reverse_bits":
            lines.append(f"    x = int('{{:08b}}'.format(x)[::-1], 2)  {comment}")
        elif name == "swap_nibbles":
            lines.append(f"    x = ((x & 0x0F) << 4) | ((x >> 4) & 0x0F)  {comment}")
        elif name == "swap_adjacent_pairs":
            lines.append(f"    x = ((x & 0x33) << 2) | ((x >> 2) & 0x33)  {comment}")
        elif name == "swap_adjacent_bits":
            lines.append(f"    x = ((x & 0x55) << 1) | ((x >> 1) & 0x55)  {comment}")

    lines.append("    return x")
    lines.append("```")
    lines.append("")

    # Verify on 2-3 examples
    lines.append("Quick verification:")
    for i in range(min(3, len(ex))):
        inp, expected = ex[i]
        lines.append(f"  apply_rule(0b{fmt_dense(inp)}) = "
                     f"0b{fmt_dense(problem.apply_rule(inp))} "
                     f"{'== expected' if problem.apply_rule(inp) == expected else '!= expected'}")

    return lines


def generate_cot(problem: BitOpsProblem, rng: random.Random) -> str:
    """Generate a detective-style CoT reasoning trace for the given problem.

    The CoT follows a hypothesis-testing structure:
    1. Observe patterns in the examples
    2. Heuristic feature analysis (hamming weight, parity) to prune search
    3. Test wrong hypotheses first (and show why they fail)
    4. Discover the correct rule through systematic elimination
    5. Code-as-Thought verification
    6. Verify the rule on all examples
    7. Apply the rule to the query
    """
    lines = []
    ex = problem.examples
    ops = problem.rule_ops
    query_in = problem.query_input
    query_out = problem.query_output

    lines.append("Let me analyze these input-output pairs to find the hidden rule.")
    lines.append("")

    # ── Phase 1: Initial observation ─────────────────────────────────────────
    lines.append("**Step 1: Initial Observation**")
    lines.append(f"We have {len(ex)} examples of 8-bit transformations. "
                 "Let me look for patterns.")
    lines.append("")

    # Show a couple of examples in spaced format for clarity
    for i in range(min(3, len(ex))):
        inp, out = ex[i]
        lines.append(f"  Example {i+1}: {fmt(inp)} -> {fmt(out)}")
    lines.append("")

    # ── Phase 1.5: Heuristic feature analysis ────────────────────────────────
    lines.append("**Step 2: Bit Count Analysis (Heuristics)**")
    lines.extend(_generate_heuristic_analysis(problem))

    # ── Phase 2: Test wrong hypotheses ───────────────────────────────────────
    wrong_hypotheses = _generate_wrong_hypotheses(problem, rng)
    step_offset = 3  # next step number after heuristics
    for hyp_idx, (hyp_name, hyp_test_fn, hyp_description) in enumerate(wrong_hypotheses):
        step = step_offset + hyp_idx
        lines.append(f"**Step {step}: Test hypothesis — {hyp_name}**")
        lines.append(f"Let me check if the rule is: {hyp_description}")

        # Test on first 2-3 examples
        all_match = True
        for i in range(min(3, len(ex))):
            inp, expected_out = ex[i]
            hyp_result = hyp_test_fn(inp)
            match = (hyp_result == expected_out)
            status = "matches!" if match else "DOES NOT match"
            lines.append(f"  Example {i+1}: {fmt(inp)} -> {fmt(hyp_result)} "
                         f"(expected {fmt(expected_out)}) — {status}")
            if not match:
                all_match = False
                lines.append(f"  This hypothesis fails on example {i+1}. Moving on.")
                break

        if all_match:
            lines.append("  Wait, this seems to work... let me check more examples.")
            lines.append("  Actually, checking further examples reveals mismatches.")

        lines.append("")

    # ── Phase 3: Discover correct rule ───────────────────────────────────────
    step_num = step_offset + len(wrong_hypotheses)
    lines.append(f"**Step {step_num}: Deeper analysis — finding the actual rule**")

    if len(ops) == 1:
        lines.extend(_explain_discovery_single(problem, rng))
    elif len(ops) == 2:
        lines.extend(_explain_discovery_two(problem, rng))
    else:
        lines.extend(_explain_discovery_three(problem, rng))

    lines.append("")

    # ── Phase 3.5: Code-as-Thought verification ─────────────────────────────
    step_num += 1
    lines.append(f"**Step {step_num}: Code Verification**")
    lines.extend(_generate_code_verification(problem))
    lines.append("")

    # ── Phase 4: Verify on all examples ──────────────────────────────────────
    step_num += 1
    lines.append(f"**Step {step_num}: Full Verification**")
    lines.append(f"Let me verify: the rule is **{problem.rule_description}**.")
    lines.append("")

    for i, (inp, expected_out) in enumerate(ex):
        computed = problem.apply_rule(inp)
        lines.append(f"  Example {i+1}: {fmt(inp)} -> {fmt(computed)} "
                     f"{'✓' if computed == expected_out else '✗'}")
    lines.append("")
    lines.append("All examples verified successfully!")
    lines.append("")

    # ── Phase 5: Apply to query ──────────────────────────────────────────────
    step_num += 1
    lines.append(f"**Step {step_num}: Apply the rule to the query input**")
    lines.append(f"Input: {fmt(query_in)}")
    lines.append("")

    # Show step-by-step application
    current = query_in
    for i, op in enumerate(ops):
        result = op.apply(current)
        lines.append(f"  Operation {i+1} ({op.description}): "
                     f"{fmt(current)} -> {fmt(result)}")
        current = result

    lines.append("")
    lines.append(f"Final answer: {fmt_dense(query_out)}")

    return "\n".join(lines)


def _generate_wrong_hypotheses(problem: BitOpsProblem,
                               rng: random.Random) -> list[tuple]:
    """Generate 2-3 plausible but wrong hypotheses to test.

    Returns list of (name, test_fn, description).
    """
    actual_op_names = {op.name for op in problem.rule_ops}
    hypotheses = []

    # Pool of common wrong hypotheses
    candidates = []

    # Simple XOR with computed "constant"
    inp0, out0 = problem.examples[0]
    xor_val = inp0 ^ out0
    if f"xor_{_fmt_bin(xor_val)}" not in actual_op_names:
        candidates.append((
            "constant XOR mask",
            lambda x, m=xor_val: (x ^ m) & 0xFF,
            f"XOR every input with the constant mask {_fmt_bin(xor_val)}",
        ))

    # Simple NOT
    if "bitwise_not" not in actual_op_names:
        candidates.append((
            "bitwise NOT",
            lambda x: (~x) & 0xFF,
            "apply bitwise NOT to the input",
        ))

    # Left shift by 1
    if "shift_left_1" not in actual_op_names:
        candidates.append((
            "left shift by 1",
            lambda x: (x << 1) & 0xFF,
            "shift all bits left by 1 position",
        ))

    # Right shift by 1
    if "shift_right_1" not in actual_op_names:
        candidates.append((
            "right shift by 1",
            lambda x: (x >> 1) & 0xFF,
            "shift all bits right by 1 position",
        ))

    # Rotate left by 1
    if "rotate_left_1" not in actual_op_names:
        candidates.append((
            "circular left rotation by 1",
            lambda x: ((x << 1) | (x >> 7)) & 0xFF,
            "rotate all bits left by 1 position (circular)",
        ))

    # Reverse bits
    if "reverse_bits" not in actual_op_names:
        candidates.append((
            "bit reversal",
            lambda x: reverse_bits(x),
            "reverse the order of all 8 bits",
        ))

    # Swap nibbles
    if "swap_nibbles" not in actual_op_names:
        candidates.append((
            "nibble swap",
            lambda x: swap_nibbles(x),
            "swap the upper 4 bits and lower 4 bits",
        ))

    # Filter out any that accidentally match the actual rule
    filtered = []
    for name, fn, desc in candidates:
        matches_all = True
        for inp, out in problem.examples[:4]:
            if fn(inp) != out:
                matches_all = False
                break
        if not matches_all:
            filtered.append((name, fn, desc))

    rng.shuffle(filtered)
    # Return 2-3 wrong hypotheses
    count = min(rng.randint(2, 3), len(filtered))
    return filtered[:count]


def _explain_discovery_single(problem: BitOpsProblem,
                              rng: random.Random) -> list[str]:
    """Explain how to discover a single-operation rule."""
    op = problem.rule_ops[0]
    ex = problem.examples
    lines = []

    lines.append("Let me try a more systematic approach and test each operation type.")
    lines.append("")

    # Show the "aha" moment
    lines.append(f"Testing: {op.description}")
    for i in range(min(4, len(ex))):
        inp, expected = ex[i]
        computed = op.apply(inp)
        lines.append(f"  {fmt(inp)} -> {fmt(computed)} "
                     f"(expected {fmt(expected)}) — "
                     f"{'match!' if computed == expected else 'no match'}")

    lines.append("")
    lines.append(f"This works! The rule is: **{op.description}**.")

    return lines


def _explain_discovery_two(problem: BitOpsProblem,
                           rng: random.Random) -> list[str]:
    """Explain how to discover a two-operation composition."""
    op1, op2 = problem.rule_ops
    ex = problem.examples
    lines = []

    lines.append("None of the single operations match perfectly. "
                 "The rule might be a composition of two operations.")
    lines.append("")

    # Show partial analysis
    lines.append(f"Let me try applying {op1.description} first:")
    for i in range(min(3, len(ex))):
        inp, expected = ex[i]
        intermediate = op1.apply(inp)
        lines.append(f"  {fmt(inp)} -> {fmt(intermediate)} (after {op1.description})")
    lines.append("")

    lines.append(f"Now comparing these intermediate results with the expected outputs...")
    lines.append(f"If I then apply {op2.description} to the intermediate results:")
    lines.append("")

    for i in range(min(3, len(ex))):
        inp, expected = ex[i]
        intermediate = op1.apply(inp)
        final = op2.apply(intermediate)
        lines.append(f"  {fmt(intermediate)} -> {fmt(final)} "
                     f"(expected {fmt(expected)}) — "
                     f"{'match!' if final == expected else 'no match'}")

    lines.append("")
    lines.append(f"The rule is: first **{op1.description}**, then **{op2.description}**.")

    return lines


def _explain_discovery_three(problem: BitOpsProblem,
                             rng: random.Random) -> list[str]:
    """Explain how to discover a three-operation composition."""
    op1, op2, op3 = problem.rule_ops
    ex = problem.examples
    lines = []

    lines.append("Single operations don't match, and simple two-step combinations "
                 "also fail. Let me try three-step compositions.")
    lines.append("")

    lines.append(f"Attempting: {op1.description} -> {op2.description} -> {op3.description}")
    lines.append("")

    for i in range(min(3, len(ex))):
        inp, expected = ex[i]
        after1 = op1.apply(inp)
        after2 = op2.apply(after1)
        after3 = op3.apply(after2)
        lines.append(f"  {fmt(inp)}")
        lines.append(f"    -> {fmt(after1)} (after {op1.description})")
        lines.append(f"    -> {fmt(after2)} (after {op2.description})")
        lines.append(f"    -> {fmt(after3)} (after {op3.description})")
        lines.append(f"    Expected: {fmt(expected)} — "
                     f"{'match!' if after3 == expected else 'no match'}")
        lines.append("")

    lines.append(f"The rule is a three-step chain: "
                 f"**{op1.description}** -> **{op2.description}** -> **{op3.description}**.")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
{examples}

Now, determine the output for: {query}"""


def format_prompt(problem: BitOpsProblem) -> str:
    """Format the problem into the competition-style prompt."""
    example_lines = []
    for inp, out in problem.examples:
        example_lines.append(f"{fmt_dense(inp)} -> {fmt_dense(out)}")
    return PROMPT_TEMPLATE.format(
        examples="\n".join(example_lines),
        query=fmt_dense(problem.query_input),
    )


def format_answer_with_cot(cot_text: str, answer: str) -> str:
    r"""Format the full answer: CoT reasoning followed by \boxed{answer}."""
    return f"{cot_text}\n\n\\boxed{{{answer}}}"


def format_answer_direct(answer: str) -> str:
    r"""Format direct answer (no CoT): just \boxed{answer}."""
    return f"\\boxed{{{answer}}}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Generate Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    num_samples: int,
    levels: list[int],
    seed: int = 42,
    num_examples_range: tuple[int, int] = (6, 10),
) -> list[dict]:
    """Generate a full CoT dataset for bit operations.

    Returns list of dicts with keys: id, prompt, answer, cot_answer,
    level, rule_description.
    """
    rng = random.Random(seed)
    op_pool = make_op_pool(rng)

    # Distribute samples across levels
    samples_per_level = num_samples // len(levels)
    remainder = num_samples % len(levels)

    dataset = []
    for level_idx, level in enumerate(levels):
        count = samples_per_level + (1 if level_idx < remainder else 0)

        for _ in range(count):
            problem = generate_problem(rng, op_pool, level, num_examples_range)

            prompt = format_prompt(problem)
            answer = fmt_dense(problem.query_output)
            cot = generate_cot(problem, rng)

            sample_id = uuid.uuid4().hex[:8]
            dataset.append({
                "id": sample_id,
                "prompt": prompt,
                "answer": answer,
                "cot_answer": format_answer_with_cot(cot, answer),
                "level": level,
                "rule_description": problem.rule_description,
            })

    rng.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], output_dir: Path):
    """Save dataset in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full JSONL (all fields)
    jsonl_path = output_dir / "bit_ops_cot.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[saved] {jsonl_path} ({len(dataset)} samples)")

    # 2. Competition-format CSV (id, prompt, answer) — direct answer only
    csv_direct_path = output_dir / "bit_ops_direct.csv"
    with open(csv_direct_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"], sample["answer"]])
    print(f"[saved] {csv_direct_path}")

    # 3. CoT-format CSV (id, prompt, answer) — answer includes CoT
    csv_cot_path = output_dir / "bit_ops_cot.csv"
    with open(csv_cot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"], sample["cot_answer"]])
    print(f"[saved] {csv_cot_path}")

    # 4. Per-level splits
    by_level = {}
    for sample in dataset:
        by_level.setdefault(sample["level"], []).append(sample)

    for level, samples in sorted(by_level.items()):
        level_path = output_dir / f"level_{level}.jsonl"
        with open(level_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"[saved] {level_path} ({len(samples)} samples, level {level})")

    # 5. Stats
    stats = {
        "total_samples": len(dataset),
        "by_level": {level: len(samples) for level, samples in sorted(by_level.items())},
        "seed": dataset[0].get("seed", "unknown"),
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[saved] {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT training data for bit manipulation tasks.")
    parser.add_argument("--output_dir", type=str, default="data/cot/bit_ops",
                        help="Output directory for generated data.")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Total number of samples to generate.")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
                        help="Curriculum levels to include (1=single op, 2=two ops, 3=three ops).")
    parser.add_argument("--min_examples", type=int, default=6,
                        help="Minimum number of demo examples per problem.")
    parser.add_argument("--max_examples", type=int, default=10,
                        help="Maximum number of demo examples per problem.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        # Relative to project root
        output_dir = Path("/scratch/jwj/NemotronFT") / output_dir

    print(f"Generating {args.num_samples} bit-ops CoT samples "
          f"(levels={args.levels}, seed={args.seed})...")
    dataset = generate_dataset(
        num_samples=args.num_samples,
        levels=args.levels,
        seed=args.seed,
        num_examples_range=(args.min_examples, args.max_examples),
    )
    save_dataset(dataset, output_dir)
    print(f"\nDone! {len(dataset)} samples saved to {output_dir}/")

    # Print a sample
    print("\n" + "=" * 70)
    print("SAMPLE PREVIEW (first entry)")
    print("=" * 70)
    sample = dataset[0]
    print(f"Level: {sample['level']}")
    print(f"Rule: {sample['rule_description']}")
    print(f"\n--- Prompt ---\n{sample['prompt']}")
    print(f"\n--- CoT Answer ---\n{sample['cot_answer'][:1500]}...")


if __name__ == "__main__":
    main()
