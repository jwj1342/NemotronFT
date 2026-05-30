"""Cryptarithm puzzle "structural fingerprints" — measurement DSL.

The hypothesis we are validating: a small set of structural measurements on a
cryptarithm puzzle's few-shot examples carries enough information to
substantially narrow the hypothesis space for what transformation rule is
being used.  If the hypothesis is correct, then:

  1. Same-rule puzzles should cluster in measurement space.
  2. The measurements should distinguish "Huikang's solver covers this" from
     "Huikang's solver fails on this."
  3. The chain of measurements can be verbalized as a CoT trace that teaches
     a model HOW to look at a puzzle, not WHICH puzzle to memorize.

Public surface: `measure(puzzle: ParsedPuzzle) -> Measurements`, plus the
parser entry point `parse_prompt`.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

ARITH_OPS = set("+-*/")


# ---------------------------------------------------------------------------
# Parsing — generic, handles both "operator at pos 2" and "operator hidden"
# ---------------------------------------------------------------------------


@dataclass
class Example:
    raw: str
    lhs: str
    rhs: str


@dataclass
class ParsedPuzzle:
    examples: list[Example]
    query: str
    answer: Optional[str] = None  # for analysis only; not used by solver


def parse_prompt(prompt: str, answer: Optional[str] = None) -> Optional[ParsedPuzzle]:
    """Robust to either 'Below are a few examples:\n' header style."""
    if "Below are a few examples:\n" not in prompt:
        return None
    body = prompt.split("Below are a few examples:\n", 1)[1]
    if "\nNow, determine the result for: " not in body:
        return None
    ex_section, query_part = body.split("\nNow, determine the result for: ", 1)
    examples: list[Example] = []
    for line in ex_section.strip().splitlines():
        if " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        examples.append(Example(raw=line, lhs=lhs, rhs=rhs))
    return ParsedPuzzle(examples=examples, query=query_part.strip(), answer=answer)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


@dataclass
class Measurements:
    # --- Structural ---
    n_examples: int = 0
    lhs_len: int = 0                          # common LHS length (e.g., 5)
    lhs_len_uniform: bool = False
    op_position: Optional[int] = None         # which LHS index is the operator
    op_chars_set: list[str] = field(default_factory=list)  # operators appearing at op_position
    op_uniform: bool = False                  # exactly one op symbol across all examples

    # --- Lengths / variance ---
    rhs_lens: list[int] = field(default_factory=list)
    rhs_len_var: float = 0.0
    rhs_len_min: int = 0
    rhs_len_max: int = 0
    rhs_len_uniform: bool = False

    # --- Alphabet ---
    alphabet: list[str] = field(default_factory=list)  # sorted union of all digit-chars
    alphabet_size: int = 0
    alphabet_contains_op_char: bool = False   # whether op char also appears as digit elsewhere

    # --- Character preservation ---
    rhs_in_lhs_rate: float = 0.0  # fraction of rhs chars that appear in their own lhs (non-op)
    rhs_uses_new_chars: bool = False
    new_chars: list[str] = field(default_factory=list)

    # --- Operator-symbol behavior ---
    op_in_rhs_count: int = 0   # examples where the operator symbol appears in RHS
    op_only_rhs: bool = False  # examples where RHS contains ONLY the operator char (negative-like)

    # --- Per-operator output-length pattern ---
    rhs_len_by_op: dict[str, list[int]] = field(default_factory=dict)

    # --- Differential ---
    diff_pairs: list[tuple[int, int, int]] = field(default_factory=list)  # (i,j, |ΔP|, |ΔC|) - flattened (i,j, ΔP, ΔC)
    diff_amplifies: int = 0  # pairs where |ΔC| > |ΔP|
    diff_collapses: int = 0  # pairs where |ΔC| < |ΔP|
    diff_preserves: int = 0  # pairs where |ΔC| == |ΔP| and both > 0

    # --- Position correlation: does rhs[j] ever equal lhs_non_op[i] for fixed (i,j)? ---
    position_lock: bool = False  # there is a fixed permutation π such that rhs[π(i)] == lhs_non_op[i] for all examples
    position_lock_perm: Optional[list[int]] = None

    # --- Concat / reverse-concat signature ---
    is_pure_concat: bool = False
    is_pure_revconcat: bool = False

    # --- Frequency ---
    char_freq: dict[str, int] = field(default_factory=dict)
    char_freq_entropy: float = 0.0

    # --- Hypothesis votes ---
    hyp_digit_cipher_arith: bool = False
    hyp_string_dsl: bool = False
    hyp_permutation: bool = False
    hyp_concat_family: bool = False
    hyp_operator_encoded: bool = False

    def summary(self) -> str:
        """One-line human-readable summary."""
        votes = []
        if self.hyp_digit_cipher_arith: votes.append("digit-cipher-arith")
        if self.hyp_concat_family: votes.append("concat-fam")
        if self.hyp_permutation: votes.append("permutation")
        if self.hyp_string_dsl: votes.append("string-dsl")
        if self.hyp_operator_encoded: votes.append("op-encoded")
        return (
            f"n={self.n_examples} lhs={self.lhs_len} alpha={self.alphabet_size} "
            f"rhs_lens={self.rhs_lens} rhs_in_lhs={self.rhs_in_lhs_rate:.2f} "
            f"ops={'/'.join(self.op_chars_set)} "
            f"votes=[{','.join(votes) or '?'}]"
        )


def measure(p: ParsedPuzzle) -> Measurements:
    m = Measurements()
    m.n_examples = len(p.examples)
    if not p.examples:
        return m

    # Structural
    lens = [len(e.lhs) for e in p.examples]
    m.lhs_len_uniform = len(set(lens)) == 1
    m.lhs_len = lens[0] if m.lhs_len_uniform else 0

    # Operator position: for cryptarithm-style puzzles, the convention is
    # position 2 of a 5-char LHS, regardless of whether the symbol is +-*/.
    # We use query's position 2 if uniform-len, else fall back to the position
    # where the example chars are most "consistent" with being a delimiter.
    if m.lhs_len_uniform and m.lhs_len == 5 and len(p.query) == 5:
        m.op_position = 2
        ops = [e.lhs[m.op_position] for e in p.examples]
        m.op_chars_set = sorted(set(ops))
        m.op_uniform = len(m.op_chars_set) == 1

    # Per-example, what are the non-op LHS chars and the RHS?
    def non_op_lhs(e: Example) -> str:
        if m.op_position is None:
            return e.lhs
        return e.lhs[: m.op_position] + e.lhs[m.op_position + 1 :]

    # RHS length analysis
    rlens = [len(e.rhs) for e in p.examples]
    m.rhs_lens = rlens
    m.rhs_len_min = min(rlens)
    m.rhs_len_max = max(rlens)
    m.rhs_len_uniform = len(set(rlens)) == 1
    m.rhs_len_var = statistics.pvariance(rlens) if len(rlens) > 1 else 0.0

    # Alphabet
    alphabet: set[str] = set()
    for e in p.examples:
        alphabet |= set(non_op_lhs(e))
        alphabet |= set(e.rhs)
    alphabet |= set(c for c in p.query if c not in ARITH_OPS or m.op_position is None)
    if m.op_position is not None and len(p.query) == m.lhs_len:
        alphabet -= {p.query[m.op_position]}  # remove the query operator char
    m.alphabet = sorted(alphabet)
    m.alphabet_size = len(alphabet)

    if m.op_position is not None:
        for op in m.op_chars_set:
            if op in alphabet:
                m.alphabet_contains_op_char = True

    # Character preservation
    preserved_total = 0
    rhs_total = 0
    new_chars_set: set[str] = set()
    for e in p.examples:
        nol = set(non_op_lhs(e))
        for c in e.rhs:
            rhs_total += 1
            if c in nol:
                preserved_total += 1
            else:
                new_chars_set.add(c)
    m.rhs_in_lhs_rate = preserved_total / rhs_total if rhs_total > 0 else 0.0
    m.rhs_uses_new_chars = len(new_chars_set) > 0
    m.new_chars = sorted(new_chars_set)

    # Operator symbol in RHS
    if m.op_position is not None:
        for e in p.examples:
            op = e.lhs[m.op_position] if m.op_position < len(e.lhs) else ""
            if op in e.rhs:
                m.op_in_rhs_count += 1
            if e.rhs and all(c == op for c in e.rhs):
                m.op_only_rhs = True

    # Per-operator RHS length pattern
    if m.op_position is not None:
        for e in p.examples:
            op = e.lhs[m.op_position] if m.op_position < len(e.lhs) else "?"
            m.rhs_len_by_op.setdefault(op, []).append(len(e.rhs))

    # Differential analysis: for every pair of examples, compute |ΔP|, |ΔC|
    def hamming(a: str, b: str) -> int:
        if len(a) != len(b):
            # treat as full difference: max(len) — min(len common matches)
            n = min(len(a), len(b))
            diff = abs(len(a) - len(b))
            return sum(1 for i in range(n) if a[i] != b[i]) + diff
        return sum(1 for x, y in zip(a, b) if x != y)

    for i in range(len(p.examples)):
        for j in range(i + 1, len(p.examples)):
            dp = hamming(non_op_lhs(p.examples[i]), non_op_lhs(p.examples[j]))
            dc = hamming(p.examples[i].rhs, p.examples[j].rhs)
            m.diff_pairs.append((i, j, dp + dc * 0))  # storing tuple for record
            if dc > dp:
                m.diff_amplifies += 1
            elif dc < dp:
                m.diff_collapses += 1
            elif dc == dp and dp > 0:
                m.diff_preserves += 1

    # Position lock: is there a permutation π such that rhs[π(i)] == lhs_non_op[i] for all examples?
    # Only well-defined when all examples share lhs_non_op length and rhs length.
    if m.op_position is not None and m.rhs_len_uniform:
        nol_lens = {len(non_op_lhs(e)) for e in p.examples}
        if len(nol_lens) == 1:
            L = list(nol_lens)[0]
            R = m.rhs_lens[0]
            # For each output position j, check if there's a single input position i such that
            # rhs[j] == lhs_non_op[i] across ALL examples
            perm: list[Optional[int]] = []
            for j in range(R):
                candidate_i = None
                for i in range(L):
                    if all(p.examples[k].rhs[j] == non_op_lhs(p.examples[k])[i] for k in range(len(p.examples))):
                        candidate_i = i
                        break
                perm.append(candidate_i)
            if all(x is not None for x in perm):
                m.position_lock = True
                m.position_lock_perm = perm  # type: ignore

    # Concat / rev-concat detection across all 4-char-rhs examples (most relaxed: majority)
    if m.op_position is not None and m.lhs_len == 5:
        fwd_matches = 0
        rev_matches = 0
        total_4 = 0
        for e in p.examples:
            if len(e.rhs) != 4:
                continue
            total_4 += 1
            nol = non_op_lhs(e)
            if e.rhs == nol:
                fwd_matches += 1
            if e.rhs == nol[2:] + nol[:2]:
                rev_matches += 1
        m.is_pure_concat = total_4 >= 2 and fwd_matches == total_4
        m.is_pure_revconcat = total_4 >= 2 and rev_matches == total_4

    # Frequency / entropy
    counter: Counter = Counter()
    for e in p.examples:
        counter.update(non_op_lhs(e))
        counter.update(e.rhs)
    m.char_freq = dict(counter)
    total = sum(counter.values())
    if total > 0:
        m.char_freq_entropy = -sum((v / total) * math.log2(v / total) for v in counter.values() if v > 0)

    # Hypothesis voting — corrected thresholds based on empirical validation.
    #
    # Critical insight from data: for true base-10 digit-cipher arithmetic,
    # the expected preservation rate is ~0.4 (the 4 distinct chars per 5-char
    # LHS overlap with result digits at random ~40%). Concat/permutation
    # rules give HIGHER preservation (output is copy/rearrangement of input).
    m.hyp_concat_family = m.is_pure_concat or m.is_pure_revconcat

    m.hyp_permutation = m.position_lock and not m.hyp_concat_family

    # Digit-cipher arithmetic signature:
    #   - alphabet within ~10 digits worth
    #   - preservation in the "mid" zone (0.2-0.7 — output is derived, not copied)
    #   - rhs lengths VARY (arithmetic produces different magnitudes)
    m.hyp_digit_cipher_arith = (
        m.alphabet_size <= 12
        and 0.20 <= m.rhs_in_lhs_rate <= 0.75
        and not m.hyp_concat_family
        and (m.rhs_len_var > 0.0 or m.rhs_len_max != m.rhs_len_min)
    )

    # Pure string DSL: alphabet much larger than 10 (so not a digit cipher),
    # OR preservation extreme (concat-like)
    m.hyp_string_dsl = (
        (m.alphabet_size > 12)
        or (m.rhs_in_lhs_rate > 0.85 and not m.hyp_concat_family)
    )

    # Operator-encoded: when the operator char appears in the RHS for ≥half the examples
    m.hyp_operator_encoded = m.op_in_rhs_count >= max(1, m.n_examples // 2)

    return m


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
    from category import detect_category  # type: ignore  # noqa: E402

    here = Path(__file__).resolve()
    data_path = here.parents[3] / "data" / "Flash-COT-Filtered" / "equation_symbolic" / "level_1.jsonl"

    by_cat: dict[str, list[ParsedPuzzle]] = {"cryptarithm_deduce": [], "cryptarithm_guess": []}
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            c = detect_category(r["prompt"])
            if c in by_cat:
                p = parse_prompt(r["prompt"], answer=r.get("answer"))
                if p:
                    by_cat[c].append(p)

    for cat, plist in by_cat.items():
        print(f"\n=== {cat}: {len(plist)} puzzles ===")
        for p in plist[:3]:
            m = measure(p)
            print(p.examples[0].raw[:60], "→", m.summary())
