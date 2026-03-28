"""
cot_synthesis/cipher.py — Synthesize CoT training data for cipher/decryption tasks.

Generates monoalphabetic substitution cipher problems where encrypted text must
be decrypted using example pairs. Each sample includes:
  - A prompt (matching the competition format)
  - A cryptanalysis-style Chain-of-Thought decryption process
  - The final answer

Supports curriculum learning:
  - Level 1: Caesar cipher (constant shift) — simple pattern recognition
  - Level 2: Random monoalphabetic substitution, 5-6 examples — ample evidence
  - Level 3: Random monoalphabetic substitution, 3-4 examples — less evidence

Usage:
    python cot_synthesis/cipher.py \
        --output_dir data/cot/cipher \
        --num_samples 500 \
        --levels 1 2 3 \
        --seed 42
"""

import argparse
import csv
import json
import random
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Step A: Word Pool & Cipher Utilities
# ═══════════════════════════════════════════════════════════════════════════════

# Word pools matching competition theme (Alice's Wonderland)
NOUNS_SUBJECT = [
    "alice", "bird", "cat", "dragon", "hatter", "king", "knight",
    "mouse", "princess", "queen", "rabbit", "student", "teacher",
    "turtle", "wizard", "fox", "owl", "prince", "guard", "fairy",
    "jester",
]

VERBS = [
    "chases", "creates", "discovers", "draws", "dreams", "explores",
    "follows", "found", "imagines", "reads", "sees", "studies",
    "watches", "writes", "guards", "solves", "opens", "hides",
    "paints", "builds",
]

ADJECTIVES = [
    "ancient", "clever", "colorful", "curious", "dark", "golden",
    "magical", "mysterious", "secret", "silver", "wise", "brave",
    "hidden", "enchanted", "tiny", "giant", "royal", "frozen",
    "sacred", "crystal",
]

NOUNS_OBJECT = [
    "book", "castle", "door", "forest", "garden", "key",
    "library", "mirror", "mountain", "ocean", "palace", "potion",
    "puzzle", "secret", "story", "valley", "village", "wonderland",
    "bridge", "river", "tower", "crown", "scroll", "lantern",
    "jewel", "ring", "sword", "shield", "map", "treasure",
]

PREPOSITIONS = ["in", "inside", "near", "through", "under", "above", "beyond"]


# ── Phrase Generation ────────────────────────────────────────────────────────

def _phrase_svo(rng: random.Random) -> list[str]:
    """subject verb object — 'cat imagines book'"""
    return [rng.choice(NOUNS_SUBJECT), rng.choice(VERBS), rng.choice(NOUNS_OBJECT)]


def _phrase_svpo(rng: random.Random) -> list[str]:
    """subject verb prep object — 'dragon chases in mountain'"""
    return [rng.choice(NOUNS_SUBJECT), rng.choice(VERBS),
            rng.choice(PREPOSITIONS), rng.choice(NOUNS_OBJECT)]


def _phrase_the_asv(rng: random.Random) -> list[str]:
    """the adj subject verb — 'the golden dragon follows'"""
    return ["the", rng.choice(ADJECTIVES), rng.choice(NOUNS_SUBJECT), rng.choice(VERBS)]


def _phrase_sv_the_ao(rng: random.Random) -> list[str]:
    """subject verb the adj object — 'princess reads the mysterious secret'"""
    return [rng.choice(NOUNS_SUBJECT), rng.choice(VERBS), "the",
            rng.choice(ADJECTIVES), rng.choice(NOUNS_OBJECT)]


def _phrase_sv_the_o(rng: random.Random) -> list[str]:
    """subject verb the object — 'mouse sees the door'"""
    return [rng.choice(NOUNS_SUBJECT), rng.choice(VERBS), "the",
            rng.choice(NOUNS_OBJECT)]


PHRASE_TEMPLATES_SHORT = [_phrase_svo]  # 3 words
PHRASE_TEMPLATES_MEDIUM = [_phrase_svo, _phrase_svpo, _phrase_the_asv,
                           _phrase_sv_the_o]  # 3-4 words
PHRASE_TEMPLATES_LONG = [_phrase_svpo, _phrase_the_asv, _phrase_sv_the_ao,
                         _phrase_sv_the_o]  # 4-5 words
PHRASE_TEMPLATES_ALL = [_phrase_svo, _phrase_svpo, _phrase_the_asv,
                        _phrase_sv_the_ao, _phrase_sv_the_o]


def generate_phrase(rng: random.Random, templates=None) -> str:
    """Generate a random plaintext phrase from word pools and templates."""
    if templates is None:
        templates = PHRASE_TEMPLATES_ALL
    template = rng.choice(templates)
    return " ".join(template(rng))


# ── Cipher Utilities ─────────────────────────────────────────────────────────

def make_caesar_cipher(shift: int) -> dict[str, str]:
    """Create a Caesar cipher mapping (plaintext → ciphertext) with given shift."""
    mapping = {}
    for c in string.ascii_lowercase:
        mapping[c] = chr((ord(c) - ord('a') + shift) % 26 + ord('a'))
    return mapping


def make_random_substitution(rng: random.Random) -> dict[str, str]:
    """Create a random monoalphabetic substitution cipher (derangement).

    Guarantees no letter maps to itself, making the cipher non-trivial.
    """
    letters = list(string.ascii_lowercase)
    shuffled = list(string.ascii_lowercase)
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(letters, shuffled)):
            break
    return dict(zip(letters, shuffled))


def encrypt_text(plaintext: str, mapping: dict[str, str]) -> str:
    """Encrypt plaintext letter by letter using the mapping (spaces preserved)."""
    return "".join(mapping.get(c, c) for c in plaintext)


def _letter_pos(c: str) -> int:
    """Return 0-based position of a lowercase letter (a=0, b=1, ..., z=25)."""
    return ord(c) - ord('a')


def _compute_shift(enc_char: str, plain_char: str) -> int:
    """Compute the Caesar shift: (cipher_pos - plain_pos) mod 26."""
    return (_letter_pos(enc_char) - _letter_pos(plain_char)) % 26


# ═══════════════════════════════════════════════════════════════════════════════
# Step B: Problem Generation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CipherProblem:
    """A fully generated cipher problem with examples and query."""
    cipher_type: str                        # "caesar" or "substitution"
    mapping: dict[str, str]                 # plaintext → ciphertext
    examples: list[tuple[str, str]]         # (ciphertext, plaintext) pairs
    query_ciphertext: str                   # encrypted query to solve
    query_plaintext: str                    # ground truth answer
    level: int                              # curriculum level (1/2/3)
    cipher_params: dict = field(default_factory=dict)

    @property
    def reverse_mapping(self) -> dict[str, str]:
        """Ciphertext → plaintext mapping (for decryption)."""
        return {v: k for k, v in self.mapping.items()}

    @property
    def cipher_description(self) -> str:
        if self.cipher_type == "caesar":
            return f"Caesar cipher with shift {self.cipher_params['shift']}"
        return "monoalphabetic substitution cipher"


def generate_problem(rng: random.Random, level: int,
                     num_examples_range: tuple[int, int] = (3, 6)
                     ) -> CipherProblem:
    """Generate a random cipher problem at the given curriculum level.

    Args:
        level: 1 = Caesar, 2 = substitution (many examples),
               3 = substitution (fewer examples).
        num_examples_range: (min, max) range for demonstration pairs.
    """
    if level == 1:
        # Caesar cipher with random shift
        shift = rng.randint(1, 25)
        mapping = make_caesar_cipher(shift)
        cipher_type = "caesar"
        cipher_params = {"shift": shift}
        n_examples = rng.randint(
            max(4, num_examples_range[0]), num_examples_range[1])
        query_templates = PHRASE_TEMPLATES_SHORT + PHRASE_TEMPLATES_MEDIUM
    elif level == 2:
        # Random substitution with ample examples
        mapping = make_random_substitution(rng)
        cipher_type = "substitution"
        cipher_params = {}
        n_examples = rng.randint(
            max(5, num_examples_range[0]), num_examples_range[1])
        query_templates = PHRASE_TEMPLATES_MEDIUM
    else:  # level 3
        # Random substitution with fewer examples
        mapping = make_random_substitution(rng)
        cipher_type = "substitution"
        cipher_params = {}
        n_examples = rng.randint(
            num_examples_range[0], min(4, num_examples_range[1]))
        query_templates = PHRASE_TEMPLATES_LONG

    # Generate example phrases and encrypt them
    examples = []
    used_phrases = set()
    covered_letters = set()  # plaintext letters seen in examples

    for _ in range(n_examples):
        for _attempt in range(50):
            phrase = generate_phrase(rng)
            if phrase not in used_phrases:
                break
        used_phrases.add(phrase)
        encrypted = encrypt_text(phrase, mapping)
        examples.append((encrypted, phrase))
        covered_letters.update(c for c in phrase if c.isalpha())

    # Generate query phrase — all its plaintext letters must appear in examples
    for _attempt in range(500):
        query = generate_phrase(rng, query_templates)
        if query in used_phrases:
            continue
        query_letters = set(c for c in query if c.isalpha())
        if query_letters.issubset(covered_letters):
            break
    else:
        # Fallback: regenerate entire problem (extremely rare)
        return generate_problem(rng, level, num_examples_range)

    query_encrypted = encrypt_text(query, mapping)

    return CipherProblem(
        cipher_type=cipher_type,
        mapping=mapping,
        examples=examples,
        query_ciphertext=query_encrypted,
        query_plaintext=query,
        level=level,
        cipher_params=cipher_params,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step C: Chain-of-Thought Generation (Cryptanalysis Style)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cot(problem: CipherProblem, rng: random.Random) -> str:
    """Generate a cryptanalysis-style CoT reasoning trace.

    The CoT follows a systematic decryption structure:
    1. Observe the example pairs
    2. Extract letter mappings from examples
    3. Test wrong hypothesis first (Atbash for Caesar / Caesar for substitution)
    4. Discover the correct cipher and build mapping
    5. Code-as-Thought verification
    6. Verify on all examples
    7. Decrypt the query word by word
    """
    if problem.cipher_type == "caesar":
        return _generate_cot_caesar(problem, rng)
    else:
        return _generate_cot_substitution(problem, rng)


def _extract_letter_pairs(ciphertext: str, plaintext: str
                          ) -> list[tuple[str, str]]:
    """Extract aligned (encrypted, plain) letter pairs from a text pair."""
    ct_letters = [c for c in ciphertext if c.isalpha()]
    pt_letters = [c for c in plaintext if c.isalpha()]
    return list(zip(ct_letters, pt_letters))


def _build_mapping_from_examples(examples: list[tuple[str, str]]
                                 ) -> dict[str, str]:
    """Build consolidated enc→plain mapping from all example pairs."""
    mapping = {}
    for enc, plain in examples:
        enc_words = enc.split()
        plain_words = plain.split()
        for ew, pw in zip(enc_words, plain_words):
            for ec, pc in zip(ew, pw):
                mapping[ec] = pc
    return mapping


def _find_multi_occurrence_letters(examples: list[tuple[str, str]]
                                   ) -> list[tuple[str, str, list[int]]]:
    """Find encrypted letters appearing in 2+ examples.

    Returns list of (enc_letter, plain_letter, example_indices).
    """
    letter_examples: dict[str, tuple[str, set[int]]] = {}
    for idx, (enc, plain) in enumerate(examples):
        pairs = _extract_letter_pairs(enc, plain)
        for e, p in pairs:
            if e not in letter_examples:
                letter_examples[e] = (p, set())
            letter_examples[e][1].add(idx)

    return [(e, p, sorted(indices))
            for e, (p, indices) in letter_examples.items()
            if len(indices) >= 2]


# ── Phrasing Diversity ────────────────────────────────────────────────────────

_OPENINGS = [
    "Let me analyze the encrypted-decrypted pairs to find the encryption rule.",
    "I need to figure out the encryption pattern from these examples.",
    "Let me work through these examples to crack the cipher.",
    "Looking at the example pairs to determine the encryption method.",
    "I'll compare the encrypted and decrypted text to find the pattern.",
]

_CAESAR_DISCOVERIES = [
    "All shifts are **{shift}**! This is a Caesar cipher with shift {shift}.",
    "Every letter is shifted by exactly **{shift}** — a Caesar cipher.",
    "The shift is constant: **{shift}**. Classic Caesar cipher.",
    "Consistent shift of **{shift}** across all pairs — Caesar cipher confirmed.",
]

_SUB_CONCLUSIONS = [
    ("The shifts are different ({shifts}), so this is NOT a Caesar cipher. "
     "It must be a general monoalphabetic substitution cipher."),
    ("Shifts vary ({shifts}) — not Caesar. This is a random substitution cipher."),
    ("Different shifts ({shifts}) rule out Caesar. "
     "Each letter has its own unique mapping."),
]


# ── Light CoT Generation (abbreviated reasoning, key insights only) ──────────

def generate_light_cot(problem: CipherProblem, rng: random.Random) -> str:
    """Generate abbreviated CoT: 3-5 sentences capturing key insights only.

    Focuses on: method choice → pattern discovery → verification → answer.
    No step numbers, no code blocks, no full mapping table.
    """
    if problem.cipher_type == "caesar":
        return _generate_light_cot_caesar(problem, rng)
    return _generate_light_cot_substitution(problem, rng)


def _generate_light_cot_caesar(problem: CipherProblem,
                                rng: random.Random) -> str:
    """Light CoT for Caesar cipher: ~4 sentences."""
    shift = problem.cipher_params["shift"]
    ex = problem.examples
    first_pairs = _extract_letter_pairs(ex[0][0], ex[0][1])

    lines = []
    lines.append("Comparing encrypted and decrypted letter positions "
                 "to find a shift pattern.")
    lines.append("")

    # Show 3 shift computations
    samples = first_pairs[:3]
    parts = [f"'{e}'→'{p}' (shift {_compute_shift(e, p)})"
             for e, p in samples]
    lines.append(f"From example 1: {', '.join(parts)} — "
                 f"constant shift of {shift}.")
    lines.append("")

    # Quick verification
    verify_pairs = _extract_letter_pairs(ex[1][0], ex[1][1])[:2]
    parts = [f"'{e}'−{shift}='{p}' ✓" for e, p in verify_pairs]
    lines.append(f"Verified on example 2: {', '.join(parts)}. "
                 f"Caesar cipher with shift {shift} confirmed.")
    lines.append("")

    # Decrypt word by word
    enc_words = problem.query_ciphertext.split()
    plain_words = problem.query_plaintext.split()
    word_parts = [f"\"{ew}\"→\"{pw}\"" for ew, pw in
                  zip(enc_words, plain_words)]
    lines.append(f"Decrypting: {', '.join(word_parts)}")

    return "\n".join(lines)


def _generate_light_cot_substitution(problem: CipherProblem,
                                      rng: random.Random) -> str:
    """Light CoT for substitution cipher: ~5 sentences."""
    ex = problem.examples
    all_mapping = _build_mapping_from_examples(ex)

    lines = []
    lines.append("Aligning encrypted and decrypted letters from all "
                 f"{len(ex)} examples to build a substitution mapping.")
    lines.append("")

    # Show shift variation to rule out Caesar
    first_pairs = _extract_letter_pairs(ex[0][0], ex[0][1])
    shifts = [_compute_shift(e, p) for e, p in first_pairs[:4]]
    lines.append(f"Shifts vary ({', '.join(str(s) for s in shifts)}) — "
                 "not Caesar, each letter has its own mapping.")
    lines.append("")

    # Show partial mapping
    sorted_pairs = sorted(all_mapping.items())[:12]
    mapping_str = ", ".join(f"{e}→{p}" for e, p in sorted_pairs)
    lines.append(f"Extracted mapping ({len(all_mapping)} letters): "
                 f"{mapping_str}, ...")
    lines.append("")

    # Quick consistency note
    multi = _find_multi_occurrence_letters(ex)
    if multi:
        e, p, _ = multi[0]
        lines.append(f"Consistency check: '{e}' always maps to '{p}' "
                     "across examples. ✓")
        lines.append("")

    # Decrypt word by word
    enc_words = problem.query_ciphertext.split()
    plain_words = problem.query_plaintext.split()
    word_parts = [f"\"{ew}\"→\"{pw}\"" for ew, pw in
                  zip(enc_words, plain_words)]
    lines.append(f"Decrypting: {', '.join(word_parts)}")

    return "\n".join(lines)


# ── Detour Generators (wrong paths / self-correction) ────────────────────────

def _generate_wrong_shift_detour(problem: CipherProblem,
                                  rng: random.Random) -> list[str]:
    """Generate a wrong-shift hypothesis (e.g. ROT13) that fails.

    The model tries a common shift, gets gibberish, then realizes
    it needs to compute the actual shift value.
    """
    correct_shift = problem.cipher_params["shift"]

    # Pick a plausible wrong shift
    common_shifts = [13, 7, 3, 11, 19, 5]
    candidates = [s for s in common_shifts if s != correct_shift]
    wrong_shift = rng.choice(candidates[:3])

    lines = []
    label = f"ROT{wrong_shift}" if wrong_shift == 13 else f"shift-by-{wrong_shift}"

    lines.append(f"A common cipher is {label}. Let me try decrypting the "
                 f"first example by shifting each letter back by {wrong_shift}:")
    lines.append("")

    # Apply wrong shift to first two words of first example
    enc_words = problem.examples[0][0].split()
    plain_words = problem.examples[0][1].split()
    for ew, pw in zip(enc_words[:2], plain_words[:2]):
        wrong_result = "".join(
            chr((ord(c) - ord('a') - wrong_shift) % 26 + ord('a'))
            if c.isalpha() else c for c in ew)
        lines.append(f"  \"{ew}\" − {wrong_shift} → \"{wrong_result}\" "
                     f"(expected \"{pw}\") — "
                     f"{'matches!' if wrong_result == pw else 'DOES NOT match'}")
    lines.append("")

    lines.append(f"That gives nonsensical results. {label} is not the right "
                 "shift. Let me compute the actual shift value from the "
                 "letter pairs instead of guessing.")
    lines.append("")

    return lines


def _generate_premature_decrypt_detour(problem: CipherProblem,
                                        rng: random.Random) -> list[str]:
    """Generate a premature decryption attempt with incomplete mapping.

    The model eagerly tries to decrypt the query using only the first
    1-2 examples, gets '?' for unmapped letters, and realizes it needs
    to use all examples.
    """
    # Build mapping from only first 1-2 examples
    n_partial = rng.randint(1, min(2, len(problem.examples)))
    partial_mapping = {}
    for enc, plain in problem.examples[:n_partial]:
        for e, p in zip(
            [c for c in enc if c.isalpha()],
            [c for c in plain if c.isalpha()],
        ):
            partial_mapping[e] = p

    # Attempt to decrypt query with partial mapping
    partial_result = []
    unknown_count = 0
    for c in problem.query_ciphertext:
        if c.isalpha():
            if c in partial_mapping:
                partial_result.append(partial_mapping[c])
            else:
                partial_result.append("?")
                unknown_count += 1
        else:
            partial_result.append(c)
    partial_str = "".join(partial_result)

    if unknown_count == 0:
        return []  # partial mapping already covers everything — skip detour

    lines = []
    lines.append(f"With the mappings from the first "
                 f"{n_partial} example{'s' if n_partial > 1 else ''}, "
                 "let me try decrypting the query right away:")
    lines.append("")
    lines.append(f"  \"{problem.query_ciphertext}\" → \"{partial_str}\"")
    lines.append("")

    # Show which words are incomplete
    partial_words = partial_str.split()
    plain_words = problem.query_plaintext.split()
    incomplete = [(pw, rw) for pw, rw in zip(plain_words, partial_words)
                  if "?" in rw]
    if incomplete:
        word_list = ", ".join(f"\"{rw}\" (should be \"{pw}\")"
                              for pw, rw in incomplete[:3])
        lines.append(f"  Incomplete words: {word_list}")
        lines.append("")

    lines.append(f"Hmm, there are still {unknown_count} unknown letters "
                 f"(shown as '?'). The mappings from just "
                 f"{n_partial} example{'s' if n_partial > 1 else ''} aren't "
                 f"enough — I need to extract mappings from all "
                 f"{len(problem.examples)} examples to fill in the gaps.")
    lines.append("")

    return lines


# ── Caesar CoT ───────────────────────────────────────────────────────────────

def _generate_cot_caesar(problem: CipherProblem, rng: random.Random) -> str:
    """Generate CoT for Caesar cipher problems.

    Dynamic structure with optional detours:
    1. Observation
    2. Extract letter mappings
    3. Wrong hypothesis: Atbash → fails
    [Optional] N. Wrong shift detour (e.g. ROT13) → fails
    N+1. Compute shifts → discover constant shift
    N+2. Verify on second example
    N+3. Code verification
    N+4. Decrypt query
    """
    lines = []
    ex = problem.examples
    shift = problem.cipher_params["shift"]
    step = 1  # dynamic step counter

    lines.append(rng.choice(_OPENINGS))
    lines.append("")

    # ── Observation ──────────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Initial Observation**")
    step += 1
    lines.append(f"We have {len(ex)} examples of encrypted text with their "
                 "decryptions. I'll align each pair letter by letter to "
                 "find the pattern.")
    lines.append("")

    # ── Extract mappings from first two examples ─────────────────────────────
    lines.append(f"**Step {step}: Extract Letter Mappings**")
    step += 1
    enc0_words = ex[0][0].split()
    plain0_words = ex[0][1].split()
    lines.append(f"Example 1: \"{ex[0][0]}\" → \"{ex[0][1]}\"")

    for ew, pw in zip(enc0_words, plain0_words):
        pairs_str = ", ".join(f"{e}→{p}" for e, p in zip(ew, pw))
        lines.append(f"  \"{ew}\" → \"{pw}\": {pairs_str}")
    lines.append("")

    if len(ex) > 1:
        enc1_words = ex[1][0].split()
        plain1_words = ex[1][1].split()
        lines.append(f"Example 2: \"{ex[1][0]}\" → \"{ex[1][1]}\"")
        for ew, pw in zip(enc1_words, plain1_words):
            pairs_str = ", ".join(f"{e}→{p}" for e, p in zip(ew, pw))
            lines.append(f"  \"{ew}\" → \"{pw}\": {pairs_str}")
        lines.append("")

    # ── Wrong hypothesis — Atbash (always) ───────────────────────────────────
    lines.append(f"**Step {step}: Test hypothesis — Atbash cipher**")
    step += 1
    lines.append("In an Atbash cipher, each letter maps to its reverse: "
                 "a↔z, b↔y, c↔x, etc.")

    first_pairs = _extract_letter_pairs(ex[0][0], ex[0][1])
    for e, p in first_pairs[:4]:
        atbash_result = chr(ord('z') - (ord(e) - ord('a')))
        match = atbash_result == p
        status = "✓" if match else f"✗ (expected '{p}')"
        lines.append(f"  Atbash('{e}') = '{atbash_result}' {status}")
        if not match:
            lines.append("This is NOT an Atbash cipher. Moving on.")
            break
    lines.append("")

    # ── [Optional] Wrong shift detour (~40%) ─────────────────────────────────
    if rng.random() < 0.4:
        lines.append(f"**Step {step}: Test hypothesis — common shift**")
        step += 1
        lines.extend(_generate_wrong_shift_detour(problem, rng))

    # ── Compute shifts → discover Caesar ─────────────────────────────────────
    lines.append(f"**Step {step}: Compute shift values**")
    step += 1
    lines.append("Let me compute the shift (encrypted_pos − plain_pos, "
                 "mod 26) for each letter pair:")
    lines.append("")

    shifts_shown = []
    for e, p in first_pairs[:6]:
        s = _compute_shift(e, p)
        shifts_shown.append(s)
        lines.append(f"  '{e}'({_letter_pos(e)}) → '{p}'({_letter_pos(p)}): "
                     f"shift = ({_letter_pos(e)} − {_letter_pos(p)}) "
                     f"mod 26 = {s}")

    lines.append("")
    lines.append(rng.choice(_CAESAR_DISCOVERIES).format(shift=shift))
    lines.append(f"To decrypt: plain = (encrypted_pos − {shift}) mod 26.")
    lines.append("")

    # ── Verify on second example ─────────────────────────────────────────────
    lines.append(f"**Step {step}: Verify on another example**")
    step += 1
    verify_idx = min(1, len(ex) - 1)
    lines.append(f"Example {verify_idx + 1}: \"{ex[verify_idx][0]}\" → "
                 f"\"{ex[verify_idx][1]}\"")
    verify_pairs = _extract_letter_pairs(
        ex[verify_idx][0], ex[verify_idx][1])
    for e, p in verify_pairs[:6]:
        decrypted = chr((ord(e) - ord('a') - shift) % 26 + ord('a'))
        lines.append(f"  '{e}' − {shift} = '{decrypted}' "
                     f"(expected '{p}') {'✓' if decrypted == p else '✗'}")

    lines.append("")
    lines.append("All letters verify correctly!")
    lines.append("")

    # ── Code verification ────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Code Verification**")
    step += 1
    lines.extend(_generate_code_verification_caesar(problem))
    lines.append("")

    # ── Decrypt query ────────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Decrypt the query**")
    lines.append(f"Query: \"{problem.query_ciphertext}\"")
    lines.append("")
    lines.extend(_decrypt_word_by_word(problem))
    lines.append("")
    lines.append(f"Decrypted text: {problem.query_plaintext}")

    return "\n".join(lines)


# ── Substitution CoT ─────────────────────────────────────────────────────────

def _generate_cot_substitution(problem: CipherProblem,
                                rng: random.Random) -> str:
    """Generate CoT for random substitution cipher problems.

    Dynamic structure with optional detours:
    1. Observation
    2. Extract letter mappings from each example
    [Optional] N. Premature decrypt attempt → partial '?' result
    N+1. Caesar hypothesis → different shifts → fails
    N+2. Build complete substitution table
    N+3. Verify consistency
    N+4. Code verification
    N+5. Full verification on all examples
    N+6. Decrypt query word by word
    """
    lines = []
    ex = problem.examples
    step = 1  # dynamic step counter

    lines.append(rng.choice(_OPENINGS))
    lines.append("")

    # ── Observation ──────────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Initial Observation**")
    step += 1
    lines.append(f"We have {len(ex)} examples of encrypted text with their "
                 "decryptions. I'll align each pair letter by letter to "
                 "build a substitution mapping.")
    lines.append("")

    # ── Extract mappings ─────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Extract Letter Mappings**")
    step += 1

    all_mapping = {}  # enc → plain
    detail_limit = min(3, len(ex))  # show first 3 in detail
    for i, (enc, plain) in enumerate(ex):
        if i < detail_limit:
            lines.append(f"Example {i + 1}: \"{enc}\" → \"{plain}\"")
            enc_words = enc.split()
            plain_words = plain.split()
            for ew, pw in zip(enc_words, plain_words):
                pairs_str = ", ".join(f"{e}→{p}" for e, p in zip(ew, pw))
                lines.append(f"  \"{ew}\" → \"{pw}\": {pairs_str}")
            lines.append("")

        # Always accumulate mapping
        enc_words = enc.split()
        plain_words = plain.split()
        for ew, pw in zip(enc_words, plain_words):
            for e, p in zip(ew, pw):
                all_mapping[e] = p

    if len(ex) > detail_limit:
        remaining = len(ex) - detail_limit
        lines.append(f"(Extracting mappings from remaining {remaining} "
                     f"example{'s' if remaining > 1 else ''} as well...)")
        lines.append("")

    # ── [Optional] Premature decrypt detour (~50%) ───────────────────────────
    detour_lines = []
    if rng.random() < 0.5:
        detour_lines = _generate_premature_decrypt_detour(problem, rng)
    if detour_lines:
        lines.append(f"**Step {step}: Early decryption attempt**")
        step += 1
        lines.extend(detour_lines)

    # ── Wrong hypothesis — Caesar ────────────────────────────────────────────
    lines.append(f"**Step {step}: Test hypothesis — Caesar cipher "
                 "(constant shift)**")
    step += 1
    lines.append("Let me check if all letters are shifted by the same amount:")
    lines.append("")

    first_pairs = _extract_letter_pairs(ex[0][0], ex[0][1])
    shifts = []
    for e, p in first_pairs[:5]:
        s = _compute_shift(e, p)
        shifts.append(s)
        lines.append(f"  '{e}'({_letter_pos(e)}) → '{p}'({_letter_pos(p)}): "
                     f"shift = {s}")

    lines.append("")
    unique_shifts = sorted(set(shifts))
    if len(unique_shifts) > 1:
        shift_strs = ", ".join(str(s) for s in shifts)
        lines.append(rng.choice(_SUB_CONCLUSIONS).format(shifts=shift_strs))
    lines.append("")

    # ── Build mapping table ──────────────────────────────────────────────────
    lines.append(f"**Step {step}: Build Complete Substitution Table**")
    step += 1
    lines.append("From all examples, I've extracted these mappings "
                 "(encrypted → decrypted):")
    lines.append("")

    sorted_pairs = sorted(all_mapping.items())
    for i in range(0, len(sorted_pairs), 8):
        chunk = sorted_pairs[i:i + 8]
        line = "  " + "  ".join(f"{e}→{p}" for e, p in chunk)
        lines.append(line)

    lines.append("")
    lines.append(f"Total: {len(all_mapping)} letter mappings extracted.")
    lines.append("")

    # ── Consistency verification ─────────────────────────────────────────────
    lines.append(f"**Step {step}: Verify Consistency**")
    step += 1
    lines.append("Let me verify that the same encrypted letter always "
                 "decrypts to the same letter:")
    lines.append("")

    multi = _find_multi_occurrence_letters(ex)
    rng.shuffle(multi)
    for e, p, indices in multi[:3]:
        example_nums = ", ".join(str(idx + 1) for idx in indices)
        lines.append(f"  '{e}' appears in examples {example_nums} → "
                     f"always maps to '{p}' ✓")

    lines.append("")
    lines.append("The mapping is consistent across all examples.")
    lines.append("")

    # ── Code verification ────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Code Verification**")
    step += 1
    lines.extend(_generate_code_verification_substitution(problem, all_mapping))
    lines.append("")

    # ── Full verification on all examples ────────────────────────────────────
    lines.append(f"**Step {step}: Full Verification**")
    step += 1
    for i, (enc, plain) in enumerate(ex):
        decrypted = "".join(all_mapping.get(c, c) for c in enc)
        match = decrypted == plain
        lines.append(f"  Example {i + 1}: \"{enc}\" → \"{decrypted}\" "
                     f"{'✓' if match else '✗'}")
    lines.append("")
    lines.append("All examples verified successfully!")
    lines.append("")

    # ── Decrypt query ────────────────────────────────────────────────────────
    lines.append(f"**Step {step}: Decrypt the Query**")
    lines.append(f"Query: \"{problem.query_ciphertext}\"")
    lines.append("")
    lines.extend(_decrypt_word_by_word(problem))
    lines.append("")
    lines.append(f"Decrypted text: {problem.query_plaintext}")

    return "\n".join(lines)


# ── Shared CoT Helpers ───────────────────────────────────────────────────────

def _generate_code_verification_caesar(problem: CipherProblem) -> list[str]:
    """Generate Python code verification block for Caesar cipher."""
    shift = problem.cipher_params["shift"]
    ex = problem.examples
    lines = []

    lines.append("To be absolutely certain, let me express this as code:")
    lines.append("")
    lines.append("```python")
    lines.append(f"def decrypt(text, shift={shift}):")
    lines.append("    return \"\".join(")
    lines.append("        chr((ord(c) - ord('a') - shift) % 26 + ord('a'))")
    lines.append("        if c.isalpha() else c for c in text")
    lines.append("    )")
    lines.append("```")
    lines.append("")

    lines.append("Quick verification:")
    for i in range(min(2, len(ex))):
        enc, plain = ex[i]
        lines.append(f"  decrypt(\"{enc}\") = \"{plain}\" ✓")

    return lines


def _generate_code_verification_substitution(
    problem: CipherProblem, mapping: dict[str, str]
) -> list[str]:
    """Generate Python code verification block for substitution cipher."""
    ex = problem.examples
    lines = []

    lines.append("To be absolutely certain, let me express this as code:")
    lines.append("")
    lines.append("```python")
    lines.append("mapping = {")

    sorted_pairs = sorted(mapping.items())
    for i in range(0, len(sorted_pairs), 8):
        chunk = sorted_pairs[i:i + 8]
        items = ", ".join(f"'{e}': '{p}'" for e, p in chunk)
        trailing = "," if i + 8 < len(sorted_pairs) else ""
        lines.append(f"    {items}{trailing}")

    lines.append("}")
    lines.append("def decrypt(text):")
    lines.append("    return \"\".join(mapping.get(c, c) for c in text)")
    lines.append("```")
    lines.append("")

    lines.append("Quick verification:")
    for i in range(min(2, len(ex))):
        enc, plain = ex[i]
        lines.append(f"  decrypt(\"{enc}\") = \"{plain}\" ✓")

    return lines


def _decrypt_word_by_word(problem: CipherProblem) -> list[str]:
    """Show step-by-step word-by-word decryption of the query."""
    lines = []
    rev_mapping = problem.reverse_mapping
    enc_words = problem.query_ciphertext.split()
    plain_words = problem.query_plaintext.split()

    for i, (ew, pw) in enumerate(zip(enc_words, plain_words)):
        pairs_str = ", ".join(f"{e}→{p}" for e, p in zip(ew, pw))
        lines.append(f"  Word {i + 1}: \"{ew}\"")
        lines.append(f"    {pairs_str} → \"{pw}\"")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = (
    "In Alice's Wonderland, secret encryption rules are used on text. "
    "Here are some examples:\n"
    "{examples}\n"
    "Now, decrypt the following text: {query}"
)


def format_prompt(problem: CipherProblem) -> str:
    """Format the problem into the competition-style prompt."""
    example_lines = []
    for enc, plain in problem.examples:
        example_lines.append(f"{enc} -> {plain}")
    return PROMPT_TEMPLATE.format(
        examples="\n".join(example_lines),
        query=problem.query_ciphertext,
    )


def format_answer_with_cot(cot_text: str, answer: str) -> str:
    r"""Format the full answer: CoT reasoning followed by \boxed{answer}."""
    return f"{cot_text}\n\n\\boxed{{{answer}}}"


def format_answer_direct(answer: str) -> str:
    r"""Format direct answer (no CoT): just \boxed{answer}."""
    return f"\\boxed{{{answer}}}"


def format_answer_think(cot_text: str, answer: str, depth: str) -> str:
    r"""Format answer for <think> training mode.

    The chat template prefix provides '<think>\n', so the answer column
    starts immediately after that.

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

# Depth distribution per curriculum level:
# Easy problems → more direct/light (model should learn to be efficient)
# Hard problems → more full CoT (model should learn to reason deeply)
_DEPTH_RATIOS = {
    #          direct  light   full
    1:        (0.40,   0.35,   0.25),   # Caesar: often straightforward
    2:        (0.25,   0.30,   0.45),   # Substitution, many examples
    3:        (0.15,   0.25,   0.60),   # Substitution, few examples: need more reasoning
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

def generate_dataset(
    num_samples: int,
    levels: list[int],
    seed: int = 42,
    num_examples_range: tuple[int, int] = (3, 6),
) -> list[dict]:
    """Generate a CoT dataset for cipher tasks with layered depth.

    Each sample is assigned a cot_depth (direct / light / full) based on
    its difficulty level. This teaches the model to adaptively calibrate
    its reasoning effort.

    Returns list of dicts with keys: id, prompt, answer, cot_answer,
    think_answer, cot_depth, level, cipher_type, cipher_description.
    """
    rng = random.Random(seed)

    # Distribute samples across levels
    samples_per_level = num_samples // len(levels)
    remainder = num_samples % len(levels)

    dataset = []
    for level_idx, level in enumerate(levels):
        count = samples_per_level + (1 if level_idx < remainder else 0)

        for _ in range(count):
            problem = generate_problem(rng, level, num_examples_range)

            prompt = format_prompt(problem)
            answer = problem.query_plaintext
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
                "cipher_type": problem.cipher_type,
                "cipher_description": problem.cipher_description,
            })

    rng.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], output_dir: Path):
    """Save dataset in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full JSONL (all fields)
    jsonl_path = output_dir / "cipher_cot.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[saved] {jsonl_path} ({len(dataset)} samples)")

    # 2. Competition-format CSV — direct answer only
    csv_direct_path = output_dir / "cipher_direct.csv"
    with open(csv_direct_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"],
                             sample["answer"]])
    print(f"[saved] {csv_direct_path}")

    # 3. CoT-format CSV — answer includes CoT reasoning
    csv_cot_path = output_dir / "cipher_cot.csv"
    with open(csv_cot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"],
                             sample["cot_answer"]])
    print(f"[saved] {csv_cot_path}")

    # 4. <think> format CSV — for train.py --enable_thinking mode
    csv_think_path = output_dir / "cipher_think.csv"
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

    # 6. Stats (including depth distribution)
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
        description="Generate CoT training data for cipher/decryption tasks.")
    parser.add_argument("--output_dir", type=str, default="data/cot/cipher",
                        help="Output directory for generated data.")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Total number of samples to generate.")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
                        help="Curriculum levels to include "
                             "(1=Caesar, 2=substitution+many, "
                             "3=substitution+few).")
    parser.add_argument("--min_examples", type=int, default=3,
                        help="Minimum number of demo examples per problem.")
    parser.add_argument("--max_examples", type=int, default=6,
                        help="Maximum number of demo examples per problem.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Generating {args.num_samples} cipher CoT samples "
          f"(levels={args.levels}, seed={args.seed})...")
    dataset = generate_dataset(
        num_samples=args.num_samples,
        levels=args.levels,
        seed=args.seed,
        num_examples_range=(args.min_examples, args.max_examples),
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
                  f"cipher={s['cipher_description']}")
            print("=" * 70)
            print(f"Prompt: {s['prompt'][:120]}...")
            think = s["think_answer"]
            print(f"\n--- Think Answer ---\n{think[:800]}"
                  f"{'...' if len(think) > 800 else ''}")


if __name__ == "__main__":
    main()
