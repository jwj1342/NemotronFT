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


# ── Caesar CoT ───────────────────────────────────────────────────────────────

def _generate_cot_caesar(problem: CipherProblem, rng: random.Random) -> str:
    """Generate CoT for Caesar cipher problems.

    Structure:
    1. Observation
    2. Extract letter mappings from first example
    3. Wrong hypothesis: Atbash cipher → fails
    4. Compute shifts → discover constant shift
    5. Verify on second example
    6. Code verification
    7. Decrypt query
    """
    lines = []
    ex = problem.examples
    shift = problem.cipher_params["shift"]

    lines.append("Let me analyze the encrypted-decrypted pairs to find "
                 "the encryption rule.")
    lines.append("")

    # ── Phase 1: Observation ─────────────────────────────────────────────────
    lines.append("**Step 1: Initial Observation**")
    lines.append(f"We have {len(ex)} examples of encrypted text with their "
                 "decryptions. I'll align each pair letter by letter to "
                 "find the pattern.")
    lines.append("")

    # ── Phase 2: Extract mappings from first example ─────────────────────────
    lines.append("**Step 2: Extract Letter Mappings**")
    enc0_words = ex[0][0].split()
    plain0_words = ex[0][1].split()
    lines.append(f"Example 1: \"{ex[0][0]}\" → \"{ex[0][1]}\"")

    for ew, pw in zip(enc0_words, plain0_words):
        pairs_str = ", ".join(f"{e}→{p}" for e, p in zip(ew, pw))
        lines.append(f"  \"{ew}\" → \"{pw}\": {pairs_str}")
    lines.append("")

    # Show second example too
    if len(ex) > 1:
        enc1_words = ex[1][0].split()
        plain1_words = ex[1][1].split()
        lines.append(f"Example 2: \"{ex[1][0]}\" → \"{ex[1][1]}\"")
        for ew, pw in zip(enc1_words, plain1_words):
            pairs_str = ", ".join(f"{e}→{p}" for e, p in zip(ew, pw))
            lines.append(f"  \"{ew}\" → \"{pw}\": {pairs_str}")
        lines.append("")

    # ── Phase 3: Wrong hypothesis — Atbash ───────────────────────────────────
    lines.append("**Step 3: Test hypothesis — Atbash cipher**")
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

    # ── Phase 4: Compute shifts → discover Caesar ────────────────────────────
    lines.append("**Step 4: Compute shift values**")
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
    lines.append(f"All shifts are **{shift}**! This is a Caesar cipher "
                 f"with shift {shift}.")
    lines.append(f"To decrypt: plain = (encrypted_pos − {shift}) mod 26.")
    lines.append("")

    # ── Phase 5: Verify on second example ────────────────────────────────────
    lines.append("**Step 5: Verify on another example**")
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

    # ── Phase 6: Code verification ───────────────────────────────────────────
    lines.append("**Step 6: Code Verification**")
    lines.extend(_generate_code_verification_caesar(problem))
    lines.append("")

    # ── Phase 7: Decrypt query ───────────────────────────────────────────────
    lines.append("**Step 7: Decrypt the query**")
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

    Structure:
    1. Observation
    2. Extract letter mappings from each example
    3. Wrong hypothesis: Caesar cipher → shifts not constant → fails
    4. Build complete substitution table
    5. Verify consistency across examples
    6. Code verification
    7. Decrypt query word by word
    """
    lines = []
    ex = problem.examples

    lines.append("Let me analyze the encrypted-decrypted pairs to find "
                 "the encryption rule.")
    lines.append("")

    # ── Phase 1: Observation ─────────────────────────────────────────────────
    lines.append("**Step 1: Initial Observation**")
    lines.append(f"We have {len(ex)} examples of encrypted text with their "
                 "decryptions. I'll align each pair letter by letter to "
                 "build a substitution mapping.")
    lines.append("")

    # ── Phase 2: Extract mappings ────────────────────────────────────────────
    lines.append("**Step 2: Extract Letter Mappings**")

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

    # ── Phase 3: Wrong hypothesis — Caesar ───────────────────────────────────
    lines.append("**Step 3: Test hypothesis — Caesar cipher (constant shift)**")
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
        lines.append(f"The shifts are different ({shift_strs}), so this is "
                     "NOT a Caesar cipher.")
        lines.append("It must be a general monoalphabetic substitution cipher "
                     "with a unique mapping for each letter.")
    lines.append("")

    # ── Phase 4: Build mapping table ─────────────────────────────────────────
    lines.append("**Step 4: Build Complete Substitution Table**")
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

    # ── Phase 5: Consistency verification ────────────────────────────────────
    lines.append("**Step 5: Verify Consistency**")
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

    # ── Phase 6: Code verification ───────────────────────────────────────────
    lines.append("**Step 6: Code Verification**")
    lines.extend(_generate_code_verification_substitution(problem, all_mapping))
    lines.append("")

    # ── Phase 7: Full verification on all examples ───────────────────────────
    lines.append("**Step 7: Full Verification**")
    for i, (enc, plain) in enumerate(ex):
        decrypted = "".join(all_mapping.get(c, c) for c in enc)
        match = decrypted == plain
        lines.append(f"  Example {i + 1}: \"{enc}\" → \"{decrypted}\" "
                     f"{'✓' if match else '✗'}")
    lines.append("")
    lines.append("All examples verified successfully!")
    lines.append("")

    # ── Phase 8: Decrypt query ───────────────────────────────────────────────
    lines.append("**Step 8: Decrypt the Query**")
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Generate Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    num_samples: int,
    levels: list[int],
    seed: int = 42,
    num_examples_range: tuple[int, int] = (3, 6),
) -> list[dict]:
    """Generate a full CoT dataset for cipher tasks.

    Returns list of dicts with keys: id, prompt, answer, cot_answer,
    level, cipher_type, cipher_description.
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
            cot = generate_cot(problem, rng)

            sample_id = uuid.uuid4().hex[:8]
            dataset.append({
                "id": sample_id,
                "prompt": prompt,
                "answer": answer,
                "cot_answer": format_answer_with_cot(cot, answer),
                "level": level,
                "cipher_type": problem.cipher_type,
                "cipher_description": problem.cipher_description,
            })

    rng.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], output_dir: Path):
    """Save dataset in multiple formats (JSONL, CSV, per-level splits)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full JSONL (all fields)
    jsonl_path = output_dir / "cipher_cot.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[saved] {jsonl_path} ({len(dataset)} samples)")

    # 2. Competition-format CSV (id, prompt, answer) — direct answer only
    csv_direct_path = output_dir / "cipher_direct.csv"
    with open(csv_direct_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"],
                             sample["answer"]])
    print(f"[saved] {csv_direct_path}")

    # 3. CoT-format CSV (id, prompt, answer) — answer includes CoT
    csv_cot_path = output_dir / "cipher_cot.csv"
    with open(csv_cot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        for sample in dataset:
            writer.writerow([sample["id"], sample["prompt"],
                             sample["cot_answer"]])
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
        "by_level": {
            str(level): len(samples)
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

    # Print a sample preview
    print("\n" + "=" * 70)
    print("SAMPLE PREVIEW (first entry)")
    print("=" * 70)
    sample = dataset[0]
    print(f"Level: {sample['level']}")
    print(f"Cipher: {sample['cipher_description']}")
    print(f"\n--- Prompt ---\n{sample['prompt']}")
    print(f"\n--- CoT Answer ---\n{sample['cot_answer'][:1500]}...")


if __name__ == "__main__":
    main()
