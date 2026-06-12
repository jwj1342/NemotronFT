"""
Synthetic cryptarithm generator (reverse-engineered from data/by_category/equation_symbolic.csv
+ the eval failure/correct split).

REVERSE-ENGINEERED GENERATOR (validated against model-correct & model-failed eval):

  Structure per puzzle:
    - a 25-symbol punctuation alphabet:  ! " # $ % & ' ( ) + - / : < > ? @ [ \\ ] ^ ` { | }
    - pick a per-puzzle RANDOM bijection cipher: 10 of those symbols -> digits 0..9
      (NOT ascii-sortable; measured order-agreement ~0.43 == random)
    - pick one (or, for "guess", several) operator symbol(s). The operator is EITHER
      a plaintext arithmetic char (+,-,*  -> ~55% of puzzles) OR another cipher symbol.
    - pick a RULE = (digit_order, operation, output_format):
        digit_order in {AB_CD, BA_DC, AB_DC, BA_CD}
        operation   in {add, mul, sub_lr, abs_sub, cat_lr, cat_rl, add1, addm1,
                        mulsub1, muladd1, ...}   (concat OR arithmetic)
        output_format in {raw, rev, abs, zpad2, ...}
      MEASURED rule frequency (this is the prior the MODEL exploits):
        * model-CORRECT puzzles: ~100% CONCATENATION (cat_lr 73%, cat_rl/BA_DC variants 27%)
        * model-FAILED  puzzles: ~100% ARITHMETIC
            BA_DC|add|rev 26%, BA_DC|mul|rev 21%, AB_CD|mul|raw 14%,
            AB_CD|add|raw 10%, AB_CD|sub_lr|raw 8%, + long tail
    - each puzzle = a few example lines  "AB<op>CD = R"  (3-5 examples) + 1 target.
      For "deduce": >=1 example shares the target operator.
      For "guess" : the target operator is NOVEL (never shown) -> rule must be guessed.

  KEY GENERATOR QUIRK that makes failures hard:
    The example set frequently does NOT expose all 10 cipher symbols, and the TARGET
    answer routinely needs a symbol the examples never showed (75% of failures).
    For concatenation the output reveals input symbols directly (so coverage is high);
    for arithmetic the output is a recomputed number, so coverage is sparse.

This module GENERATES synthetic puzzles WITH the ground-truth answer (we are the
oracle), so synthetic training data is automatically label-correct.

Two generation modes:
  - "easy_match_model"  : concatenation-heavy, full symbol coverage (mimics what the
                          model already does; low training value).
  - "hard_solvable"     : ARITHMETIC rules BUT constrained so the target's symbols are
                          ALL exposed by the examples (i.e. info-COMPLETE). This is the
                          salvageable slice: teaches the model to do the crack->op->encode
                          chain on cases that ARE determined. We deliberately reject
                          puzzles whose target needs an unseen symbol.

Pure stdlib.
"""
from __future__ import annotations
import random, string

ALPHABET = list("!\"#$%&'()+-/:<>?@[\\]^`{|}")  # 25 symbols (matches observed)
PLAINTEXT_OPS = ["+", "-", "*"]

# rule frequency for the HARD (arithmetic) slice -- mirrors measured failed-set dist
HARD_RULES = [
    (("BA_DC", "add", "rev"), 26),
    (("BA_DC", "mul", "rev"), 21),
    (("AB_CD", "mul", "raw"), 14),
    (("AB_CD", "add", "raw"), 10),
    (("AB_CD", "sub_lr", "raw"), 8),
    (("AB_CD", "mulsub1", "raw"), 3),
    (("BA_DC", "mulsub1", "rev"), 3),
    (("BA_DC", "sub_lr", "rev"), 3),
    (("BA_DC", "addm1", "rev"), 3),
    (("BA_DC", "abs_sub", "rev"), 3),
    (("AB_CD", "muladd1", "raw"), 3),
]
EASY_RULES = [
    (("AB_CD", "cat_lr", "raw"), 73),
    (("BA_DC", "cat_lr", "rev"), 18),
    (("BA_DC", "cat_rl", "rev"), 6),
    (("AB_CD", "cat_rl", "raw"), 3),
]


def _wpick(rng, weighted):
    r = rng.random() * sum(w for _, w in weighted)
    acc = 0.0
    for item, w in weighted:
        acc += w
        if r <= acc:
            return item
    return weighted[-1][0]


def _lr(order, A, B, C, D):
    return {
        "AB_CD": (A*10+B, C*10+D), "BA_DC": (B*10+A, D*10+C),
        "AB_DC": (A*10+B, D*10+C), "BA_CD": (B*10+A, C*10+D),
    }[order]


def _op(name, L, R):
    return {
        "add": L+R, "mul": L*R, "sub_lr": L-R, "abs_sub": abs(L-R),
        "cat_lr": int(f"{L:02d}{R:02d}"), "cat_rl": int(f"{R:02d}{L:02d}"),
        "add1": L+R+1, "addm1": L+R-1, "mulsub1": L*R-1, "muladd1": L*R+1,
    }[name]


def _fmt(name, v):
    if name == "raw":
        return str(v)
    if name == "rev":
        return str(abs(v))[::-1]
    if name == "abs":
        return str(abs(v))
    if name == "zpad2":
        return f"{abs(v):02d}"
    return str(v)


def _apply(rule, A, B, C, D):
    order, opn, fmt = rule
    L, R = _lr(order, A, B, C, D)
    v = _op(opn, L, R)
    if v < 0 and fmt in ("raw",):
        return None  # generator avoids negatives in raw concat-like cases
    s = _fmt(fmt, v)
    return s if s.isdigit() else None


def make_puzzle(rng, mode="hard_solvable", n_examples=4, max_tries=200):
    """
    Return (prompt_str, answer_str, meta) or None if generation failed within tries.
    meta = {cipher, rule, op_sym, op_plaintext}
    """
    for _ in range(max_tries):
        # cipher: choose 10 distinct symbols -> digits 0..9
        syms = rng.sample(ALPHABET, 10)
        dig2sym = {d: syms[d] for d in range(10)}
        rule = _wpick(rng, HARD_RULES if mode == "hard_solvable" else EASY_RULES)
        op_plain = rng.random() < 0.55
        op_sym = rng.choice(PLAINTEXT_OPS) if op_plain else rng.choice(
            [s for s in ALPHABET if s not in syms] or ALPHABET)

        def enc(num_str):
            return "".join(dig2sym[int(c)] for c in num_str)

        def gen_pair():
            for _ in range(50):
                A, B, C, D = (rng.randint(0, 9) for _ in range(4))
                out = _apply(rule, A, B, C, D)
                if out is None:
                    continue
                inp = dig2sym[A] + dig2sym[B] + op_sym + dig2sym[C] + dig2sym[D]
                return (A, B, C, D), inp, enc(out)
            return None

        examples = []
        seen_syms = set()
        ok = True
        for _ in range(n_examples):
            g = gen_pair()
            if g is None:
                ok = False
                break
            (A, B, C, D), inp, out = g
            examples.append((inp, out))
            seen_syms |= {dig2sym[A], dig2sym[B], dig2sym[C], dig2sym[D]} | set(out)
        if not ok:
            continue
        # target
        g = gen_pair()
        if g is None:
            continue
        (A, B, C, D), tinp, tout = g
        # HARD_SOLVABLE constraint: target answer must use only EXPOSED symbols
        if mode == "hard_solvable":
            need = {dig2sym[A], dig2sym[B], dig2sym[C], dig2sym[D]} | set(tout)
            if not need <= seen_syms:
                continue  # reject info-incomplete puzzle -> keep only salvageable

        lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"]
        for inp, out in examples:
            lines.append(f"{inp} = {out}")
        lines.append(f"Now, determine the result for: {tinp}")
        prompt = "\n".join(lines)
        meta = {"cipher": dig2sym, "rule": rule, "op_sym": op_sym,
                "op_plaintext": op_plain, "target_digits": (A, B, C, D)}
        return prompt, tout, meta
    return None


if __name__ == "__main__":
    rng = random.Random(0)
    for mode in ("easy_match_model", "hard_solvable"):
        print("="*60)
        print("MODE:", mode)
        p = make_puzzle(rng, mode=mode)
        if p:
            prompt, ans, meta = p
            print(prompt)
            print("ANSWER:", ans)
            print("rule:", meta["rule"], "op:", repr(meta["op_sym"]), "plaintext:", meta["op_plaintext"])
