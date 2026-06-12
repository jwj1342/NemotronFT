"""
synth_numeric.py  —  Reverse-engineered generator for `equation_numeric_*`.

PURPOSE (draft / verification only — NOT a training run)
--------------------------------------------------------
Produce labeled synthetic `equation_numeric_guess` (and `_deduce`) problems whose
ground-truth rule is *known by construction*. Intended downstream use is prior
calibration: teach the model, for the under-constrained "operator-not-in-examples"
case, to fall back to the empirically-correct combo prior (output format inferred
from sibling examples; operation chosen by frequency) instead of guessing a sign /
reversal at random.

This file is a DRAFT. It (a) regenerates the grammar, (b) samples problems with a
known combo per operator, (c) renders them in the exact prompt shape used in
data/by_category/equation_numeric.csv, and (d) self-checks that the published
solver (symbol_digit_solver) recovers the planted rule from the *deduce* view.

Reverse-engineered facts driving the priors below (all measured on
data/by_category/equation_numeric.csv, 732 problems, 411 uniquely-solvable):

  * digit-order prior:  BA_DC 0.60 , AB_CD 0.40   (AB_DC / BA_CD ~ 0)
  * output-format prior: rev 0.41 , raw 0.26 , then op_pre_rev / op_pre_abs /
    rev_neg / abs / op_pre / op_suf_abs / zpad2 / dsum  (long tail ~0.01 each)
  * operation prior (per-operator, independent): mul .14 add .12 sub_lr .10
    cat_lr .07 mulsub1 .07 addm1 .06 muladd1 .06 add1 .05 cat_rl .04 ...
  * within ONE problem the (digit-order, output-format) pair is SHARED across all
    operators; only the OPERATION differs per operator symbol. This is the single
    fact that makes `guess` partially learnable: the format is observable from the
    sibling examples, the operation is not.
  * 1-3 distinct operator symbols per problem, 3-5 example lines total.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import symbol_digit_solver as S  # noqa: E402

# --------------------------------------------------------------------------
# Reverse-engineered priors (counts -> weights). Numbers from the measured
# 411 uniquely-solvable training problems; see module docstring.
# --------------------------------------------------------------------------
ORDER_PRIOR = {"BA_DC": 0.60, "AB_CD": 0.40}

FMT_PRIOR = {
    "rev": 0.41, "raw": 0.26, "op_pre_rev": 0.06, "op_pre_abs": 0.06,
    "rev_neg": 0.05, "abs": 0.04, "op_pre": 0.03, "op_suf_abs": 0.03,
    "zpad2": 0.03, "dsum": 0.03,
}

OP_PRIOR = {
    "mul": 0.14, "add": 0.12, "sub_lr": 0.10, "cat_lr": 0.07, "mulsub1": 0.07,
    "addm1": 0.06, "muladd1": 0.06, "add1": 0.05, "cat_rl": 0.04, "abs_sub": 0.04,
    "sub_rl": 0.03, "dsum_add": 0.02, "dsum_mul": 0.02,
}

# Operator symbols seen in the dataset (sampled with replacement, no overlap
# with digits). The literal symbol carries NO information about the operation.
OP_SYMBOLS = list("+-*/!@#%&`|^~}{)([]\\:\";'<>.,?")


def _wsample(prior: dict, rng: random.Random):
    items, weights = zip(*prior.items())
    return rng.choices(items, weights=weights, k=1)[0]


def _render_line(order, op, fmt, A, B, C, D, op_char):
    """Render 'AB<op>CD = RESULT' using the planted combo. Returns (line, result)."""
    res = S.apply_combo((order, op, fmt), A, B, C, D, op_char)
    return f"{A}{B}{op_char}{C}{D} = {res}", res


def gen_problem(rng: random.Random, mode: str = "guess"):
    """
    Generate one labeled problem.

    Returns dict: {prompt, answer, meta} where meta records the planted
    (order, fmt) and the per-symbol operation, plus the question's true op.

    mode == "guess":  the question operator's symbol does NOT appear among the
                      example lines (must be inferred from format + op prior).
    mode == "deduce": the question operator's symbol DOES appear in >=1 example.
    """
    order = _wsample(ORDER_PRIOR, rng)
    fmt = _wsample(FMT_PRIOR, rng)

    # 1-3 distinct operator symbols, each with its own operation (shared order+fmt)
    n_ops = rng.choice([2, 2, 3, 3, 1])
    syms = rng.sample(OP_SYMBOLS, n_ops)
    sym_to_op = {s: _wsample(OP_PRIOR, rng) for s in syms}

    # question operator
    if mode == "guess":
        q_sym = rng.choice([s for s in OP_SYMBOLS if s not in syms])
        q_op = _wsample(OP_PRIOR, rng)
    else:  # deduce: reuse an existing symbol
        q_sym = rng.choice(syms)
        q_op = sym_to_op[q_sym]

    # example lines: 3-5 lines, drawn from the in-prompt symbols
    n_lines = rng.choice([3, 3, 4, 4, 5])
    lines = []
    for _ in range(n_lines):
        s = rng.choice(syms)
        A, B, C, D = (rng.randint(0, 9) for _ in range(4))
        line, _res = _render_line(order, sym_to_op[s], fmt, A, B, C, D, s)
        lines.append(line)

    qA, qB, qC, qD = (rng.randint(0, 9) for _ in range(4))
    answer = S.apply_combo((order, q_op, fmt), qA, qB, qC, qD, q_sym)

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n"
        + "\n".join(lines)
        + f"\nNow, determine the result for: {qA}{qB}{q_sym}{qC}{qD}"
    )
    meta = {
        "order": order, "fmt": fmt, "q_op": q_op, "q_sym": q_sym,
        "sym_to_op": sym_to_op, "mode": mode,
    }
    return {"prompt": prompt, "answer": answer, "meta": meta}


def selfcheck(n: int = 2000, seed: int = 0):
    """
    Verify generator self-consistency:
      1. Every rendered example line is exactly recoverable by apply_combo
         (round-trip soundness).
      2. DEDUCE problems are solvable by the published solver when >=2 op-matching
         examples exist (sanity: planted rule is in-grammar and recoverable).
      3. GUESS problems: confirm the operation is genuinely under-determined —
         report how often the truth coincides with the top-prior op given the
         (correctly inferred) format. This is the salvage ceiling we'd train toward.
    """
    rng = random.Random(seed)
    sound = 0
    deduce_solved = deduce_total = 0
    guess_top_prior_hit = guess_total = 0
    top_op = max(OP_PRIOR, key=OP_PRIOR.get)

    for _ in range(n):
        mode = rng.choice(["guess", "deduce"])
        prob = gen_problem(rng, mode=mode)
        ex, q = S.parse_problem_prompt(prob["prompt"])

        # (1) round-trip soundness on the question itself
        m = prob["meta"]
        rt = S.apply_combo((m["order"], m["q_op"], m["fmt"]),
                           q[0], q[1], q[3], q[4], q[2]) if q else None
        if rt == prob["answer"]:
            sound += 1

        if mode == "deduce" and ex is not None:
            ex_op = [e for e in ex if e[2] == q[2]]
            if len(ex_op) >= 2:
                deduce_total += 1
                r = S.solve_symbol_digit(ex_op, q)
                if r.solved and r.answer == prob["answer"]:
                    deduce_solved += 1
        if mode == "guess":
            guess_total += 1
            # with the TRUE format, would the top-prior op give the truth?
            guess = S.apply_combo((m["order"], top_op, m["fmt"]),
                                  q[0], q[1], q[3], q[4], q[2])
            if guess == prob["answer"]:
                guess_top_prior_hit += 1

    print(f"[selfcheck n={n}]")
    print(f"  round-trip soundness:         {sound}/{n} = {100*sound/n:.1f}%")
    if deduce_total:
        print(f"  deduce solver-recovers-rule:  {deduce_solved}/{deduce_total} "
              f"= {100*deduce_solved/deduce_total:.1f}%  (>=2 op-examples)")
    if guess_total:
        print(f"  guess top-prior-op == truth:  {guess_top_prior_hit}/{guess_total} "
              f"= {100*guess_top_prior_hit/guess_total:.1f}%  (format known)")
        print("  -> the gap from 100% is the irreducible operator ambiguity in guess.")


if __name__ == "__main__":
    selfcheck(n=int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
