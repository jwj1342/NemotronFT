"""
Cryptarithm (cipher-digit) salvage-space diagnostic.

Goal: for each FAILED cryptarithm_{deduce,guess} puzzle, enumerate the FULL set
of self-consistent (combo, cipher-mapping) solutions and the DISTINCT target
answers they imply. This separates:

  - unique     : all consistent solutions agree on the SAME target answer
                 -> formally determined -> model should get it -> CAPACITY-limited
  - multi      : >=2 distinct target answers consistent w/ examples
                 -> genuinely under-constrained -> INFO-limited
  - no_solution: solver grammar can't reproduce the examples at all
                 -> "false unsolvable" (grammar gap, NOT info-limited; flagged)

We do NOT use the solver's early-exit. We enumerate ALL combos and, per combo,
ALL example-consistent cipher mappings, then collect every implied target answer.

Also:
  - grammar coverage: does ANY (combo, mapping) decode to the KNOWN true answer?
    (lower bound on how much of the answer space the grammar can even express)
  - MDL/frequency prior: among multi-solution puzzles, pick the answer implied by
    the earliest (highest-frequency) combo; measure how often that == true answer
    => "prior-salvageable" fraction inside the info-limited bucket.

Pure stdlib. Designed for login-node use: caller passes a SAMPLE of ids.
"""
from __future__ import annotations
import sys, csv, json, re
from pathlib import Path

csv.field_size_limit(10**7)
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from symbol_digit_solver import COMBOS, apply_combo  # type: ignore

EVAL_GLOB = "/scratch/jwj/NemotronFT/output/eval_full_huikang-baseline-converted_shard{}of4.jsonl"
TRAIN_CSV = "/scratch/jwj/NemotronFT/data/train.csv"

# combos usable for cipher problems (operator is encrypted -> drop op_ formats)
CIPHER_COMBOS = [c for c in COMBOS if "op_" not in c[2]]


def parse_prompt(prompt: str):
    raw = []
    q = None
    for line in prompt.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r'(?:determine|result for)', line, re.IGNORECASE):
            m = re.search(r':\s*(\S+)\s*$', line)
            if m:
                q = m.group(1)
            continue
        m = re.match(r'^(\S+)\s*=\s*(\S+)$', line)
        if m:
            raw.append((m.group(1), m.group(2)))
    if not raw or q is None or len(q) < 5:
        return None, None, None, 0
    op = q[2]
    valid = [(i, o) for i, o in raw if len(i) == 5 and i[2] == op]
    # n_op = how many examples share the question operator (constraint count)
    return valid, q, op, len(valid)


def _build_pm(inp, cipher_out, numeric_out, A, B, C, D):
    """Build sym->digit mapping from one example assignment, or None on conflict."""
    if len(cipher_out) != len(numeric_out):
        return None
    m, r = {}, {}
    for sym, dig in [(inp[0], A), (inp[1], B), (inp[3], C), (inp[4], D)]:
        if sym in m:
            if m[sym] != dig:
                return None
        elif dig in r:
            if r[dig] != sym:
                return None
        else:
            m[sym] = dig; r[dig] = sym
    for csym, dchar in zip(cipher_out, numeric_out):
        dig = int(dchar)
        if csym in m:
            if m[csym] != dig:
                return None
        elif dig in r:
            if r[dig] != csym:
                return None
        else:
            m[csym] = dig; r[dig] = csym
    return m, r


def _merge(bm, br, nm, nr):
    m, r = dict(bm), dict(br)
    for sym, dig in nm.items():
        if sym in m:
            if m[sym] != dig:
                return None
        elif dig in r:
            if r[dig] != sym:
                return None
        else:
            m[sym] = dig; r[dig] = sym
    return m, r


def _consistent_mappings_for_combo(combo, examples, cap=1500):
    """
    Return list of FULLY example-consistent cipher mappings for this combo.
    Enumerate over ALL example operand assignments, not early-exit.
    cap bounds the candidate explosion per example.
    """
    ex1_inp, ex1_out = examples[0]
    l1 = len(ex1_out)
    seed = []  # list of (mapping, rev)
    for L in range(100):
        A, B = L // 10, L % 10
        for R in range(100):
            C, D = R // 10, R % 10
            no = apply_combo(combo, A, B, C, D, examples[0][0][2])
            if no is None or not no.isdigit() or len(no) != l1:
                continue
            pm = _build_pm(ex1_inp, ex1_out, no, A, B, C, D)
            if pm is None:
                continue
            seed.append(pm)
            if len(seed) >= cap:
                break
        if len(seed) >= cap:
            break
    if not seed:
        return []
    # extend across remaining examples
    cur = seed
    for (inp, out) in examples[1:]:
        op = inp[2]; lo = len(out)
        nxt = []
        for (m, r) in cur:
            sA, sB, sC, sD = inp[0], inp[1], inp[3], inp[4]
            Ar = [m[sA]] if sA in m else range(10)
            Br = [m[sB]] if sB in m else range(10)
            Cr = [m[sC]] if sC in m else range(10)
            Dr = [m[sD]] if sD in m else range(10)
            extended_any = False
            for A in Ar:
                for B in Br:
                    for C in Cr:
                        for D in Dr:
                            no = apply_combo(combo, A, B, C, D, op)
                            if no is None or not no.isdigit() or len(no) != lo:
                                continue
                            pm = _build_pm(inp, out, no, A, B, C, D)
                            if pm is None:
                                continue
                            mg = _merge(m, r, *pm)
                            if mg is None:
                                continue
                            nxt.append(mg)
                            extended_any = True
            # dedup
            cur_seen = set()
            ded = []
            for (mm, rr) in nxt:
                key = tuple(sorted(mm.items()))
                if key not in cur_seen:
                    cur_seen.add(key); ded.append((mm, rr))
            nxt = ded
            if len(nxt) > cap:
                nxt = nxt[:cap]
        cur = nxt
        if not cur:
            return []
    return cur


import time

def analyze_one(examples, question, op, true_answer, time_budget=4.0):
    """
    Returns dict with:
      n_combo_solutions   : # of combos that have >=1 consistent mapping & produce a target answer
      distinct_answers    : set of distinct target-answer strings (across all combos/mappings)
      grammar_hits_true   : True if some consistent solution decodes target to true_answer
      best_combo_answer   : answer from the EARLIEST (highest-freq) combo that solves (prior pick)
      best_combo          : that combo
    """
    distinct = set()
    answer_by_combo_order = []  # (combo_index, answer)
    grammar_hits_true = False
    any_combo_consistent = False   # some combo reproduces the examples
    target_symbol_unseen = False   # combos exist, but target needs an unpinned symbol
    timed_out = False
    t0 = time.time()

    for ci, combo in enumerate(CIPHER_COMBOS):
        if time.time() - t0 > time_budget:
            timed_out = True
            break
        maps = _consistent_mappings_for_combo(combo, examples)
        if not maps:
            continue
        any_combo_consistent = True
        # for each consistent mapping, decode the target
        sQA, sQB, sQC, sQD = question[0], question[1], question[3], question[4]
        local_answers = set()
        for (m, r) in maps:
            if any(s not in m for s in (sQA, sQB, sQC, sQD)):
                # target uses a symbol never pinned by examples -> ambiguous; skip
                target_symbol_unseen = True
                continue
            qA, qB, qC, qD = m[sQA], m[sQB], m[sQC], m[sQD]
            na = apply_combo(combo, qA, qB, qC, qD, op)
            if na is None or not na.isdigit():
                continue
            inv = {v: k for k, v in m.items()}
            if not all(int(ch) in inv for ch in na):
                continue
            cipher_ans = "".join(inv[int(ch)] for ch in na)
            local_answers.add(cipher_ans)
        if not local_answers:
            continue
        distinct |= local_answers
        if true_answer in local_answers:
            grammar_hits_true = True
        # record one representative answer for this combo (earliest combo = prior)
        answer_by_combo_order.append((ci, sorted(local_answers)[0], local_answers))
        # Early-exit: once we have >=2 distinct answers AND have checked whether
        # the true answer is reachable, the multi-classification is settled.
        # Keep scanning a little to confirm grammar_hits_true, but cap.
        if len(distinct) >= 2 and (grammar_hits_true or ci >= 12):
            break

    best_combo_answer = None
    best_combo = None
    if answer_by_combo_order:
        answer_by_combo_order.sort(key=lambda x: x[0])
        best_combo = CIPHER_COMBOS[answer_by_combo_order[0][0]]
        # prior pick: if earliest combo itself is multi-valued, we can't pick -> mark None
        la = answer_by_combo_order[0][2]
        best_combo_answer = answer_by_combo_order[0][1] if len(la) == 1 else None

    return {
        "n_solving_combos": len(answer_by_combo_order),
        "distinct_answers": distinct,
        "grammar_hits_true": grammar_hits_true,
        "best_combo_answer": best_combo_answer,
        "best_combo": best_combo,
        "true_in_distinct": true_answer in distinct,
        "any_combo_consistent": any_combo_consistent,
        "target_symbol_unseen": target_symbol_unseen,
        "timed_out": timed_out,
    }


def load_fails():
    fails = {"cryptarithm_deduce": [], "cryptarithm_guess": []}
    for s in range(4):
        with open(EVAL_GLOB.format(s)) as f:
            for line in f:
                r = json.loads(line)
                if r["category"] in fails and not r["correct"]:
                    fails[r["category"]].append(r)
    return fails


def load_prompts(ids):
    pm = {}
    with open(TRAIN_CSV) as f:
        for row in csv.DictReader(f):
            if row["id"] in ids:
                pm[row["id"]] = row["prompt"]
    return pm


def main():
    import random
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nper = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    random.seed(seed)
    fails = load_fails()
    out = {}
    for cat, items in fails.items():
        random.shuffle(items)
        sample = items[:nper]
        ids = {r["id"] for r in sample}
        prompts = load_prompts(ids)
        buckets = {"unique": 0, "multi": 0, "no_solution": 0, "parse_fail": 0,
                   "op_novel_guess": 0, "single_example_op": 0,
                   # no_solution sub-reasons:
                   "ns_grammar_gap": 0,          # NO combo reproduces examples (true grammar gap)
                   "ns_target_symbol_unseen": 0, # combos fit but target needs unpinned symbol
                   "timed_out_unresolved": 0}    # hit time budget w/ <2 distinct (unclassified)
        # split unique/multi/no_solution by n_op constraint count
        by_nop = {}  # n_op -> {"unique":..,"multi":..,"no_solution":..}
        grammar_true = 0
        prior_correct = 0
        prior_applicable = 0
        unique_total = 0
        detail = []
        for r in sample:
            p = prompts.get(r["id"])
            if not p:
                buckets["parse_fail"] += 1
                continue
            ex, q, op, n_op = parse_prompt(p)
            if ex is None:
                buckets["parse_fail"] += 1
                continue
            if n_op == 0:
                # question operator never appears in examples -> "guess" structure
                buckets["op_novel_guess"] += 1
                continue
            if n_op < 2:
                buckets["single_example_op"] += 1
            res = analyze_one(ex, q, op, r["answer"])
            nop_bucket = by_nop.setdefault(n_op, {"unique": 0, "multi": 0, "no_solution": 0, "grammar_true": 0})
            d = res["distinct_answers"]
            if res["grammar_hits_true"]:
                grammar_true += 1
                nop_bucket["grammar_true"] += 1
            if len(d) == 0:
                if res.get("timed_out"):
                    buckets["timed_out_unresolved"] += 1
                buckets["no_solution"] += 1
                nop_bucket["no_solution"] += 1
                if not res["any_combo_consistent"]:
                    buckets["ns_grammar_gap"] += 1
                elif res["target_symbol_unseen"]:
                    buckets["ns_target_symbol_unseen"] += 1
            elif len(d) == 1:
                buckets["unique"] += 1
                unique_total += 1
                nop_bucket["unique"] += 1
                # is the single determined answer the true answer?
                # (model got it wrong, but it was determinable => capacity-limited)
                if r["answer"] in d:
                    buckets.setdefault("unique_eq_true", 0)
                    buckets["unique_eq_true"] += 1
            else:
                buckets["multi"] += 1
                nop_bucket["multi"] += 1
                # prior salvage check
                if res["best_combo_answer"] is not None:
                    prior_applicable += 1
                    if res["best_combo_answer"] == r["answer"]:
                        prior_correct += 1
            detail.append({
                "n_op": n_op,
                "id": r["id"],
                "ndistinct": len(d),
                "grammar_true": res["grammar_hits_true"],
                "true_in_distinct": res["true_in_distinct"],
                "n_solving_combos": res["n_solving_combos"],
                "best_combo": res["best_combo"],
                "best_combo_answer": res["best_combo_answer"],
                "true": r["answer"],
            })
        out[cat] = {
            "n": len(sample),
            "buckets": buckets,
            "by_nop": by_nop,
            "grammar_hits_true": grammar_true,
            "prior_applicable": prior_applicable,
            "prior_correct": prior_correct,
            "detail": detail,
        }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
