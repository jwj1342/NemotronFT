"""
Branch-budget ceiling audit for cryptarithm_deduce.

For each of the 659 real crd problems we compute branch_width = the MINIMUM over
fact-processing orderings of the MAXIMUM live-hypothesis count across all steps,
where a "live hypothesis" is a distinct partial joint assignment
(operator-map entries + cipher entries) consistent with the facts processed so
far, under a FIXED global preprocessing scheme pp in {none,swap}^3.

A problem is "K-solvable" if there exists an ordering+pp such that:
  * the max live-hypothesis count across every step is <= K, AND
  * the anchor (first) fact's STANDALONE consistent (op,cipher) count is <= K
    (no hidden full [0,99]^2 scan), AND
  * after all facts the surviving live set, applied to the query, reproduces the
    KNOWN answer with a UNIQUE prediction (i.e. converges to the correct
    op-map+cipher+pp).

We reuse crd_solver's parsing + op table + cipher machinery.
"""
from __future__ import annotations

import sys, os, json, glob, csv, statistics, itertools
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crd_solver as S

PROJECT = Path("/scratch/jwj/NemotronFT")
PREPROCS = S.PREPROCS  # ["none","swap"]
ALL_OPS = S.ALL_OPS

# A "hypothesis" carried through processing = (op_constraints_dict, cipher_dict).
# We store them as frozen tuples for de-duplication of the live SET.

def _freeze(op_c, cip):
    return (tuple(sorted(op_c.items())), tuple(sorted(cip.items())))


def extend_live(live, fact, op_syms, pp, cap):
    """Extend a list of live hypotheses by one fact under fixed pp.
    Returns (new_live_list, overflow_bool). De-duplicates. Bounded by `cap`:
    if the new live set would exceed `cap` distinct hypotheses we stop and flag
    overflow (this ordering is then not <=cap)."""
    op_sym = fact[1]
    seen = {}
    overflow = False
    for op_c, cip in live:
        ops = [op_c[op_sym]] if op_sym in op_c else ALL_OPS
        for opn, nc in S.extend_fact(fact, cip, op_syms, pp, ops, budget=None):
            new_oc = dict(op_c)
            new_oc[op_sym] = opn
            key = _freeze(new_oc, nc)
            if key not in seen:
                seen[key] = (new_oc, nc)
                if len(seen) > cap:
                    overflow = True
                    return list(seen.values()), True
    return list(seen.values()), overflow


def standalone_count(fact, op_syms, pp, cap):
    """Standalone consistent (op,cipher) count for ONE fact starting from empty.
    Bounded by cap+1 (returns cap+1 sentinel on overflow)."""
    seed = [({}, {})]
    new, overflow = extend_live(seed, fact, op_syms, pp, cap)
    return (cap + 1) if overflow else len(new)


def query_predictions(live, query, op_syms, sign_sym, fwd_alphabet):
    """Apply each live hypothesis to the query; return set of distinct predicted
    answer strings (mirrors crd_solver.solve's query-application logic)."""
    qa, qop, qb = query
    preds = set()
    for op_c, cip in live:
        opn = op_c.get(qop)
        if opn is None:
            continue
        cipc = S._complete_cipher(cip, fwd_alphabet)
        if opn in ("concat", "rconcat"):
            ans = S._encode_concat(qa, qb, opn, pp_holder[0], sign_sym)
        else:
            a2 = S._decode(qa, cipc)
            b2 = S._decode(qb, cipc)
            if a2 is None or b2 is None:
                continue
            a_val = int(S._operand_string(a2, pp_holder[0][0]))
            b_val = int(S._operand_string(b2, pp_holder[0][1]))
            raw = S._apply_op(opn, a_val, b_val)
            if raw is None:
                continue
            ans = S._encode_answer(raw, pp_holder[0][2], cipc, sign_sym, op_syms)
        if ans is not None:
            preds.add(ans)
    return preds


# pp is needed inside query_predictions for concat/operand strings; pass via a
# tiny holder to avoid threading it through every call.
pp_holder = [None]


OVERFLOW = object()  # sentinel: live set exceeds cap (treated as > K_max)


def _live_subset(facts, op_syms, pp, cap):
    """Compute live(S) for every subset S of fact-indices via memoized lattice
    DP (order-independent). live(S) = all distinct partial hypotheses consistent
    with exactly the facts in S. Capped at `cap`; an overflowing subset stores
    the OVERFLOW sentinel (and so does every superset).
    Returns {frozenset: live_list_or_OVERFLOW}."""
    nf = len(facts)
    memo = {frozenset(): [({}, {})]}
    # build by increasing subset size so predecessors are ready
    idxs = list(range(nf))
    for size in range(1, nf + 1):
        for S in itertools.combinations(idxs, size):
            Sf = frozenset(S)
            # extend from the cheapest available predecessor
            best_pred = None
            best_pred_len = None
            for i in S:
                pred = memo[Sf - {i}]
                if pred is OVERFLOW:
                    continue
                if best_pred_len is None or len(pred) < best_pred_len:
                    best_pred = (i, pred)
                    best_pred_len = len(pred)
            if best_pred is None:
                memo[Sf] = OVERFLOW
                continue
            i, pred = best_pred
            new, of = extend_live(pred, facts[i], op_syms, pp, cap)
            memo[Sf] = OVERFLOW if of else new
    return memo


def best_branch_width(facts, query, op_syms, sign_sym, answer, fwd_alphabet, K_max):
    """Return dict with:
      branch_width : min over (pp, ordering) of (max live-hyp count across steps),
                     considering ONLY (pp,ordering) whose anchor standalone count
                     is in [1, cap] (no hidden full scan; a 0-count anchor is a
                     self-contradictory fact and cannot anchor) AND whose final
                     live set converges to the UNIQUE correct query answer.
                     None if no such ordering exists at any width <= K_max.
      bw_any       : same minimax width but WITHOUT the convergence requirement
                     (for the branch_width distribution).
      min_anchor   : min POSITIVE standalone count over all facts & pp (the
                     cheapest valid anchor available). None if no fact can anchor
                     under any pp within cap.
      best_meta    : (pp, op_map, cipher) of a representative converging solution
                     at the best (smallest) branch_width.

    Order-independent live(S) memoization makes the minimax-width DP
      bw(S) = min_{i in S} max(|live(S)|, bw(S\{i}))
    exact and cheap (<=2^nf subsets).
    """
    cap = K_max
    nf = len(facts)
    full = frozenset(range(nf))

    min_anchor = None          # min POSITIVE singleton count over facts & pp
    best_converge_bw = None
    best_meta = None
    bw_any = None

    for pp in [(a, b, c) for a in PREPROCS for b in PREPROCS for c in PREPROCS]:
        pp_holder[0] = pp
        memo = _live_subset(facts, op_syms, pp, cap)

        # cheapest valid (>0, <=cap) anchor under this pp
        for i in range(nf):
            sing = memo[frozenset({i})]
            if sing is OVERFLOW:
                continue
            c = len(sing)
            if 1 <= c <= cap and (min_anchor is None or c < min_anchor):
                min_anchor = c

        # minimax elimination-ordering width DP, restricted to orderings whose
        # FIRST fact (the anchor, a singleton prefix) has count in [1,cap].
        # bw_memo[S] = min over orderings of S of max prefix live-size, or None
        # if every ordering of S overflows / has an invalid anchor.
        bw_memo = {frozenset(): 0}

        def bw(S):
            if S in bw_memo:
                return bw_memo[S]
            cur = memo[S]
            if cur is OVERFLOW:
                bw_memo[S] = None
                return None
            cur_len = len(cur)
            best = None
            for i in S:
                rest = S - {i}
                if not rest:
                    # i is the anchor (first processed). require 1<=count<=cap.
                    if 1 <= cur_len <= cap:
                        cand = cur_len
                    else:
                        cand = None
                else:
                    sub = bw(rest)
                    if sub is None:
                        cand = None
                    else:
                        cand = max(cur_len, sub)
                if cand is not None and (best is None or cand < best):
                    best = cand
            bw_memo[S] = best
            return best

        # bw over the full set without convergence (for distribution)
        w = bw(full)
        if w is not None:
            if bw_any is None or w < bw_any:
                bw_any = w

        # convergence: full live set must yield a unique correct query prediction
        full_live = memo[full]
        if full_live is OVERFLOW or not full_live:
            continue
        preds = query_predictions(full_live, query, op_syms, sign_sym, fwd_alphabet)
        if len(preds) == 1 and answer in preds:
            if w is not None and (best_converge_bw is None or w < best_converge_bw):
                best_converge_bw = w
                op_map, cip = full_live[0]
                best_meta = (pp, dict(op_map), dict(cip))

    return {
        "branch_width": best_converge_bw,
        "bw_any": bw_any,
        "min_anchor": min_anchor,
        "best_meta": best_meta,
    }


def classify_meta(meta, answer, op_syms):
    """Flags for the BEST converging solution: swap-pp, sign result, mul+-1."""
    pp, op_map, cip = meta
    needs_swap = (pp != ("none", "none", "none"))
    needs_sign = (len(answer) >= 2 and answer[0] in op_syms)
    needs_mulpm = any(op in ("mul+1", "mul-1") for op in op_map.values())
    return needs_swap, needs_sign, needs_mulpm


def main():
    K_LEVELS = [5, 10, 20]
    K_MAX = max(K_LEVELS)

    # join
    ids = set()
    for f in glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d.get("category") == "cryptarithm_deduce":
                ids.add(d["id"])
    baseline_correct = {}
    for f in glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d.get("category") == "cryptarithm_deduce":
                baseline_correct[d["id"]] = bool(d.get("correct"))
    rows = {}
    with open(PROJECT / "data/train.csv") as f:
        for r in csv.DictReader(f):
            if r["id"] in ids:
                rows[r["id"]] = r

    N = len(rows)
    results = {}            # id -> result dict from best_branch_width
    branch_widths = []      # converging branch width (for K-solvable problems); use bw_any for dist
    bw_any_list = []
    min_anchor_list = []
    bad_parse = 0

    items = list(rows.items())
    for idx, (pid, v) in enumerate(items):
        facts, query, op_syms = S.parse_prompt(v["prompt"])
        if facts is None:
            bad_parse += 1
            results[pid] = None
            continue
        sign_sym = S._find_sign_symbol(facts, op_syms)
        fwd_alphabet = S._digit_alphabet(facts, query, None, op_syms)
        answer = v["answer"].strip()
        res = best_branch_width(facts, query, op_syms, sign_sym, answer,
                                fwd_alphabet, K_MAX)
        results[pid] = res
        if res["bw_any"] is not None:
            bw_any_list.append(res["bw_any"])
        if res["branch_width"] is not None:
            branch_widths.append(res["branch_width"])
        if res["min_anchor"] is not None:
            min_anchor_list.append(res["min_anchor"])
        if (idx + 1) % 50 == 0:
            print(f"  ...processed {idx+1}/{N}", file=sys.stderr)

    # ---- per-K aggregation ----
    out = []
    def w(s=""):
        out.append(s); print(s)

    w(f"=== crd branch-budget ceiling audit: {N} problems "
      f"({bad_parse} parse-fail) ===")
    w(f"baseline-correct: {sum(baseline_correct.values())}  "
      f"baseline-wrong: {N - sum(baseline_correct.values())}")
    w("")

    per_k_table = []
    for K in K_LEVELS:
        ksolv = 0
        bw_and_wrong = 0
        need_swap = need_sign = need_mulpm = 0
        ksolv_ids = []
        for pid, res in results.items():
            if res is None:
                continue
            bw = res["branch_width"]
            ok = (bw is not None and bw <= K and res["min_anchor"] is not None
                  and res["min_anchor"] <= K)
            if ok:
                ksolv += 1
                ksolv_ids.append(pid)
                if not baseline_correct.get(pid, False):
                    bw_and_wrong += 1
                    ns, nsg, nm = classify_meta(res["best_meta"], rows[pid]["answer"].strip(),
                                                S.parse_prompt(rows[pid]["prompt"])[2])
                    need_swap += ns; need_sign += nsg; need_mulpm += nm
        lb_gain = 0.075 * (bw_and_wrong / N)
        abs_pts = 100 * (bw_and_wrong / N)
        per_k_table.append((K, ksolv, bw_and_wrong, need_swap, need_sign,
                            need_mulpm, lb_gain, abs_pts, ksolv_ids))

    w("Per-K table:")
    w(f"{'K':>4} {'K-solv':>10} {'K-solv%':>8} {'wrong&Ksolv':>12} {'frac659':>8} "
      f"{'swap':>6} {'sign':>6} {'mul±1':>6} {'estLBgain':>10} {'absCryptPts':>12}")
    for (K, ksolv, bw_wrong, ns, nsg, nm, lbg, abspts, _ids) in per_k_table:
        w(f"{K:>4} {ksolv:>10} {100*ksolv/N:>7.1f}% {bw_wrong:>12} "
          f"{bw_wrong/N:>8.3f} {ns:>6} {nsg:>6} {nm:>6} "
          f"{lbg:>10.4f} {abspts:>11.2f}p")
    w("")
    w("(swap/sign/mul±1 counts are over the wrong&K-solvable HEADROOM subset)")
    w("")

    # ---- distributions ----
    def dist(name, lst):
        if not lst:
            w(f"{name}: (empty)")
            return
        s = sorted(lst)
        med = statistics.median(s)
        p90 = s[min(len(s)-1, int(0.90*len(s)))]
        p95 = s[min(len(s)-1, int(0.95*len(s)))]
        w(f"{name}: n={len(lst)} median={med} p90={p90} p95={p95} max={max(s)} min={min(s)}")

    w("Distributions:")
    dist("branch_width (converging, capped @%d)" % K_MAX, bw_any_list)
    dist("min anchor-fact standalone count (capped @%d)" % K_MAX, min_anchor_list)
    # how many never converge at all within K_MAX
    n_converge = sum(1 for r in results.values() if r and r["branch_width"] is not None)
    n_noconverge = sum(1 for r in results.values() if r and r["branch_width"] is None)
    w(f"problems that CONVERGE to truth at some width<= {K_MAX}: {n_converge}")
    w(f"problems that do NOT converge within width<= {K_MAX}: {n_noconverge}")
    w("")

    # ---- verdict ----
    best = max(per_k_table, key=lambda t: t[7])  # max abs_pts among K<=20
    bestK, _, _, _, _, _, _, best_abspts, _ = best
    w(f"BEST coverage (abs crypt points over baseline-wrong) at K={bestK}: {best_abspts:.2f}p")
    if best_abspts < 15:
        verdict = "KILL cryptarithm (<15 pts)"
    elif best_abspts < 25:
        verdict = "worth ONE narrow subset training run (15-25 pts)"
    elif best_abspts < 35:
        verdict = "worth a serious 2-node run (25-35 pts)"
    else:
        verdict = "real 0.89 path (35+ pts)"
    w(f"VERDICT: {verdict}")
    w("")

    # ---- representative examples: baseline-wrong AND K-solvable at best K ----
    w("Representative branch-solvable baseline-WRONG problems:")
    examples_ids = []
    for pid in best[8]:
        if not baseline_correct.get(pid, False):
            examples_ids.append(pid)
        if len(examples_ids) >= 3:
            break
    for pid in examples_ids:
        res = results[pid]
        pp, op_map, cip = res["best_meta"]
        w(f"  id={pid}  branch_width={res['branch_width']}  anchor={res['min_anchor']}  pp={pp}")
        w(f"     answer={rows[pid]['answer'].strip()!r}  op_map={op_map}  cipher={cip}")
        # show the prompt compactly
        plines = [l.strip() for l in rows[pid]['prompt'].strip().split('\n') if l.strip()]
        for pl in plines[-7:]:
            w(f"        {pl}")
        w("")

    Path(PROJECT / "data/crd_branch_audit.txt").write_text("\n".join(out) + "\n")
    print("\n[written to data/crd_branch_audit.txt]", file=sys.stderr)


if __name__ == "__main__":
    main()
