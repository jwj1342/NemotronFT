"""
Build the cryptarithm_deduce SFT dataset + honest holdout.

Outputs (under /scratch/jwj/NemotronFT/data):
  crd_sft_train.jsonl   {prompt, answer, cot, source}  (synthetic + real-train)
  crd_holdout.jsonl     {prompt, answer, category, cot} (~100 held-out REAL well-posed)

Pure CPU. Parallelises synthetic generation + real-trace rendering across cores.
Prints final counts, trace-token distribution, and 2 full sample traces
(one swap-pp, one negative-result) for eyeballing forward-deducibility.

Run (login node ok, uses up to ~$SLURM_CPUS_PER_TASK or os.cpu_count() workers):
  module purge && module load python/3.11.5 && source venv/bin/activate && unset PIP_PREFIX
  python training/opd/build_crd_sft.py --n_synth 4000 --n_holdout 100
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys
from pathlib import Path

SOLVERS = "/scratch/jwj/NemotronFT/training/opd/solvers"
sys.path.insert(0, SOLVERS)

PROJECT = Path("/scratch/jwj/NemotronFT")
DATA = PROJECT / "data"


# ---------------------------------------------------------------------------
# Worker entry points (module-level so multiprocessing can pickle them).
# ---------------------------------------------------------------------------

def _synth_chunk(args):
    """Generate up to n problems with a seeded RNG; return list of rows."""
    seed, n, cap = args
    import crd_synth
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n and tries < cap:
        tries += 1
        p = crd_synth.gen_problem(rng)
        if p is not None:
            out.append({
                "prompt": p["prompt"], "answer": p["answer"], "cot": p["cot"],
                "n_facts": p["n_facts"], "has_negative": bool(p["has_negative"]),
                "swap": p["pp"] != ("none", "none", "none"),
                "source": "synthetic",
            })
    return out


def _real_trace(args):
    """Render+verify a real problem; return row dict or None."""
    pid, prompt, answer = args
    from crd_solver import solve, parse_prompt, _split_sign
    from crd_trace import render_trace, verify_trace
    s = solve(prompt, answer)
    if not (s.get("solved") and s.get("well_posed") and s.get("predicted") == answer):
        return ("notwp", pid)
    tr = render_trace(prompt, answer)
    if tr is None:
        return ("skip", pid)
    good, why = verify_trace(prompt, answer, tr)
    if not good:
        return ("fail", pid, why)
    facts, query, op_syms = parse_prompt(prompt)
    has_neg = any((_split_sign(f[3], op_syms) or (1, ""))[0] < 0 for f in facts)
    swap = s["preprocess"] != ("none", "none", "none")
    return ("ok", {
        "id": pid, "prompt": prompt, "answer": answer, "cot": tr,
        "has_negative": has_neg, "swap": swap, "source": "real",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_synth", type=int, default=4000)
    ap.add_argument("--n_holdout", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8)))
    args = ap.parse_args()

    import multiprocessing as mp
    from transformers import AutoTokenizer

    # ---- collect real crd problems ----
    ids = set()
    for f in glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d["category"] == "cryptarithm_deduce":
                ids.add(d["id"])
    real = []
    with open(PROJECT / "data/train.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] in ids:
                real.append((row["id"], row["prompt"], row["answer"]))
    print(f"real crd problems: {len(real)}", flush=True)

    workers = max(1, args.workers)
    print(f"workers: {workers}", flush=True)

    # ---- render real traces (parallel) ----
    with mp.Pool(workers) as pool:
        real_results = pool.map(_real_trace, real, chunksize=4)
    real_rows = [r[1] for r in real_results if r[0] == "ok"]
    n_notwp = sum(1 for r in real_results if r[0] == "notwp")
    n_skip = sum(1 for r in real_results if r[0] == "skip")
    n_fail = sum(1 for r in real_results if r[0] == "fail")
    fail_why = {}
    for r in real_results:
        if r[0] == "fail":
            fail_why[r[2].split(":")[0]] = fail_why.get(r[2].split(":")[0], 0) + 1
    print(f"real well-posed rendered+verified: {len(real_rows)}  "
          f"(not-well-posed/uncovered: {n_notwp}, skipped(unbounded/degenerate): {n_skip}, "
          f"fwd_fail: {n_fail} {fail_why})", flush=True)

    # ---- split holdout from REAL well-posed rows ----
    rng = random.Random(args.seed)
    rng.shuffle(real_rows)
    n_hold = min(args.n_holdout, len(real_rows) // 3)
    holdout = real_rows[:n_hold]
    real_train = real_rows[n_hold:]
    hold_ids = {r["id"] for r in holdout}
    assert not (hold_ids & {r["id"] for r in real_train}), "holdout leak into train"

    # ---- generate synthetic (parallel) ----
    per = args.n_synth // workers + 1
    chunk_args = [(args.seed * 1000 + i, per, per * 60) for i in range(workers)]
    with mp.Pool(workers) as pool:
        synth_lists = pool.map(_synth_chunk, chunk_args)
    synth_rows = [r for lst in synth_lists for r in lst][:args.n_synth]
    print(f"synthetic generated: {len(synth_rows)}", flush=True)

    # ---- write dataset ----
    DATA.mkdir(exist_ok=True)
    train_rows = synth_rows + [
        {"prompt": r["prompt"], "answer": r["answer"], "cot": r["cot"], "source": "real"}
        for r in real_train
    ]
    rng.shuffle(train_rows)
    with open(DATA / "crd_sft_train.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps({"prompt": r["prompt"], "answer": r["answer"],
                                "cot": r["cot"], "source": r["source"]}) + "\n")
    with open(DATA / "crd_holdout.jsonl", "w") as f:
        for r in holdout:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"],
                                "answer": r["answer"], "cot": r["cot"],
                                "category": "cryptarithm_deduce"}) + "\n")

    # ---- token-length distribution (model tokenizer) ----
    tok = AutoTokenizer.from_pretrained(str(PROJECT / "model"), trust_remote_code=True)
    lens = sorted(len(tok(r["cot"])["input_ids"]) for r in train_rows)

    def q(p):
        return lens[min(int(p * len(lens)), len(lens) - 1)]

    print("\n==== FINAL COUNTS ====", flush=True)
    print(f"train rows: {len(train_rows)}  (synthetic {len(synth_rows)} + real {len(real_train)})")
    print(f"holdout rows: {len(holdout)}  (all REAL well-posed, category=cryptarithm_deduce)")
    print(f"synth swap-pp: {sum(1 for r in synth_rows if r.get('swap'))}  "
          f"synth negative: {sum(1 for r in synth_rows if r.get('has_negative'))}")
    print(f"real-train swap-pp: {sum(1 for r in real_train if r.get('swap'))}  "
          f"real-train negative: {sum(1 for r in real_train if r.get('has_negative'))}")
    print(f"\n==== TRACE-TOKEN DISTRIBUTION (cot, model tokenizer) ====")
    print(f"median {statistics.median(lens)}  p95 {q(0.95)}  p99 {q(0.99)}  max {max(lens)}")
    print(f"targets: median<2500 p95<4500 p99<6000 -> "
          f"{'PASS' if statistics.median(lens) < 2500 and q(0.95) < 4500 and q(0.99) < 6000 else 'FAIL'}")

    # ---- 2 sample traces: one swap, one negative ----
    swap_ex = next((r for r in train_rows if r.get("source") and
                    (r in synth_rows and r.get("swap"))), None)
    # fall back: find swap among real_train
    if swap_ex is None:
        swap_ex = next((r for r in real_train if r.get("swap")), None)
    neg_ex = next((r for r in synth_rows if r.get("has_negative")), None)
    if neg_ex is None:
        neg_ex = next((r for r in real_train if r.get("has_negative")), None)
    print("\n==== SAMPLE TRACE (swap-pp) ====")
    if swap_ex:
        print("PROMPT:", repr(swap_ex["prompt"]))
        print("ANSWER:", swap_ex["answer"])
        print(swap_ex["cot"])
    else:
        print("(none found)")
    print("\n==== SAMPLE TRACE (negative result) ====")
    if neg_ex:
        print("PROMPT:", repr(neg_ex["prompt"]))
        print("ANSWER:", neg_ex["answer"])
        print(neg_ex["cot"])
    else:
        print("(none found)")


if __name__ == "__main__":
    main()
