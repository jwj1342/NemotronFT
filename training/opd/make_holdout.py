#!/usr/bin/env python3
"""make_holdout.py — stratified train/holdout splitter for RFT/STaR self-training.

WHY THIS EXISTS
---------------
RFT/STaR is self-training: the model generates its own CoT, we keep the
answer-correct ones, and we SFT on them. The cardinal failure mode is
*fake gains*: the model memorizes the specific puzzles it was sampled on, the
train-set "accuracy" climbs, but nothing transfers to unseen puzzles of the
same family. Each cryptarithm/bit puzzle is an independent random cipher, so
this is not a theoretical worry — it is exactly what killed the canary run
(ITERATION_REPORT.md §: canary 2/30 vs baseline 3/30 even though train_loss
dropped 95 -> 13).

To detect that, we must NEVER train on the same puzzles we score on. This tool
carves a disjoint, same-distribution `holdout_split` out of each target
category *before any sampling happens*, and the holdout is touched by exactly
one piece of code: the eval step. Concretely:

  train_split  = pool the RFT prober samples CoT from + SFTs on
  holdout_split = pool we ONLY ever eval on (positive-transfer gate)

Same-distribution guarantee: the split is stratified by `category` (and, within
a category, the holdout draws from BOTH the model's current failures and a
matched fraction of its current successes). Why mix successes into holdout?
Because if holdout were 100% current-failures, a true generalization gain and a
mean-reversion artifact look identical; including known-correct puzzles lets you
check that RFT did not *regress* the things the baseline already had right
(the canary lesson: SFT on cryptarithm caused negative transfer on the held
puzzles). The `--holdout_correct_frac` knob controls that mix.

WHAT IT READS
-------------
  1. Baseline eval JSONL shards (authoritative `correct` + `category` per id):
       output/eval_full_<adapter>_shard{0..3}of4.jsonl
     Each line: {id, category, answer, predicted, correct, num_chars, num_tokens}
  2. data/train.csv (id -> prompt; columns: id, prompt, answer) via stdlib csv.

WHAT IT WRITES (one dir per run, default training/opd/runs/holdout/<tag>/)
--------------------------------------------------------------------------
  train_split.jsonl      one line per train-pool puzzle
  holdout_split.jsonl    one line per holdout puzzle  (NEVER sampled/SFT'd)
  split_manifest.json    config + per-category counts + the exact id lists,
                         so the split is reproducible and auditable.

Each JSONL line is the schema the rest of the pipeline already speaks
(rollout.py / build_hotpool.py / eval_vllm.py):
    {"id", "prompt", "gt", "category", "baseline_correct"}
`gt` (not `answer`) matches rollout/reward; `baseline_correct` lets the prober
prioritise failures and lets eval compute delta-vs-baseline on the holdout.

This script is CPU-only, stdlib-only, runs in seconds on the login node.
It does NOT call the model and does NOT need a GPU.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/scratch/jwj/NemotronFT")

# The three categories where the baseline has real headroom AND a verifier
# exists (so RFT filtering is meaningful). bit_manipulation: 87.5% -> headroom
# in the 200 fails; cryptarithm_deduce: 7%; equation_numeric_guess: 16%.
# cipher/gravity/numeral/unit_conversion are >=99% — nothing to gain, skip.
DEFAULT_TARGET_CATS = [
    "bit_manipulation",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
]


def load_eval_shards(pattern: str) -> dict[str, dict]:
    """id -> {category, answer, correct} from the baseline eval shards."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[make_holdout] no eval shards match: {pattern}")
    by_id: dict[str, dict] = {}
    for fn in files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                by_id[d["id"]] = {
                    "category": d["category"],
                    "answer": str(d["answer"]),
                    "correct": bool(d["correct"]),
                }
    print(f"[make_holdout] loaded {len(by_id)} eval rows from {len(files)} shards")
    return by_id


def load_prompts(train_csv: Path) -> dict[str, str]:
    """id -> prompt from data/train.csv (stdlib csv, no pandas)."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    id2prompt: dict[str, str] = {}
    with open(train_csv) as f:
        for r in csv.DictReader(f):
            id2prompt[r["id"]] = r["prompt"]
    print(f"[make_holdout] loaded {len(id2prompt)} prompts from {train_csv.name}")
    return id2prompt


def stratified_split(
    by_id: dict[str, dict],
    id2prompt: dict[str, str],
    target_cats: list[str],
    holdout_fail_frac: float,
    holdout_correct_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Carve disjoint train/holdout pools, stratified by category and correctness.

    For each target category we split FAILS and CORRECTS independently so the
    holdout contains a controlled, same-distribution mix of both:
      - holdout_fail_frac    of the category's CURRENT failures go to holdout
      - holdout_correct_frac of the category's CURRENT successes go to holdout
    Everything else (all remaining fails + corrects) goes to train_split. The
    RFT prober will then sample CoT only over train_split fails (primary) and
    optionally re-confirm train_split corrects (to keep distinct correct paths).
    """
    rng = random.Random(seed)

    # Bucket ids by (category, correct), keeping only ids we have prompts for.
    buckets: dict[tuple[str, bool], list[str]] = defaultdict(list)
    n_no_prompt = 0
    for pid, info in by_id.items():
        if info["category"] not in target_cats:
            continue
        if pid not in id2prompt:
            n_no_prompt += 1
            continue
        buckets[(info["category"], info["correct"])].append(pid)
    if n_no_prompt:
        print(f"[make_holdout] WARNING: {n_no_prompt} target ids had no prompt in train.csv")

    def _make_rec(pid: str) -> dict:
        info = by_id[pid]
        return {
            "id": pid,
            "prompt": id2prompt[pid],
            "gt": info["answer"],
            "category": info["category"],
            "baseline_correct": info["correct"],
        }

    train_recs: list[dict] = []
    holdout_recs: list[dict] = []
    per_cat: dict[str, dict] = {}

    for cat in target_cats:
        fails = sorted(buckets.get((cat, False), []))   # sort -> deterministic pre-shuffle
        corrects = sorted(buckets.get((cat, True), []))
        rng.shuffle(fails)
        rng.shuffle(corrects)

        n_hold_fail = int(round(len(fails) * holdout_fail_frac))
        n_hold_corr = int(round(len(corrects) * holdout_correct_frac))

        hold_fail = fails[:n_hold_fail]
        train_fail = fails[n_hold_fail:]
        hold_corr = corrects[:n_hold_corr]
        train_corr = corrects[n_hold_corr:]

        for pid in train_fail + train_corr:
            train_recs.append(_make_rec(pid))
        for pid in hold_fail + hold_corr:
            holdout_recs.append(_make_rec(pid))

        per_cat[cat] = {
            "n_fail_total": len(fails),
            "n_correct_total": len(corrects),
            "holdout_fail": len(hold_fail),
            "holdout_correct": len(hold_corr),
            "train_fail": len(train_fail),
            "train_correct": len(train_corr),
            "holdout_ids": hold_fail + hold_corr,
        }

    manifest = {
        "seed": seed,
        "target_cats": target_cats,
        "holdout_fail_frac": holdout_fail_frac,
        "holdout_correct_frac": holdout_correct_frac,
        "n_train": len(train_recs),
        "n_holdout": len(holdout_recs),
        "per_category": per_cat,
    }
    return train_recs, holdout_recs, manifest


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def assert_disjoint(train_recs: list[dict], holdout_recs: list[dict]) -> None:
    """Hard guard: train and holdout id-sets MUST NOT intersect. This is the
    single invariant the whole anti-fake-gain argument rests on."""
    t = {r["id"] for r in train_recs}
    h = {r["id"] for r in holdout_recs}
    overlap = t & h
    if overlap:
        raise SystemExit(
            f"[make_holdout] FATAL: {len(overlap)} ids in BOTH splits "
            f"(e.g. {sorted(overlap)[:5]}). Refusing to write a leaky split."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_glob",
                    default=str(PROJECT / "output" /
                                "eval_full_huikang-baseline-converted_shard*of4.jsonl"),
                    help="glob for the baseline eval JSONL shards (authoritative correct/category)")
    ap.add_argument("--train_csv", default=str(PROJECT / "data" / "train.csv"),
                    help="id->prompt source (stdlib csv)")
    ap.add_argument("--out_dir",
                    default=str(PROJECT / "training" / "opd" / "runs" / "holdout" / "v1"),
                    help="where to write train_split.jsonl / holdout_split.jsonl / manifest")
    ap.add_argument("--target_cats", nargs="*", default=DEFAULT_TARGET_CATS,
                    help="categories to include (eval `category` field is authoritative)")
    ap.add_argument("--holdout_fail_frac", type=float, default=0.30,
                    help="fraction of each category's CURRENT failures reserved for holdout")
    ap.add_argument("--holdout_correct_frac", type=float, default=0.10,
                    help="fraction of CURRENT successes reserved for holdout (regression guard)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    by_id = load_eval_shards(args.eval_glob)
    id2prompt = load_prompts(Path(args.train_csv))
    train_recs, holdout_recs, manifest = stratified_split(
        by_id, id2prompt, args.target_cats,
        args.holdout_fail_frac, args.holdout_correct_frac, args.seed,
    )
    assert_disjoint(train_recs, holdout_recs)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train_split.jsonl", train_recs)
    write_jsonl(out_dir / "holdout_split.jsonl", holdout_recs)
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[make_holdout] wrote -> {out_dir}")
    print(f"  train_split.jsonl   : {manifest['n_train']} puzzles")
    print(f"  holdout_split.jsonl : {manifest['n_holdout']} puzzles")
    print(f"  split_manifest.json : config + per-category counts + holdout ids")
    print("\n  per-category (fail/correct -> train | holdout):")
    for cat in args.target_cats:
        c = manifest["per_category"].get(cat)
        if not c:
            print(f"    {cat:26s} (absent from eval)")
            continue
        print(f"    {cat:26s} "
              f"fail {c['train_fail']:4d}+{c['holdout_fail']:3d}h  "
              f"corr {c['train_correct']:4d}+{c['holdout_correct']:3d}h")
    print("\n  GUARANTEE: train ∩ holdout = ∅ (asserted). Score holdout ONLY.")


if __name__ == "__main__":
    main()
