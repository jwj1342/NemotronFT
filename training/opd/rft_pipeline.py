#!/usr/bin/env python3
"""rft_pipeline.py — RFT/STaR self-training pipeline for the Nemotron reasoning challenge.

================================================================================
DESIGN (read this whole header before touching the code)
================================================================================

The thesis (CLAUDE.md "the live direction"): the model is its own strongest
generator. External distillation and hand-written solvers both LOSE to the
baseline (solver bit 59.9% vs model 87.5%, numeric_deduce 39.8% vs 90.9%, etc.).
So we do RFT / STaR (a.k.a. answer-filtered self-training):

  ROUND r:
    (A) PROBE   — sample K CoT per train_split puzzle at T>0 (diversity matters
                  more than count). GPU phase, vLLM, runs in venv-vllm. Reuses
                  the rollout machinery: rollout.py already does exactly this
                  (n completions/prompt, T>0, no seed-with-n bug). The probe is
                  rollout.py with reward replaced by the two-gate filter below.
    (B) FILTER  — keep a completion ONLY if BOTH gates pass:
                  gate-1 (answer):  extract_final_answer(text) verify()s == gt.
                  gate-2 (rule):    the rule the CoT *committed to* reproduces
                                    EVERY in-context example (solver as VERIFIER,
                                    not generator). This rejects right-answer-by-
                                    luck / hallucinated-rule chains — the thing
                                    that poisons SFT. CPU, stdlib. THIS FILE.
    (C) DEDUP   — maximize DISTINCT correct reasoning paths. Gains scale with
                  distinct paths, not raw sample count (CLAUDE.md). Keep up to
                  --max_keep_per_id structurally-distinct chains per puzzle.
    (D) BUILD   — emit a CoT CSV in the EXACT schema HuikangCoTDataset wants:
                  columns id,prompt,answer,generated_cot,type. CPU, stdlib.
    (E) TRAIN   — LoRA SFT continuing from the Huikang baseline adapter
                  (NOT cold-start). Reuses training/train.py via a thin sbatch
                  modeled on train_v3.sbatch (--resume_adapter_dir + chain).
    (F) EVAL    — score the NEW adapter on holdout_split (eval/eval_vllm.py).
                  POSITIVE-TRANSFER GATE: keep the adapter only if holdout
                  accuracy rises vs baseline AND no target category regresses.
                  holdout was never sampled (make_holdout.py guarantees disjoint).
    (G) ITERATE — fold the new adapter back in as the prober for round r+1.

Anti-fake-gain discipline (Codex's emphasis):
  * train_split and holdout_split are disjoint by construction (make_holdout.py
    asserts it). We NEVER probe/SFT on holdout.
  * The transfer gate is holdout accuracy, never train accuracy / train loss.
    (The canary proved train_loss 95->13 with NEGATIVE transfer.)
  * gate-2 (rule-consistency) is what separates "learned the family" from
    "memorized this cipher": a chain that got the answer but whose committed
    rule fails an in-context example is a hallucination and is DROPPED, even
    though its final answer is correct.

Why this can work even though puzzles are "info-limited":
  Measured on 150 baseline-FAILED bit puzzles: 0% are fully bit-determined,
  mean of only ~1.1/8 output bits uniquely fixed by the examples — yet the
  prior-weighted voter still recovers 28% of them and the model recovers most
  globally. "Under-constrained" is a spectrum the family-prior partly closes.
  RFT harvests exactly the chains where the model's prior lands correctly AND
  stays example-consistent, then amplifies that prior. It does NOT need the
  puzzle to be uniquely solvable from examples alone.

================================================================================
INTERFACES WITH EXISTING CODE (no rewrites needed)
================================================================================
  training/train.py            SFT entrypoint. --resume_adapter_dir loads the
                               Huikang adapter; --csv_path takes our built CSV;
                               target_modules already includes Mamba in/out_proj.
  training/train_v3.sbatch     template for the SIGUSR1 chain-resume sbatch. The
                               RFT train step is a near-copy with csv_path and
                               OUTPUT_DIR swapped (see rft_train.sbatch scaffold).
  training/dataset.py          HuikangCoTDataset reads columns
                               id,prompt,answer,generated_cot,type — build_csv()
                               below emits exactly that. It re-wraps cot in
                               <think>..</think>\\boxed{answer} and masks prompt,
                               so generated_cot must be the RAW reasoning WITHOUT
                               the final \\boxed{} (dataset strips stray boxes and
                               appends the canonical one). build_csv() strips the
                               trailing boxed line for that reason.
  training/rl/rollout.py       PROBE phase. Reuse verbatim with --temperature>0,
                               --group_size=K; swap reward.score for keep-the-text
                               (we filter on CPU afterward, so even the stock
                               rollout JSONL is fine — it already stores group[i].text).
  training/rl/utils.py         build_user_content / apply_chat_template_for_user —
                               the exact prompt wrapper used at train+infer time.
  eval/eval_vllm.py            EVAL phase on holdout_split.jsonl.
  eval/extract.py              extract_final_answer + verify — gate-1, identical
                               to the grader.
  eval/category.py             detect_category — sanity-check eval's category tag.
  training/opd/solvers/*       VERIFIERS for gate-2 (never generators).

This module implements the CPU phases (FILTER + DEDUP + BUILD) end-to-end and is
runnable on the login node over a rollout JSONL. PROBE/TRAIN/EVAL are GPU/sbatch
phases; their orchestration lives in the `rft_round.sbatch` scaffold and the
`run_round()` docstring below.

GPU REQUIRED: yes — for PROBE (vLLM sampling) and TRAIN (FSDP SFT) and EVAL.
              The code in THIS file (filter/dedup/build) is CPU-only and can be
              dry-run on the login node against a small rollout JSONL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

PROJECT = Path("/scratch/jwj/NemotronFT")
sys.path.insert(0, str(PROJECT / "eval"))
sys.path.insert(0, str(PROJECT / "training" / "opd" / "solvers"))

from extract import extract_final_answer, verify  # type: ignore  # noqa: E402


# ============================================================================
# gate-2: rule-consistency verifiers (solver-as-verifier, per category)
# ============================================================================
# Each verifier answers ONE question: "does the in-context example set, under
# the family rule, actually produce the example outputs?" We use the solvers
# only to PARSE the examples and to RE-DERIVE outputs from a committed rule —
# never to solve the puzzle. A chain passes gate-2 if every example is
# reproduced by the structure the chain claims to have used.
#
# Pragmatic note on the cipher/bit families: the model's CoT does not expose a
# machine-parseable "rule" object, and re-parsing free-text reasoning is
# brittle. So gate-2 is implemented at two strengths, selectable per category:
#
#   STRONG  (cryptarithm_deduce, equation_numeric_deduce): the examples THEMSELVES
#           constrain a small combo/cipher space. We run the answer-guided
#           solver/cot-builder (cipher_cot_builder.build_cot) which, GIVEN the
#           known correct answer, searches for a (combo, cipher) consistent with
#           ALL same-op examples. If it finds one, the puzzle is provably
#           example-consistent at the claimed answer -> gate-2 PASS, and we even
#           get a clean structured CoT we can prefer over the model's. If it
#           finds none, the answer is example-inconsistent under the known
#           family -> the model's right answer is luck/hallucination -> DROP.
#
#   WEAK    (bit_manipulation, *_guess): no tractable answer-guided consistency
#           oracle (bit is 0% fully-determined; guess families have no same-op
#           anchor). gate-2 degrades to: (a) gate-1 must pass, and (b) a
#           structural sanity check on the chain (answer well-formed for the
#           category: 8-bit string for bit; right length/charset for others).
#           For these, DEDUP-for-distinctness does the heavy lifting and the
#           positive-transfer holdout gate is the real safety net.
#
# This split is deliberate and is the project's hard-won lesson encoded as code:
# we trust the solver as a *consistency oracle* exactly where it is sound, and
# we fall back to answer+holdout everywhere else, rather than letting a weak
# solver veto (or worse, generate) chains it cannot judge.


def _verify_strong_cipher(prompt: str, gt: str) -> Optional[bool]:
    """STRONG gate-2 for cryptarithm/symbolic families.

    Returns True  if a (combo, cipher) consistent with ALL same-op examples
                   reproduces the known answer `gt`,
            False if no such structure exists (answer is example-inconsistent),
            None  if the verifier could not parse the prompt (abstain -> caller
                   falls back to WEAK).
    """
    try:
        from cipher_cot_builder import build_cot  # type: ignore
    except Exception:
        return None
    try:
        res = build_cot(prompt, gt)
    except Exception:
        return None
    if res is None:
        return None
    if res.cot_trace.startswith("(parse") or res.cot_trace.startswith("(question") \
            or res.cot_trace.startswith("(no same-op"):
        return None  # parse failure -> abstain, not a definitive False
    return bool(res.solved)


def _structural_ok(category: str, ans: str) -> bool:
    """WEAK gate-2: answer is well-formed for its category (cheap sanity)."""
    ans = ans.strip()
    if not ans:
        return False
    if category == "bit_manipulation":
        return bool(re.fullmatch(r"[01]{8}", ans))
    if category in ("equation_numeric_deduce", "equation_numeric_guess"):
        return bool(re.fullmatch(r"-?\d+", ans))
    # cryptarithm_*: cipher symbols, just require non-empty single token
    return len(ans.split()) == 1


# Which categories get the STRONG oracle. (cipher_cot_builder anchors on
# same-op examples, present in *_deduce; *_guess has no same-op anchor.)
STRONG_CATS = {"cryptarithm_deduce", "equation_numeric_deduce"}


def gate2_rule_consistent(prompt: str, gt: str, category: str, pred_answer: str) -> tuple[bool, str]:
    """The rule-consistency gate. Returns (passed, reason)."""
    if not _structural_ok(category, pred_answer):
        return False, "ill-formed-answer"
    if category in STRONG_CATS:
        verdict = _verify_strong_cipher(prompt, gt)
        if verdict is True:
            return True, "strong-consistent"
        if verdict is False:
            return False, "strong-inconsistent"  # answer not reproducible -> hallucination
        # verdict is None -> oracle abstained, fall through to weak pass
        return True, "strong-abstain->weak"
    return True, "weak-structural"


# ============================================================================
# DEDUP: distinct-reasoning-path signature
# ============================================================================

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")


def cot_signature(text: str) -> str:
    """Structural fingerprint of a reasoning chain for distinctness dedup.

    We want chains that differ in *reasoning structure*, not in incidental
    numbers/whitespace. Normalize: strip the final boxed answer, collapse
    whitespace, mask all digit runs (so "A->3" and "A->7" don't count as
    distinct paths), then hash. Two chains with the same signature are treated
    as the same path and only one is kept.
    """
    body = re.sub(r"\\boxed\{[^}]*\}", "", text)
    body = _NUM.sub("#", body)
    body = _WS.sub(" ", body).strip().lower()
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def strip_trailing_boxed(text: str) -> str:
    """Remove the final \\boxed{...} (and a trailing </think>) so the CSV's
    generated_cot is the RAW reasoning. HuikangCoTDataset re-wraps it in
    <think>..</think>\\boxed{answer}, so leaving a box in would double it."""
    text = re.sub(r"\\boxed\{[^}]*\}\s*$", "", text.rstrip())
    text = re.sub(r"</think>\s*$", "", text.rstrip())
    text = re.sub(r"<think>\s*", "", text.lstrip())  # dataset re-adds the open tag
    return text.strip()


# ============================================================================
# FILTER + DEDUP over a rollout JSONL
# ============================================================================

def iter_rollout(path: Path):
    """Yield (rec, completion_text) over a rollout.py-format JSONL.

    rollout.py writes: {prompt_id, category, prompt, gt, group:[{text,...}]}.
    Also accepts a flatter probe format: {id, category, prompt, gt, completions:[...]}.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            comps = []
            if "group" in rec:
                comps = [g.get("text", "") for g in rec["group"]]
            elif "completions" in rec:
                comps = rec["completions"]
            elif "text" in rec:
                comps = [rec["text"]]
            base = {
                "id": rec.get("prompt_id") or rec.get("id"),
                "category": rec["category"],
                "prompt": rec["prompt"],
                "gt": str(rec["gt"]),
            }
            for t in comps:
                yield base, t


def filter_and_dedup(rollout_path: Path, max_keep_per_id: int, prefer_solver_cot: bool):
    """Apply both gates + distinctness dedup. Returns (kept_rows, stats)."""
    kept_by_id: dict[str, list[dict]] = defaultdict(list)
    sigs_by_id: dict[str, set[str]] = defaultdict(set)
    stats = Counter()
    drop_reason = Counter()
    seen_completions = 0

    for base, text in iter_rollout(rollout_path):
        seen_completions += 1
        pid = base["id"]
        cat = base["category"]
        gt = base["gt"]

        # gate-1: answer correct (grader-identical)
        pred = extract_final_answer(text)
        if not verify(gt, pred):
            drop_reason["gate1-wrong-answer"] += 1
            continue

        # gate-2: rule consistent with all in-context examples
        ok, reason = gate2_rule_consistent(base["prompt"], gt, cat, pred)
        if not ok:
            drop_reason[f"gate2-{reason}"] += 1
            continue

        # distinctness: skip near-identical reasoning
        sig = cot_signature(text)
        if sig in sigs_by_id[pid]:
            drop_reason["dup-path"] += 1
            continue
        if len(kept_by_id[pid]) >= max_keep_per_id:
            drop_reason["over-cap"] += 1
            continue

        cot = strip_trailing_boxed(text)

        # Optionally prefer the solver's clean structured CoT over the model's
        # free-text, when STRONG gate-2 produced one (cryptarithm/symbolic).
        if prefer_solver_cot and cat in STRONG_CATS:
            solver_cot = _solver_cot(base["prompt"], gt)
            if solver_cot:
                cot = strip_trailing_boxed(solver_cot)
                stats["used-solver-cot"] += 1

        if len(cot) < 20:
            drop_reason["empty-cot"] += 1
            continue

        sigs_by_id[pid].add(sig)
        kept_by_id[pid].append({
            "id": pid, "prompt": base["prompt"], "answer": gt,
            "generated_cot": cot, "type": cat, "gate2_reason": reason,
        })
        stats["kept"] += 1
        stats[f"kept-{cat}"] += 1

    kept_rows = [r for rows in kept_by_id.values() for r in rows]
    stats["seen_completions"] = seen_completions
    stats["distinct_ids_with_keep"] = len(kept_by_id)
    return kept_rows, stats, drop_reason


def _solver_cot(prompt: str, gt: str) -> Optional[str]:
    try:
        from cipher_cot_builder import build_cot  # type: ignore
        res = build_cot(prompt, gt)
        return res.cot_trace if (res and res.solved) else None
    except Exception:
        return None


# ============================================================================
# BUILD: emit HuikangCoTDataset-compatible CSV
# ============================================================================

def build_csv(rows: list[dict], out_csv: Path, mix_csv: Optional[Path],
              mix_keep_cats: Optional[set[str]]) -> None:
    """Write id,prompt,answer,generated_cot,type. Optionally blend in rows from
    an existing CoT CSV (e.g. the high-acc categories from train_cot_v3.csv) so
    the SFT does not catastrophically forget the >=99% categories while we push
    the hard ones. mix_keep_cats restricts which `type`s to carry over."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "prompt", "answer", "generated_cot", "type"]
    n_new = 0
    n_mix = 0
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
            n_new += 1
        if mix_csv and mix_csv.exists():
            with open(mix_csv, newline="") as mf:
                for row in csv.DictReader(mf):
                    if mix_keep_cats and row.get("type") not in mix_keep_cats:
                        continue
                    w.writerow({k: row.get(k, "") for k in fieldnames})
                    n_mix += 1
    print(f"[build_csv] wrote {out_csv}  ({n_new} RFT rows + {n_mix} mixed-in rows)")


# ============================================================================
# CLI: the CPU half of one RFT round (filter -> dedup -> build CSV)
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", required=True, type=Path,
                    help="rollout.py-format JSONL from the PROBE phase")
    ap.add_argument("--out_csv", required=True, type=Path,
                    help="output CoT CSV (HuikangCoTDataset schema)")
    ap.add_argument("--max_keep_per_id", type=int, default=4,
                    help="cap on distinct correct chains per puzzle (distinctness > count)")
    ap.add_argument("--prefer_solver_cot", action="store_true",
                    help="for STRONG categories, replace model CoT with the clean "
                         "answer-guided solver CoT when available")
    ap.add_argument("--mix_csv", type=Path, default=None,
                    help="existing CoT CSV to blend in (anti-forgetting)")
    ap.add_argument("--mix_keep_cats", nargs="*", default=None,
                    help="restrict mixed-in rows to these `type`s "
                         "(e.g. cipher gravity numeral unit_conversion)")
    ap.add_argument("--stats_out", type=Path, default=None,
                    help="optional JSON dump of filter stats")
    args = ap.parse_args()

    rows, stats, drop_reason = filter_and_dedup(
        args.rollout, args.max_keep_per_id, args.prefer_solver_cot,
    )
    print("\n[rft filter] stats:")
    for k in sorted(stats):
        print(f"  {k:32s} {stats[k]}")
    print("[rft filter] drops:")
    for k in sorted(drop_reason):
        print(f"  {k:32s} {drop_reason[k]}")

    by_cat = Counter(r["type"] for r in rows)
    print("\n[rft filter] kept distinct chains by category:")
    for c in sorted(by_cat):
        print(f"  {c:30s} {by_cat[c]}")

    build_csv(rows, args.out_csv, args.mix_csv,
              set(args.mix_keep_cats) if args.mix_keep_cats else None)

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_out, "w") as f:
            json.dump({"stats": dict(stats), "drops": dict(drop_reason),
                       "by_cat": dict(by_cat)}, f, indent=2)


# ============================================================================
# run_round(): orchestration reference (the GPU phases live in sbatch)
# ============================================================================
def run_round() -> None:
    """REFERENCE ONLY — describes the full round; the GPU steps are sbatch jobs.

    Round r, given adapter A_r (A_0 = output/huikang-baseline-converted):

      0. make_holdout.py  (once, round 0 only)  ->  runs/holdout/v1/{train,holdout}_split.jsonl
      1. PROBE  [GPU, venv-vllm, sbatch]:
           python training/rl/rollout.py \
             --pool runs/holdout/v1/train_split.jsonl \
             --adapter_dir <A_r> \
             --out runs/rft/r{r}/rollout.jsonl \
             --group_size 16 --temperature 1.0 --top_p 0.95
         (G=16/T=1.0 per the diversity lesson; rollout.py omits SamplingParams.seed.)
      2. FILTER+BUILD  [CPU, this file]:
           python training/opd/rft_pipeline.py \
             --rollout runs/rft/r{r}/rollout.jsonl \
             --out_csv runs/rft/r{r}/train_cot_rft.csv \
             --max_keep_per_id 4 --prefer_solver_cot \
             --mix_csv training/train_cot_v3.csv \
             --mix_keep_cats cipher gravity numeral unit_conversion
      3. TRAIN  [GPU, venv, sbatch rft_train.sbatch]:
           train.py --resume_adapter_dir <A_r> --csv_path .../train_cot_rft.csv
           -> A_{r+1} = output/rft-r{r}
      4. EVAL  [GPU, venv-vllm, sbatch]:
           eval/eval_vllm.py --adapter output/rft-r{r}
             --pool runs/holdout/v1/holdout_split.jsonl
         TRANSFER GATE: accept A_{r+1} iff holdout acc(A_{r+1}) > acc(A_r) and
                        no target category regresses. Else discard, change
                        sampling (more distinct paths), retry.
      5. If accepted, A_{r+1} becomes the prober for round r+1. Stop when the
         marginal distinct-correct-path yield plateaus.
    """
    raise SystemExit("run_round() is a reference docstring; run the phases via sbatch.")


if __name__ == "__main__":
    main()
