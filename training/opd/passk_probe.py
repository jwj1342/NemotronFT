"""
pass@k probe — empirically measure the "recoverable space" of the Huikang
baseline adapter on stratified FAILED problems.

Motivation (see CLAUDE.md / ITERATION_REPORT.md):
  The baseline LB is stuck at 0.85. The dominant misses are cryptarithm_deduce
  (613), bit_manipulation (200), cryptarithm_guess (154), equation_numeric_guess
  (114). The strategic question is: of the problems the grader (T=0 greedy)
  gets WRONG, how many can the *same adapter* solve at least once when sampled
  at T>0 with n trials?  That fraction is the upper bound on what answer-filtered
  self-training (RFT/STaR) could recover without any new generator.

This script:
  1. Reads the baseline eval shards, keeps only correct==false rows in the
     four target categories, and takes a stratified random sample per category.
  2. Loads the prompt text for each sampled id from data/train.csv (stdlib csv).
  3. Loads model + Huikang LoRA adapter under vLLM (same load path as
     training/rl/rollout.py), applies the exact grader chat template
     (enable_thinking=True + BOXED_INSTRUCTION).
  4. Samples n completions per prompt at T>0 (NO SamplingParams.seed — that
     kills diversity on vllm 0.17.1; see CLAUDE.md / rollout.py).
  5. Scores every completion with the grader-aligned verify() from eval/extract.py.
     Records per problem:
       - pass@k        : >=1 completion answer-correct
       - n_correct     : how many of the n are correct
       - n_distinct_correct : distinct correct extracted answers (should be 1
                              for well-posed; >1 means the GT itself is ambiguous
                              under grader tolerance, rare)
       - n_distinct_any: distinct extracted answers across all n (diversity proxy)
       - solver_verifier (best-effort): does the programmatic solver uniquely
         constrain this puzzle from its examples, and does its answer agree with
         the ground-truth answer?  This separates "information-limited /
         under-constrained" (solver can't pin it down) from "capability-limited"
         (solver CAN pin it down, so the unique answer is derivable — the model
         just didn't, at T=0).
  6. Writes per-completion-aware JSONL + a per-problem JSONL + a per-category
     summary CSV under training/opd/runs/passk/.

Runs in venv-vllm. TP=4 within one node. NOT a training script — read-only on
the adapter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # /scratch/jwj/NemotronFT
MODEL_PATH = str(PROJECT_ROOT / "model")
DEFAULT_ADAPTER = str(PROJECT_ROOT / "output" / "huikang-baseline-converted")
DEFAULT_EVAL_GLOB = "output/eval_full_huikang-baseline-converted_shard{i}of4.jsonl"
DEFAULT_TRAIN_CSV = str(PROJECT_ROOT / "data" / "train.csv")
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "training" / "opd" / "runs" / "passk")

# Categories of interest (eval `category` field is authoritative).
DEFAULT_CATEGORIES = [
    "bit_manipulation",
    "equation_numeric_guess",
    "cryptarithm_deduce",
    "cryptarithm_guess",
]

# --- grader-aligned answer extraction / verification -------------------------
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from extract import BOXED_INSTRUCTION, extract_final_answer, verify  # type: ignore  # noqa: E402

# --- best-effort programmatic verifiers (solvers) ----------------------------
# These are only used as a *constrainability* signal, never as generators.
# Any solver failure must be caught and treated as "unknown" — it must never
# abort the probe.
sys.path.insert(0, str(PROJECT_ROOT / "training" / "opd" / "solvers"))


def solver_verdict(category: str, prompt: str, gt: str) -> dict:
    """Best-effort: does a programmatic solver UNIQUELY constrain this puzzle
    from its examples, and does that unique answer agree with the ground truth?

    Returns a dict:
      {"solver": <name|None>,
       "solved": bool,            # solver found a unique answer
       "agrees": bool|None,       # solver answer == GT (grader verify); None if unsolved
       "answer": str|None,
       "confidence": float|None}  # bit solver only

    Interpretation:
      solved & agrees   -> well-constrained, unique answer derivable
                           => model miss is CAPABILITY-limited (recoverable target).
      solved & not agrees-> solver disagrees with GT (solver wrong family or GT
                           anomaly) — inconclusive, lean capability.
      not solved        -> solver can't pin a unique answer from examples
                           => lean INFORMATION-limited / under-constrained.
    NB: this is a heuristic. The solvers are known weaker than the model, so a
    "not solved" does NOT prove the puzzle is truly ambiguous (the model may use
    priors the solver can't). pass@k is the ground truth; this column is a hint.
    """
    out = {"solver": None, "solved": False, "agrees": None,
           "answer": None, "confidence": None}
    try:
        if category == "bit_manipulation":
            from bit_solver_v2 import solve_bit_v2  # type: ignore
            r = solve_bit_v2(prompt)
            out["solver"] = "bit_solver_v2"
            out["solved"] = bool(getattr(r, "solved", False))
            out["answer"] = getattr(r, "answer", None)
            out["confidence"] = float(getattr(r, "confidence", 0.0) or 0.0)
        elif category in ("cryptarithm_deduce", "cryptarithm_guess"):
            from symbol_digit_solver import solve_cipher_digit  # type: ignore
            r = solve_cipher_digit(prompt)
            out["solver"] = "symbol_digit_solver"
            out["solved"] = bool(getattr(r, "solved", False))
            out["answer"] = getattr(r, "answer", None)
        elif category in ("equation_numeric_deduce", "equation_numeric_guess"):
            # Numeric: parse prompt then solve_symbol_digit on same-op examples.
            from symbol_digit_solver import (  # type: ignore
                parse_problem_prompt, solve_symbol_digit,
            )
            examples, question = parse_problem_prompt(prompt)
            out["solver"] = "symbol_digit_solver"
            if examples and question:
                r = solve_symbol_digit(examples, question)
                out["solved"] = bool(getattr(r, "solved", False))
                out["answer"] = getattr(r, "answer", None)
        if out["solved"] and out["answer"] is not None:
            out["agrees"] = bool(verify(str(gt), str(out["answer"])))
    except Exception as e:  # never let a solver crash kill the probe
        out["solver_error"] = f"{type(e).__name__}: {e}"
    return out


# --- data loading ------------------------------------------------------------

def load_failed_ids(eval_glob_tmpl: str, categories: set[str]) -> dict[str, list[dict]]:
    """Return {category: [ {id, answer}, ... ]} for correct==false rows."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for i in range(4):
        fn = PROJECT_ROOT / eval_glob_tmpl.format(i=i)
        if not fn.exists():
            print(f"[load] WARNING: missing eval shard {fn}")
            continue
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("correct"):
                    continue
                if r["category"] in categories:
                    by_cat[r["category"]].append(
                        {"id": r["id"], "answer": str(r["answer"]),
                         "num_tokens": r.get("num_tokens")}
                    )
    return by_cat


def load_prompts(train_csv: str, wanted_ids: set[str]) -> dict[str, str]:
    csv.field_size_limit(sys.maxsize)
    prompts: dict[str, str] = {}
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] in wanted_ids:
                prompts[row["id"]] = row["prompt"]
    return prompts


# --- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter_dir", default=DEFAULT_ADAPTER, type=Path)
    ap.add_argument("--eval_glob", default=DEFAULT_EVAL_GLOB,
                    help="Template with {i} for shard index (relative to project root).")
    ap.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV, type=Path)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, type=Path)
    ap.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                    help="Comma-separated eval category names to probe.")
    ap.add_argument("--per_cat", type=int, default=60,
                    help="Stratified sample size of FAILED problems per category.")
    ap.add_argument("--n", type=int, default=16, help="Completions per problem (pass@n).")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1, help="-1 disables top_k.")
    ap.add_argument("--max_tokens", type=int, default=7680, help="Grader budget.")
    ap.add_argument("--max_model_len", type=int, default=8192, help="Grader budget.")
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Seed for the PROBLEM SAMPLER ONLY (python random). "
                         "Does NOT touch vLLM SamplingParams (which must stay seedless).")
    ap.add_argument("--save_texts", action="store_true",
                    help="Also dump full completion texts (large). Default off; "
                         "only correct completions' text is saved regardless.")
    ap.add_argument("--no_solver", action="store_true",
                    help="Skip the programmatic solver verifier column.")
    ap.add_argument("--tag", default="", help="Optional suffix for output filenames.")
    args = ap.parse_args()

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"_{args.tag}" if args.tag else "")

    # --- 1. stratified sample of failed problems -----------------------------
    print(f"[probe] loading failed ids for {categories} ...")
    by_cat = load_failed_ids(args.eval_glob, set(categories))
    rng = random.Random(args.sample_seed)
    sampled: list[dict] = []
    for cat in categories:
        rows = by_cat.get(cat, [])
        rng.shuffle(rows)
        take = rows[: args.per_cat]
        for r in take:
            r["category"] = cat
        sampled.extend(take)
        print(f"  {cat:28s} failed={len(rows):4d}  sampled={len(take)}")
    if not sampled:
        print("[probe] no problems sampled — nothing to do.")
        return

    # --- 2. load prompts ------------------------------------------------------
    wanted = {r["id"] for r in sampled}
    print(f"[probe] loading {len(wanted)} prompts from {args.train_csv} ...")
    prompts = load_prompts(str(args.train_csv), wanted)
    missing = [r["id"] for r in sampled if r["id"] not in prompts]
    if missing:
        print(f"[probe] WARNING: {len(missing)} sampled ids have no prompt "
              f"in train.csv (dropping): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    sampled = [r for r in sampled if r["id"] in prompts]
    print(f"[probe] {len(sampled)} problems with prompts.")

    # --- 3. load model + adapter (same path as training/rl/rollout.py) --------
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[vllm] loading model {MODEL_PATH}  +  adapter {args.adapter_dir} ...")
    t_load = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"[vllm] loaded in {time.time() - t_load:.1f}s")

    # --- 4. build prompts with the exact grader template ----------------------
    prompt_texts: list[str] = []
    for r in sampled:
        user_content = prompts[r["id"]] + BOXED_INSTRUCTION
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_texts.append(text)

    # NOTE: deliberately NO seed on SamplingParams (vllm 0.17.1 collapses
    # n>1 diversity with a fixed seed). top_k only passed if >=1.
    sp_kwargs = dict(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    if args.top_k is not None and args.top_k >= 1:
        sp_kwargs["top_k"] = args.top_k
    sampling = SamplingParams(**sp_kwargs)
    lora = LoRARequest("adapter", 1, str(args.adapter_dir))

    print(f"[generate] {len(prompt_texts)} prompts × n={args.n} "
          f"@ T={args.temperature} top_p={args.top_p} top_k={args.top_k} ...")
    t_gen = time.time()
    outputs = llm.generate(prompt_texts, sampling, lora_request=lora)
    gen_elapsed = time.time() - t_gen
    print(f"[generate] {gen_elapsed:.1f}s "
          f"({gen_elapsed / max(1, len(prompt_texts)):.2f}s/prompt)")

    # --- 5. score + aggregate -------------------------------------------------
    per_problem: list[dict] = []
    cat_stats: dict[str, dict] = defaultdict(lambda: {
        "n_probs": 0, "n_passk": 0,
        "n_solver_solved": 0, "n_solver_agrees": 0,
        "sum_pass1": 0.0, "sum_distinct_any": 0,
    })

    for r, out in zip(sampled, outputs):
        gt = r["answer"]
        extracted = [extract_final_answer(c.text) for c in out.outputs]
        correct_flags = [verify(gt, e) for e in extracted]
        n_correct = sum(1 for f in correct_flags if f)
        correct_answers = [e for e, f in zip(extracted, correct_flags) if f]
        distinct_correct = sorted(set(a.strip() for a in correct_answers))
        distinct_any = Counter(e.strip() for e in extracted)
        passk = n_correct >= 1

        sv = ({} if args.no_solver
              else solver_verdict(r["category"], prompts[r["id"]], gt))

        rec = {
            "id": r["id"],
            "category": r["category"],
            "gt": gt,
            "baseline_num_tokens": r.get("num_tokens"),
            "n": args.n,
            "pass_k": passk,
            "n_correct": n_correct,
            "pass_at_1_rate": n_correct / args.n,
            "n_distinct_correct": len(distinct_correct),
            "distinct_correct": distinct_correct,
            "n_distinct_any": len(distinct_any),
            "top_answers": distinct_any.most_common(5),
            "completion_lens": [len(c.token_ids) for c in out.outputs],
            "solver": sv,
        }
        if args.save_texts:
            rec["completions"] = [
                {"text": c.text, "extracted": e, "correct": f}
                for c, e, f in zip(out.outputs, extracted, correct_flags)
            ]
        else:
            # always keep at most a couple of *correct* CoT for inspection
            rec["sample_correct_cot"] = [
                c.text for c, f in zip(out.outputs, correct_flags) if f
            ][:2]
        per_problem.append(rec)

        cs = cat_stats[r["category"]]
        cs["n_probs"] += 1
        cs["n_passk"] += int(passk)
        cs["sum_pass1"] += n_correct / args.n
        cs["sum_distinct_any"] += len(distinct_any)
        if sv.get("solved"):
            cs["n_solver_solved"] += 1
            if sv.get("agrees"):
                cs["n_solver_agrees"] += 1

    # --- 6. write outputs -----------------------------------------------------
    pp_path = args.out_dir / f"passk_per_problem{tag}.jsonl"
    with open(pp_path, "w") as f:
        for rec in per_problem:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = args.out_dir / f"passk_summary{tag}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "category", "n_probs", "passk_count", "passk_rate",
            "mean_pass_at_1", "mean_distinct_any",
            "solver_solved", "solver_agrees_gt",
        ])
        for cat in categories:
            cs = cat_stats.get(cat)
            if not cs or cs["n_probs"] == 0:
                continue
            np_ = cs["n_probs"]
            w.writerow([
                cat, np_, cs["n_passk"], f"{cs['n_passk'] / np_:.4f}",
                f"{cs['sum_pass1'] / np_:.4f}",
                f"{cs['sum_distinct_any'] / np_:.3f}",
                cs["n_solver_solved"], cs["n_solver_agrees"],
            ])

    # config echo for reproducibility
    cfg_path = args.out_dir / f"passk_config{tag}.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "adapter_dir": str(args.adapter_dir),
            "categories": categories,
            "per_cat": args.per_cat,
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "sample_seed": args.sample_seed,
            "n_problems": len(per_problem),
        }, f, indent=2)

    # --- 7. console summary ---------------------------------------------------
    print()
    print("=" * 78)
    print(f"pass@{args.n} PROBE SUMMARY  (T={args.temperature}, top_p={args.top_p}, "
          f"top_k={args.top_k}, adapter={Path(args.adapter_dir).name})")
    print("-" * 78)
    print(f"{'category':28s} {'probs':>5s} {'pass@k':>7s} {'pass@1':>7s} "
          f"{'distinct':>8s} {'slv_ok':>6s} {'slv=GT':>6s}")
    tot_probs = tot_passk = 0
    for cat in categories:
        cs = cat_stats.get(cat)
        if not cs or cs["n_probs"] == 0:
            continue
        np_ = cs["n_probs"]
        tot_probs += np_
        tot_passk += cs["n_passk"]
        print(f"{cat:28s} {np_:5d} {cs['n_passk'] / np_:7.3f} "
              f"{cs['sum_pass1'] / np_:7.3f} {cs['sum_distinct_any'] / np_:8.2f} "
              f"{cs['n_solver_solved']:6d} {cs['n_solver_agrees']:6d}")
    if tot_probs:
        print("-" * 78)
        print(f"{'ALL':28s} {tot_probs:5d} {tot_passk / tot_probs:7.3f}")
    print("=" * 78)
    print(f"[probe] per-problem  -> {pp_path}")
    print(f"[probe] summary csv  -> {summary_path}")
    print(f"[probe] config       -> {cfg_path}")
    print()
    print("READING THE RESULT:")
    print("  pass@k high  -> the answer-correct CoT EXISTS in the adapter's")
    print("                  sampling distribution; RFT/STaR can mine it. (recoverable)")
    print("  pass@k ~0    -> the adapter never produces the right answer even at")
    print("                  T>0×n; self-training has nothing to filter. (hard)")
    print("  slv_ok≈probs & low pass@k -> well-constrained but model can't get it")
    print("                  => capability gap (good RL/SFT target).")
    print("  slv_ok≈0    -> solver can't constrain from examples; lean")
    print("                  information-limited, but verify with pass@k.")


if __name__ == "__main__":
    main()
