# What Actually Moves the Needle on Nemotron-Nano-30B Reasoning — and What Provably Doesn't

*An Open-Contribution writeup for the NVIDIA Nemotron Model Reasoning Challenge.*
*Target model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (32B MoE, hybrid Mamba–Transformer, `nemotron_h`).*
*Grader: fixed LoRA adapter, single greedy (T=0) pass, `max_tokens=7680`, `max_model_len=8192`, `max_lora_rank=32`.*

> **DRAFT — autonomously assembled 2026-06-13 from the project's experiment log
> (`ITERATION_REPORT.md`, `ROADMAP.md`) and the team memory. Numbers are
> grader-aligned (T=0) local evals on the 9500-problem train set or its shard-0
> (2375 problems), cross-checked against public-LB scores where noted. Pending
> the team's review before posting.**

## TL;DR for the next competitor

If you are forking the public 0.85/0.86 adapter and wondering why you can't push past it, here is the compressed map of the terrain, paid for in GPU-weeks of negative results:

1. **The 0.85 → 0.86 gap is a *packaging* problem, not a modeling problem.** It is caused by (a) lossy adapter conversion and (b) a LoRA key-naming mismatch — both fixable without touching the weights. Section 1.
2. **The 0.86 → 0.87+ gap is a *capability* problem you cannot fine-tune around with the available supervision.** Every training route — SFT distillation, hand-written-solver→CoT, RFT, STaR, and GRPO — nets ≤ 0 on the leaderboard. Sections 2–4.
3. **The reason is two walls**: a *prior-coupling* wall on the categories the model is already good at (improving the hard slice regresses the larger correct set), and a *capability/unlearnability* wall on cryptarithm (the correct solutions are not in the model's sample support at any temperature, so there is no gradient to learn from). Sections 2–3.
4. **Your local proxy will lie to you.** We document a holdout metric that is *anti-correlated* with the leaderboard, and a methodology rule that survived a three-way swing of contradictory evidence. Section 4.

---

## 1. The 0.85 → 0.86 gap is conversion loss + a key-naming bug

The strongest public adapter is Huikang's winning Tinker adapter. Getting it to score on this competition's grader requires converting it to a PEFT LoRA that loads against the `nemotron_h` modules. Two independent traps cost ~1pp each:

### 1a. SVD conversion is lossy by ~25%
Hand-rolling the Tinker→PEFT conversion (SVD on the Mamba `gate_proj` + `x_proj` to synthesize `in_proj`) **discards ~25% of the singular mass** and lands at **LB 0.85**. The *lossless* conversion via `tinker_cookbook.build_lora_adapter` preserves it and lands at **LB 0.86**. Same adapter, same weights conceptually — the difference is entirely in how faithfully the Mamba projections are reconstructed.

### 1b. The LoRA key prefix decides whether the grader loads your adapter at all
The lossless adapter ships with keys prefixed `base_model.model.model.layers...`. The competition's base model (and grader) expose the Mamba stack under `backbone.layers...`. If you submit with the original prefix, PEFT silently fails to attach the LoRA to the Mamba modules → the grader runs **≈ the base model** → **LB 0.85** (and, if you try to *train* from it, the KL-to-reference term explodes because policy ≈ base while the reference is the intended adapter).

**Fix:** rename `base_model.model.model.` → `base_model.model.backbone.`. The renamed adapter (a) loads correctly → **LB 0.86**, and (b) casts cleanly to bf16 at **1.78 GB**, which is **under the ~2 GB file-upload limit** — so you can submit 0.86 by *file upload*, no notebook-submission gymnastics required. There is **no "bf16 ceiling"**: the earlier belief that bf16 capped us at 0.85 was this key-naming bug, not precision loss.

> Practical takeaway: before concluding "my trained adapter regressed," verify your LoRA keys actually match the base module names. A mis-prefixed adapter and an untrained base score identically — a silent, expensive failure mode.

---

## 2. The prior-coupling wall (why "fix the weak slice" backfires)

The model's per-category accuracy is bimodal:

| Category | Base acc (9500, T=0) | Headroom |
|---|---|---|
| numeral | 100.0% | none |
| unit_conversion | 99.8% | none |
| gravity | 99.6% | none |
| cipher | 99.0% | none |
| equation_numeric_deduce | 90.9% | small |
| bit_manipulation | 87.5% | small |
| equation_numeric_guess | 16.2% | **cliff** |
| cryptarithm_guess | 6.1% | **cliff** |
| cryptarithm_deduce | 7.0% | **cliff** |

All the apparent headroom is in three "cliff" categories. The intuitive move — fine-tune to lift them — runs into **prior-coupling**: the cliff categories and the saturated ones share low-level priors (number formatting, disambiguation of under-specified constraints, search-vs-recall heuristics). Pulling on the hard slice drifts those shared priors and **regresses the much larger already-correct set**.

We measured this on the leaderboard, not just locally:

- **Bit-only GRPO progressively regresses**: starting from the true-0.86 base, LB went **0.86 → 0.85 (iter 1) → 0.85 (iter 3) → 0.84 (iter 5)** as it iterated, *while* the 100-problem bit-holdout it was optimizing **climbed** (0.59 → 0.61 → 0.62). The proxy you optimize and the metric you care about moved in **opposite directions** (more on this in §4).
- **Multi-category GRPO holds 0.86**: including the saturated categories with a KL anchor (so the policy can't over-optimize one slice) keeps LB at **0.86** with no regression — strictly better than bit-only's 0.84–0.85 — but cannot *exceed* 0.86, because the gradient-bearing categories have no learnable headroom left.
- **STaR on the under-constrained bit slice regresses the target category**: sampling the model's own correct chains-of-thought on few-example ("under-constrained") bit problems and SFT-ing on them (a pure self-distillation, near-zero training loss) moved shard-0 bit from **89.6% → 87.8%** (−1.8pp, outside the ±0.3pp eval-noise floor) with overall flat — the same regression signature as bit-only GRPO. **The local signature held on the leaderboard: LB 0.85 (regressed from 0.86).** Self-distillation on a narrow uncertain slice drifts the shared prior just as supervised distillation and RL do.

**Conclusion:** single-slice optimization is net-negative; the best a training run achieves here is *net-neutral* (hold 0.86), obtained only by training broadly enough — or anchoring tightly enough — that the shared prior doesn't drift.

---

## 3. Cryptarithm is unlearnable with the available supervision (a capability wall)

The single largest headroom is `cryptarithm_deduce` (≈6.5pp of LB if solved). It is also the most thoroughly falsified dead-end of the project. Cryptarithm is *not* a missing-supervision problem you can SFT your way out of — it's an execution/search capability the model does not have and cannot acquire from any signal we can generate:

- **No gradient at any temperature.** pass@16 on well-posed cryptarithm holdout is **flat at ~10% across T = 1.0 / 1.3 / 1.6** (and pass@1 *drops* with temperature). The correct letter→digit cipher assignments are simply **not in the policy's sample support** at any temperature → zero positive samples → GRPO / RFT / STaR have **no gradient** to learn from. This is a stronger unlearnability result than a single-temperature pass@16 = 0.
- **SFT on correct solver-generated CoT is flat.** Training from the true-0.86 base on 4226 cipher-search traces (median 629 tokens, train loss 43 → 7) leaves well-posed cryptarithm holdout **flat at 9–10% = the base's own rate**. The model fits the *format* of a search trace but cannot *execute* the search on a novel cipher.
- **Hand-written solvers are weaker than the model**, so "solver → CoT → SFT" can only mis-teach. Even with the correct combinatorial framework, the solvers score 59.9% (bit) / 39.8% (numeric_deduce) / ~4% (cipher-digit) versus the model's 87.5 / 90.9 / ~7%. Cross-tabulated against the baseline, training on solver output would teach +58 problems but mis-teach −500 on bit. **Solvers are useful only as verifiers, never as generators.**
- **The dominant failure cluster is structurally out of reach**: of 643 unsolved cryptarithm, 84% are digit-cipher with a harder operation; a 21-operator DSL solver gets 0/100 on them. Wrong solver *family*, not a missing operator.

Each puzzle is an independent random cipher, so memorization does not transfer — a holdout *canary* (train on a split, eval the rest) showed **negative** transfer even as train loss collapsed 95 → 13.

**Conclusion:** the 6.5pp of cryptarithm headroom is a mirage under the current model + closed-grader regime. It would require a genuinely stronger base policy (none is publicly available above 0.86) or inference-time search (ruled out — see §5).

---

## 4. Methodology: the only ceiling estimate you can trust

This is the most transferable contribution. We were repeatedly and confidently *wrong* about what was salvageable, and each wrong estimate was corrected only by a stronger class of evidence:

| Estimator | Said about bit | Verdict |
|---|---|---|
| Static-solver coverage | "unsalvageable" | wrong |
| pass@k upper bound | "+0.2–0.7pp recoverable" | wrong (over-optimistic) |
| Holdout-*subset* accuracy | "checkpoint +3, looks good" | wrong (didn't extrapolate) |
| **Holdout net-gain extrapolated to the full set** | "net ≈ 0 / negative" | **correct (matched LB)** |

**The rule:** *salvageability and the ceiling can only be judged by **holdout net-gain (recovered − regressed − sentinel-drift) extrapolated to the full set.*** A checkpoint that recovers 8% of a failing slice but regresses 1.4% of a 10× larger correct slice is **net-negative**, even though every cheaper proxy says "improving."

The sharpest illustration: the bit-holdout we optimized with GRPO is **anti-correlated** with the leaderboard. Climbing the proxy 0.59 → 0.62 *cost* us LB 0.86 → 0.84. If you tune against a narrow proxy here, you will confidently walk downhill.

Corollary on eval noise: vLLM + LoRA multi-GPU decoding is **not bit-deterministic even at T=0** (~±0.3pp run-to-run on a 2375-problem shard). Trust only the leaderboard and full-set differences ≥ 1pp; sub-0.5pp shard differences are noise, and we burned time chasing some.

---

## 5. The grader is closed — the adapter is the only lever

Confirmed with the organizers: submission is a **fixed adapter evaluated in a single greedy (T=0) pass**. No multi-sampling, no best-of-N, no test-time training, no tool use. So every "inference-time search" idea (which is exactly what cryptarithm needs) is off the table. The only thing you control is the LoRA weights, and §§2–3 show those weights are already at the reachable ceiling for the public base.

Also load-bearing and easy to get wrong:
- **Train and eval at T=0**, not the `temperature=1.0` shown as a default in the public `metric.py` — the grader overrides it to greedy. A single token with <50% logprob derails the whole chain.
- **Mamba layers must be in `target_modules`** (`in_proj`, `out_proj`). Adapters that only target attention+MLP leave the Mamba reasoning circuits frozen and score 0.27–0.60.
- **Don't filter training data to short responses.** The model needs to learn long, complete reasoning that ends in `\boxed{}` within the 7680-token budget; truncating supervision is counterproductive.

---

## 6. What we would try with a stronger base or an open grader

Honesty about the one genuinely-untested edge: **KL-anchored RL with true iterative rollout** (resample fresh rollouts each round → ≤1–2 epochs → eval holdout → repeat, with β>0 KL-to-warmstart). Our bit-GRPO regressions were partly confounded by *staleness* (one rollout reused across 8 epochs at β=0), which conflates the prior-coupling wall with off-policy overfitting. A clean iterative loop *might* let the KL anchor control regression more gently than MLE-SFT. We judge the expected gain small (the prior-coupling wall is mechanism, not artifact) and the engineering cost high, but it is the only lever we did not fully exhaust. Everything else in the training toolbox is mapped to ≤ 0.

---

*Artifacts and full per-experiment numbers: `ITERATION_REPORT.md`, `ROADMAP.md`, `RL_ARCHITECTURE.md`. The file-submittable 0.86 adapter recipe (lossless convert + key rename + bf16 cast) is the durable, reproducible result of this work.*
