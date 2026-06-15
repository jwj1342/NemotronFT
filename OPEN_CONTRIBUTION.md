# What Actually Moves the Needle on Nemotron-Nano-30B Reasoning — and What Provably Doesn't

*An open-contribution writeup for the NVIDIA Nemotron Model Reasoning Challenge.*
*Target model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (32B MoE, hybrid Mamba–Transformer, `nemotron_h`).*
*Grader: a fixed LoRA adapter (`max_lora_rank=32`), evaluated in a single greedy (T=0) pass, `max_tokens=7680`, `max_model_len=8192`.*

### What this contributes

A large fraction of the leaderboard sits at exactly 0.86 — the score of the public
adapter. The interesting question is therefore not "how do you get to 0.86" but
**"what is the *boundary* of 0.86, and why is it so hard to cross?"** This writeup is a
rigorous, **leaderboard-validated** answer, and it offers two reusable things to anyone
fine-tuning Nemotron:

1. **A reproducible recipe that captures the *full* gain available from the public
   adapter** — a lossless conversion + a one-line LoRA key-rename that many teams at
   0.86 likely left partly on the table (the lossy path scores 0.85). The notebook
   rebuilds the exact file-submittable 0.86 artifact from public inputs (§1).
2. **A falsification map of the 0.86 ceiling**: ~12 distinct LB submissions across
   SFT-distillation, GRPO, RFT, STaR, model-soup, and delta-extrapolation, plus a
   **methodology** (`holdout-net-gain`) that predicts the LB where every cheaper proxy
   *misleads* — including a holdout metric we show is **anti-correlated** with the LB.
   This is the part that saves the next team GPU-weeks: it says precisely which doors
   are already closed, and how to tell (§2–§6).

> All numbers are grader-aligned (T=0) — either local evals on the 9500-problem
> training set (or its 2375-problem shard-0), or **public-leaderboard (LB) scores**,
> labelled as such. Training/eval/RL code, per-experiment logs, and the recipe are in
> the repository (`ITERATION_REPORT.md`, `ROADMAP.md`, `RL_ARCHITECTURE.md`, `training/rl/`).

## TL;DR for the next competitor

If you have forked the public ~0.86 adapter and are wondering why you can't push past it, here is the compressed map of the terrain, paid for in GPU-weeks of negative results:

1. **The 0.85 → 0.86 gap is a *packaging* problem, not a modeling one** — lossy adapter conversion + a LoRA key-naming mismatch, both fixable without touching the weights. **§1.**
2. **The 0.86 → 0.87 gap is a *capability* problem you cannot fine-tune around with the available supervision.** We ran ~12 distinct LB submissions across SFT-distillation, hand-solver→CoT, RFT, STaR, GRPO, model-soup, and delta-extrapolation. **Every one landed ≤ 0.86; none reached 0.87.** **§2–§3.**
3. **Two walls explain it.** A *prior-coupling* wall on the categories the model is already good at (improving a hard slice drifts a shared prior and regresses the larger correct set), and an *under-determination / capability* wall on all three "cliff" categories (the correct answers are not in the model's sample support at any temperature, so there is no gradient). **§2–§3.**
4. **The decisive lever is a KL anchor, not the data.** Anything that anchors to the 0.86 adapter (a KL term, or a soup that stays near it) holds exactly 0.86; anything that drifts (un-anchored SFT/STaR, or a delta amplified past ~1.5×) falls to 0.84–0.85. Nothing escapes *upward*. **§2.**
5. **Your local proxy will lie to you.** We document a holdout metric that is *anti-correlated* with the LB, and a methodology rule that survived a three-way swing of contradictory evidence. **§4.**
6. **Two dead ends worth knowing**: the grader is closed (no inference-time search), and you **cannot LoRA the MoE router** in vLLM 0.17.1. **§5–§6.**

---

## 1. The 0.85 → 0.86 gap is conversion loss + a key-naming bug

The strongest public adapter is Huikang's winning Tinker adapter. Getting it to score on this grader means converting it to a PEFT LoRA that loads against the `nemotron_h` modules. Two independent traps cost ~1pp each.

### 1a. SVD conversion is lossy by ~25%
Hand-rolling the Tinker→PEFT conversion (SVD on the Mamba `gate_proj` + `x_proj` to synthesize `in_proj`) **discards ~25% of the singular mass** → **LB 0.85**. The *lossless* conversion via `tinker_cookbook.build_lora_adapter` preserves it → **LB 0.86**. Same weights, conceptually — the difference is entirely how faithfully the Mamba projections are reconstructed.

### 1b. The LoRA key prefix decides whether the grader loads your adapter at all
The lossless adapter ships with keys prefixed `base_model.model.model.layers...`. The base model (and grader) expose the Mamba stack under `backbone.layers...`. Submit with the original prefix and PEFT silently fails to attach the LoRA to the Mamba modules → the grader runs **≈ the base model** → **LB 0.85**. (And if you try to *train* from it, the KL-to-reference term explodes, because the policy ≈ base while the reference is the intended adapter — a symptom that initially looked like a training bug.)

**Fix:** rename `base_model.model.model.` → `base_model.model.backbone.`. The renamed adapter (a) loads correctly → **LB 0.86**, and (b) casts cleanly to bf16 at **1.78 GB** — under the ~2 GB file-upload limit, so 0.86 is submittable by *file upload*, no notebook needed. There is **no "bf16 ceiling"**: the earlier belief that bf16 capped us at 0.85 was this key-naming bug, not precision.

> Practical takeaway: before concluding "my trained adapter regressed," verify your LoRA keys match the base module names. A mis-prefixed adapter and an untrained base score identically — a silent, expensive failure mode.

---

## 2. The prior-coupling wall, and why the KL anchor is the only thing that holds

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

All apparent headroom is in three "cliff" categories. The intuitive move — fine-tune to lift them — hits **prior-coupling**: the cliff and saturated categories share low-level priors (number formatting, disambiguation of under-specified constraints, search-vs-recall heuristics). Pulling on the hard slice drifts those shared priors and **regresses the much larger already-correct set**.

We measured this on the **leaderboard**, across every training family. The cleanest summary is a 2×2 over *(single- vs multi-category)* × *(anchored vs not)*:

| | un-anchored (SFT / STaR, β=0) | KL-anchored (GRPO / STaR, β=0.02) |
|---|---|---|
| **single-category** | 0.85 (STaR on under-constrained bit) | 0.85 → 0.84 (bit-only GRPO, iterating) |
| **multi-category** | **0.84** (broad self-distillation) | **0.86** (multi-cat GRPO + true KL anchor) |

The load-bearing variable is **not** single-vs-multi-category — it is the **KL anchor**. Concretely:

- **Bit-only GRPO progressively *regresses*** from the true-0.86 base: LB **0.86 → 0.85 → 0.85 → 0.84** as it iterates, *while* the 100-problem bit-holdout it optimizes **climbs** 0.59 → 0.62. Proxy and target move in opposite directions (§4).
- **Un-anchored self-distillation (STaR, β=0) regresses regardless of breadth** — single-cat = LB 0.85, broad multi-cat = LB **0.84** (broader drift is *worse*). The near-zero training loss (the model already assigns its own samples high probability) barely moves the weights, yet the small drift still costs 1–2pp on the harder hidden set.
- **A KL-to-warmstart anchor (β=0.02) holds exactly 0.86** and never exceeds it. Multi-cat GRPO with the saturated categories pinned as KL anchors = LB **0.86**, no regression — strictly better than bit-only's 0.84–0.85, but the gradient-bearing categories have no learnable headroom left to climb.
- **Model soup holds 0.86 by cancelling drift.** Weight-space averaging of two independent 0.86 optima (the clean adapter + a GRPO checkpoint) = LB **0.86** — it *cancels* the idiosyncratic drift that sank the individual trained adapters, but produces no net gain.
- **Delta extrapolation overshoots *downward*.** Amplifying a trained delta (`base + α·(trained − base)`) past α≈1.3 falls to **0.85**; α=2.0 = **0.85**. No coefficient yields 0.87. There is no "good direction" to amplify — the deltas are net-neutral-to-negative.

**Conclusion:** single-slice optimization is net-negative; the best any training run achieves is *net-neutral* (hold 0.86), and only by anchoring tightly (KL) or averaging (soup) so the shared prior cannot drift. The headroom that *would* move the LB is entirely in the cliff categories — which brings us to §3.

---

## 3. All three "cliff" categories are under-determined — unlearnable with the available supervision

The cliff categories hold ~13pp of nominal headroom, but it is a mirage. Each is an *execution / under-determination* wall, not a missing-supervision one: **the correct answers are not in the policy's sample support, so there is no positive signal to learn from.**

**Cryptarithm (`_deduce` 7.0%, `_guess` 6.1%) — the most thoroughly falsified dead end:**
- **No gradient at any temperature.** pass@16 on a well-posed cryptarithm holdout is **flat at ~10% across T = 1.0 / 1.3 / 1.6** (pass@1 *drops* with T). The correct letter→digit assignments are simply not sampled at any temperature → zero positive examples → GRPO / RFT / STaR have no gradient.
- **SFT on correct solver traces is flat.** From the true-0.86 base, 4226 cipher-search traces (median 629 tokens, train loss 43 → 7) leave the holdout **flat at 9–10% = the base rate**. The model fits the *format* of a search trace but cannot *execute* the search on a novel cipher.
- **Hand-written solvers are weaker than the model**, so "solver → CoT → SFT" only mis-teaches: solvers score 59.9% (bit) / 39.8% (eq-deduce) / ~4% (cipher-digit) vs the model's 87.5 / 90.9 / ~7%. Solvers are useful only as *verifiers*, never as generators.
- Each puzzle is an independent random cipher, so memorization does not transfer — a holdout *canary* showed **negative** transfer even as train loss collapsed 95 → 13.

**equation_numeric_guess (16.2%) — confirmed under-determined too (final probe):**
- A pass@16 probe (G=16, T=1.0) on this category returns **16.2% — identical to the base greedy accuracy**, with **84% of problems having zero correct sample in 16 tries**. Like cryptarithm, the answer is not in support, so there is no gradient. It is a "guess a global prior when the disambiguating operation is absent" task, not a learnable solver slice.

So: all three cliff categories are unlearnable; `bit_manipulation` regresses when trained (§2); and the saturated categories have no headroom. **There is no trainable path to 0.87 from the public 0.86 base.** Reaching it would require a genuinely stronger base policy (none is public above 0.86) or inference-time search (ruled out — §5).

---

## 4. Methodology: the only ceiling estimate you can trust

The most transferable lesson. We were repeatedly and confidently *wrong* about what was salvageable, and each wrong estimate was corrected only by a stronger class of evidence:

| Estimator | Said about bit_manipulation | Verdict |
|---|---|---|
| Static-solver coverage | "unsalvageable" | wrong |
| pass@k upper bound | "+0.2–0.7pp recoverable" | wrong (over-optimistic) |
| Holdout-*subset* accuracy | "checkpoint +3, looks good" | wrong (didn't extrapolate) |
| **Holdout net-gain extrapolated to the full set** | "net ≈ 0 / negative" | **correct (matched LB)** |

**The rule:** *salvageability and the ceiling can only be judged by **holdout net-gain (recovered − regressed − sentinel-drift) extrapolated to the full set.*** A checkpoint that recovers 8% of a failing slice but regresses 1.4% of a 10× larger correct slice is **net-negative**, even though every cheaper proxy says "improving."

The sharpest illustration: the bit-holdout we optimized with GRPO is **anti-correlated** with the LB — climbing the proxy 0.59 → 0.62 *cost* us LB 0.86 → 0.84. And the strongest form of the trap: a multi-category STaR adapter scored **identically to the base on every shard-0 category** (overall 88.00% = 88.00%, bit 89.8 = 89.8) yet was **−2pp on the LB**. For these near-0.86, small-perturbation adapters, local shard eval has *essentially zero discriminative power*.

Corollary on eval noise: vLLM + LoRA multi-GPU decoding is **not bit-deterministic even at T=0** (~±0.3pp run-to-run on a 2375-problem shard). Trust only the LB and full-set differences ≥ 1pp. (We submitted soups and extrapolations *without* local eval, precisely because it cannot distinguish them — the right call.)

---

## 5. The grader is closed, and the ceiling is not an artifact

Submission is a **fixed adapter, single greedy (T=0) pass** — no multi-sampling, best-of-N, test-time training, or tools. Every "inference-time search" idea (exactly what cryptarithm needs) is off the table; the LoRA weights are the only lever, and §2–§3 show they are already at the reachable ceiling for the public base.

We also audited the pipeline to rule out a *masked* gain — i.e. that the "0.86 ceiling" was a bug. It is not: the RL reward uses the **same** `extract`/`verify` as the grader; the local eval is grader-faithful (T=0, chat template with `enable_thinking`, boxed extraction, 7680/8192). One real bug *was* found — a `drop_zero_std_groups` default that silently dropped the saturated KL-anchor groups, so multi-cat GRPO wasn't fully anchoring; fixing it made the anchor hold 0.86 cleanly, but still did not exceed it.

Three more things that are load-bearing and easy to get wrong:
- **Train and eval at T=0**, not the `temperature=1.0` shown in the public `metric.py` — the grader overrides it to greedy. A single token with <50% logprob derails the whole chain.
- **Mamba layers must be in `target_modules`** (`in_proj`, `out_proj`). Adapters that only target attention+MLP leave the Mamba reasoning circuits frozen and score 0.27–0.60.
- **Don't filter training data to short responses** — the model needs to learn long, complete reasoning ending in `\boxed{}` within the 7680-token budget.

---

## 6. You cannot LoRA the MoE router in vLLM 0.17.1

A tempting idea for dodging prior-coupling: add LoRA to the 128-expert **router/gate**, so hard-category tokens can be re-routed without perturbing the shared FFN/attention priors. **It is a dead end on this grader, for a concrete reason worth documenting:**

- The `nemotron_h` router is a `GateLinear` module (`...mixer.gate`), a *subclass* of `ReplicatedLinear`.
- vLLM 0.17.1 matches LoRA-able modules by **exact type** (`type(layer) is ReplicatedLinear`, etc.). A subclass does not match, and the LoRA manager asserts the replaced module is a `BaseLayerWithLoRA`.
- Therefore an adapter containing `*.mixer.gate.lora_A/B` is **never applied at inference** — best case rejected, worse case silently ignored.

So router-LoRA cannot be a submission artifact without patching the grader's vLLM (which the rules forbid). The supported LoRA targets are the expert `up_proj`/`down_proj`, Mamba `in_proj`/`out_proj`, attention, and `lm_head` — all of which we exhausted in §2.

---

## 7. The one lever we did not fully exhaust

For honesty: the single genuinely-untested edge is **KL-anchored RL with a *clean iterative* rollout** — resample fresh rollouts each round → ≤1–2 epochs → eval holdout → repeat, with β>0 KL-to-warmstart. Our iterative GRPO reused one rollout across many epochs, which conflates the prior-coupling wall with off-policy *staleness*. A clean loop *might* let the KL anchor control regression more gently than MLE-SFT. We judge the expected gain small (the wall is a mechanism, not an artifact — every anchored config converged to exactly 0.86) and the engineering cost high, but it is the only stone left unturned. Beyond it, real progress needs a stronger base policy or an open grader.

---

*Reproducible artifacts: the file-submittable 0.86 recipe (lossless convert → key rename → bf16 cast), the hand-rolled multi-node GRPO/STaR pipeline (`training/rl/`), and the model-soup utility (`training/soup_adapters.py`). Full per-experiment numbers in `ITERATION_REPORT.md` / `ROADMAP.md`.*
