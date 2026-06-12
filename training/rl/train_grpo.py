"""GRPO trainer: consume rollout JSONL, update LoRA adapter via FSDP + PEFT.

Run under accelerate launch with the project's fsdp_config.yaml. Designed as
a minimal hand-rolled trainer rather than an HF Trainer subclass — keeps the
unusual two-forward-pass (policy + reference) inside compute_loss clean.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# === Compatibility patches (torch 2.5.1 + transformers 5.3.0+computecanada) ===
import transformers.generation as _tg  # noqa: E402
if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
    @dataclass
    class _Dummy:
        pass

    _tg.GreedySearchDecoderOnlyOutput = _Dummy
    _tg.SampleDecoderOnlyOutput = _Dummy

import torch  # noqa: E402
import torch.distributed.fsdp as _fsdp_mod  # noqa: E402

if not hasattr(_fsdp_mod, "register_fsdp_forward_method"):
    _fsdp_mod.register_fsdp_forward_method = lambda *a, **kw: None

from accelerate import Accelerator  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "model")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import iter_jsonl  # noqa: E402
from ref_policy import disabled_adapter  # noqa: E402
from grpo_loss import compute_token_logprobs, grpo_loss  # noqa: E402


class RolloutDataset(Dataset):
    """Flattens K prompts × G rollouts → K*G (input_ids, n_prompt, advantage) items.

    If the JSONL carries pre-computed `ref_logp` per response (when rollout was
    run with --compute_ref_logp), they are loaded too — train_grpo will then
    skip the disable_adapter forward and use these directly.
    """

    def __init__(self, jsonl_paths: list[Path], max_length: int = 8192):
        self.items: list[dict] = []
        self.has_ref_logp: bool = False
        for path in jsonl_paths:
            for rec in iter_jsonl(path):
                prompt_ids = rec["prompt_ids"]
                for g in rec["group"]:
                    resp_ids = list(g["response_ids"])
                    if not resp_ids:
                        continue
                    total = list(prompt_ids) + resp_ids
                    n_prompt = len(prompt_ids)
                    keep_n = len(resp_ids)
                    if len(total) > max_length:
                        # Truncate response end (keep prompt intact, keep at least 1 resp token).
                        keep = max_length - n_prompt
                        if keep <= 1:
                            continue
                        total = total[: n_prompt + keep]
                        keep_n = keep
                    item = {
                        "input_ids": total,
                        "n_prompt": n_prompt,
                        "advantage": float(g.get("advantage", 0.0)),
                        "reward": float(g.get("reward", 0.0)),
                        "category": rec.get("category", "?"),
                    }
                    if "ref_logp" in g:
                        rlp = list(g["ref_logp"])[:keep_n]
                        # pad to keep_n with 0.0 if shorter
                        if len(rlp) < keep_n:
                            rlp = rlp + [0.0] * (keep_n - len(rlp))
                        item["ref_logp"] = rlp
                        self.has_ref_logp = True
                    self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return dict(self.items[idx])


def collate(batch: list[dict], pad_id: int, pad_to_mult: int = 8) -> dict:
    if not batch:
        raise ValueError("empty batch")
    max_len = max(len(b["input_ids"]) for b in batch)
    max_len = int(math.ceil(max_len / pad_to_mult)) * pad_to_mult
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    response_mask = torch.zeros((B, max_len), dtype=torch.long)
    advantages = torch.zeros(B, dtype=torch.float32)
    has_ref = all("ref_logp" in b for b in batch)
    # ref_logp aligns with shift positions [0..T-2]; response position t in input_ids
    # corresponds to shift index t-1 (predicting input_ids[t] from earlier context).
    ref_logp = torch.zeros((B, max_len - 1), dtype=torch.float32) if has_ref else None
    for i, b in enumerate(batch):
        T = len(b["input_ids"])
        input_ids[i, :T] = torch.tensor(b["input_ids"], dtype=torch.long)
        attention_mask[i, :T] = 1
        response_mask[i, b["n_prompt"]: T] = 1
        advantages[i] = b["advantage"]
        if has_ref:
            rlp = b["ref_logp"]
            n_prompt = b["n_prompt"]
            # The j-th response token (0-indexed) is at input_ids position n_prompt+j,
            # and its logprob corresponds to shift index n_prompt+j-1.
            for j, lp in enumerate(rlp):
                idx = n_prompt + j - 1
                if 0 <= idx < max_len - 1:
                    ref_logp[i, idx] = lp
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "advantages": advantages,
    }
    if has_ref:
        out["ref_logp"] = ref_logp
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_jsonl", nargs="+", required=True, type=Path,
                    help="One or more rollout JSONL files (multi-node shards merged here).")
    ap.add_argument("--adapter_dir", required=True, type=Path,
                    help="PEFT adapter to load (Huikang baseline initially; previous iter thereafter).")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL penalty. β=0 (default) skips ref logprob entirely (Codex 2026-05-25 recommendation).")
    ap.add_argument("--drop_zero_std_groups", action="store_true", default=True,
                    help="Dynamic sampling (NeMo-RL/DAPO): drop groups where all rollouts have the same reward "
                         "(adv_std=0 → zero gradient → wasted compute, also explodes KL gradient relative to PG).")
    ap.add_argument("--no_drop_zero_std_groups", dest="drop_zero_std_groups", action="store_false")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--tb_dir", type=str, default=None,
                    help="TensorBoard run dir (e.g. $HOME/tensorboard_logs/bit_grpo_iter). "
                         "Per-step scalars are written here on rank 0.")
    ap.add_argument("--global_step_offset", type=int, default=0,
                    help="Added to the local optimizer step for TB x-axis so a multi-"
                         "iteration GRPO run forms one continuous curve.")
    ap.add_argument("--save_only", action="store_true",
                    help="Skip training, just resave adapter (smoke convenience).")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    args = ap.parse_args()

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    tb_writer = None
    if is_main and args.tb_dir:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=args.tb_dir)
        print(f"[grpo] TensorBoard → {args.tb_dir} (step offset {args.global_step_offset})")

    if is_main:
        print("=" * 60)
        print("GRPO trainer config:")
        for k, v in sorted(vars(args).items()):
            print(f"  {k:20s} = {v}")
        print(f"  world_size           = {accelerator.num_processes}")
        print("=" * 60)

    torch.manual_seed(args.seed + accelerator.process_index)

    # Tokenizer (for pad_id)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    if is_main:
        print(f"[grpo] loading rollouts from {len(args.rollout_jsonl)} file(s) ...")
    dataset = RolloutDataset(args.rollout_jsonl, max_length=args.max_length)

    # Dynamic sampling filter (Codex priority #1): drop groups whose rewards
    # are all the same (std=0) — they produce zero advantage gradient.
    if args.drop_zero_std_groups:
        from collections import defaultdict
        by_prompt: dict = defaultdict(list)
        for it in dataset.items:
            by_prompt[(it["category"], id(it))].append(it)
        # Re-group by reward identity within each rollout JSONL — we use the
        # advantage values (already group-normalized at rollout time). A group
        # of identical rewards yields advantage=0 for every member.
        kept: list = []
        dropped_groups = 0
        for it in dataset.items:
            if abs(it["advantage"]) > 1e-9:
                kept.append(it)
            else:
                dropped_groups += 1
        n_before = len(dataset.items)
        dataset.items = kept
        if is_main:
            print(f"[grpo] dynamic sampling filter: kept {len(kept)}/{n_before} items "
                  f"({dropped_groups} dropped — zero-advantage from degenerate groups)")
        if not kept:
            if is_main:
                print("[grpo] FATAL: every rollout had zero advantage — no signal to learn from.")
                print("       Lift baseline pass rate, raise temperature, or use bigger G.")
            return

    if is_main:
        print(f"[grpo] {len(dataset)} (prompt, response) items")
        cat_counts: dict[str, int] = {}
        for it in dataset.items:
            cat_counts[it["category"]] = cat_counts.get(it["category"], 0) + 1
        for k, v in sorted(cat_counts.items()):
            print(f"  {k:30s} {v}")

    if len(dataset) == 0:
        if is_main:
            print("[grpo] empty dataset — aborting")
        return

    def _collate(batch):
        return collate(batch, pad_id)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    if is_main:
        print(f"[grpo] loading base model from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if is_main:
        print(f"[grpo] loading LoRA adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(model, str(args.adapter_dir), is_trainable=True)
    model = model.to(torch.bfloat16)
    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        if is_main:
            print("[grpo] enabling gradient checkpointing (use_reentrant=False)")
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Older signature
            model.gradient_checkpointing_enable()

    if is_main:
        model.print_trainable_parameters()

    # Plan A: keep lm_head + embed_tokens UNSHARDED across FSDP ranks.
    # Nemotron has a 131K vocab; lm_head.weight alone is ~2.15 GB and FSDP
    # gather of it during forward/backward spikes peak VRAM by ~2.5 GB at the
    # exact moment we have the least headroom — causing OOM at backward.
    # Putting these modules in `ignored_modules` makes FSDP keep them fully
    # replicated on each rank: +4.3 GB static / rank, but no gather spike.
    # Find modules to keep UNSHARDED across FSDP ranks. We target two specific
    # cases:
    #   1) Standalone nn.Embedding for the token embedding (not LoRA-wrapped)
    #   2) The frozen nn.Linear INSIDE a PEFT LoraLayer wrapping lm_head — i.e.
    #      ".lm_head.base_layer". We do NOT ignore the wrapping LoraLayer itself
    #      because its tiny lora_A/B params SHOULD still be FSDP-sharded.
    #
    # Architecture name variations:
    #   Llama/Qwen: "embed_tokens"
    #   Nemotron-H: "embeddings" (path: backbone.embeddings)
    #   GPT-style : "wte"
    ignored_modules: list[nn.Module] = []
    ignored_names: list[str] = []
    EMBED_NAMES = {"embed_tokens", "embeddings", "wte"}
    for name, mod in model.named_modules():
        last = name.split(".")[-1]
        if isinstance(mod, nn.Embedding) and last in EMBED_NAMES:
            ignored_modules.append(mod)
            ignored_names.append(name)
        elif last == "base_layer" and name.endswith(".lm_head.base_layer"):
            ignored_modules.append(mod)
            ignored_names.append(name)
        elif isinstance(mod, nn.Linear) and last == "lm_head":
            # Non-PEFT-wrapped lm_head (defensive — won't trigger when LoRA covers lm_head)
            ignored_modules.append(mod)
            ignored_names.append(name)
    if is_main:
        print(f"[fsdp] keeping unsharded ({len(ignored_modules)} modules):")
        for n in ignored_names:
            print(f"  {n}")

    # FSDP doesn't move ignored_modules to the device — and with
    # cpu_ram_efficient_loading they may still be on CPU. Move them to the
    # local GPU manually so forward doesn't blow up with cuda↔cpu mismatch.
    target_device = accelerator.device
    if is_main:
        print(f"[fsdp] moving ignored modules to {target_device}")
    for mod in ignored_modules:
        # If params are on meta, materialize on the local GPU first
        has_meta = any(p.is_meta for p in mod.parameters(recurse=True))
        if has_meta:
            mod.to_empty(device=target_device, recurse=True)
        else:
            mod.to(target_device)

    # Inject into accelerate's FSDP plugin BEFORE prepare() so FSDP excludes
    # these modules from its wrap policy.
    if hasattr(accelerator.state, "fsdp_plugin") and accelerator.state.fsdp_plugin is not None:
        accelerator.state.fsdp_plugin.ignored_modules = ignored_modules
        if is_main:
            print(f"[fsdp] injected ignored_modules into plugin")
    else:
        if is_main:
            print("[fsdp] WARNING: no fsdp_plugin found on accelerator.state — "
                  "ignored_modules will not take effect")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if args.save_only:
        if is_main:
            print("[grpo] --save_only: skipping training, saving adapter ...")
        _save_adapter(accelerator, model, tokenizer, args.output_dir)
        return

    model.train()
    step = 0           # = number of optimizer.step() invocations
    micro_step = 0     # = number of forward+backward micro-batches
    optimizer.zero_grad(set_to_none=True)
    progress: list[dict] = []
    sanity_logged = False
    abort = False

    t0 = time.time()
    epoch = 0
    while step < args.max_steps and not abort:
        epoch += 1
        if is_main:
            print(f"[grpo] epoch {epoch}  step={step}/{args.max_steps}")
        for batch in loader:
            if step >= args.max_steps or abort:
                break
            input_ids = batch["input_ids"]
            attn = batch["attention_mask"]
            resp_mask = batch["response_mask"]
            advantages = batch["advantages"]

            # NOTE: do NOT skip zero-advantage batches. Zero-advantage means PG
            # term = 0, but KL term still contributes (= keeps policy from
            # drifting). Skipping breaks the grad-accumulation counter and
            # leaves optimizer.step() un-fired. Letting them flow costs ~no
            # extra time relative to the rollout.

            logp_new, shift_mask = compute_token_logprobs(model, input_ids, attn, resp_mask)
            if args.beta == 0.0:
                # β=0 path: skip ref logprob entirely (no KL term, no second forward).
                logp_ref = torch.zeros_like(logp_new)
            elif "ref_logp" in batch:
                # Pre-computed by rollout phase via vLLM (no LoRA). Skips the
                # disable_adapter forward → halves training-phase memory.
                logp_ref = batch["ref_logp"].to(logp_new.device, dtype=logp_new.dtype)
                logp_ref = logp_ref * shift_mask
            else:
                with torch.no_grad():
                    with disabled_adapter(model):
                        logp_ref, _ = compute_token_logprobs(model, input_ids, attn, resp_mask)

            if not sanity_logged and is_main and args.beta != 0.0:
                # G0 sanity: at iter 0 with a freshly-loaded adapter on rollouts taken
                # from THIS same adapter, logp_new should differ from logp_ref by
                # a NON-zero amount (adapter is active in policy, disabled in ref).
                diff = (logp_new - logp_ref).abs().sum().item()
                masked_tokens = shift_mask.sum().item()
                print(f"[sanity] |logp_new - logp_ref| sum = {diff:.4f} over "
                      f"{masked_tokens:.0f} response tokens "
                      f"({'OK — adapter active' if diff > 1e-3 else '⚠ ZERO — disable_adapter no-op?'})")
                sanity_logged = True

            loss, metrics = grpo_loss(logp_new, logp_ref, shift_mask, advantages, beta=args.beta)
            loss = loss / args.grad_acc
            accelerator.backward(loss)

            micro_step += 1
            if micro_step % args.grad_acc == 0:
                # FSDP-aware grad clipping (gathers grads across ranks).
                grad_norm = accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if is_main and step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[step {step:4d}/{args.max_steps}] "
                          f"loss={metrics['loss']:+.4f} pg={metrics['pg_loss']:+.4f} "
                          f"kl={metrics['kl_loss']:.4f} grad_norm={float(grad_norm):.3f} "
                          f"adv_mean={metrics['adv_mean']:+.3f} "
                          f"adv_std={metrics['adv_std']:.3f} "
                          f"n_resp={metrics['n_resp_tokens_mean']:.0f} "
                          f"elapsed={elapsed:.1f}s")
                    progress.append({
                        "step": step,
                        "grad_norm": float(grad_norm),
                        "elapsed": elapsed,
                        **metrics,
                    })
                    if tb_writer is not None:
                        gstep = args.global_step_offset + step
                        tb_writer.add_scalar("train/loss", metrics["loss"], gstep)
                        tb_writer.add_scalar("train/pg_loss", metrics["pg_loss"], gstep)
                        tb_writer.add_scalar("train/kl_loss", metrics["kl_loss"], gstep)
                        tb_writer.add_scalar("train/grad_norm", float(grad_norm), gstep)
                        tb_writer.add_scalar("train/adv_mean", metrics["adv_mean"], gstep)
                        tb_writer.add_scalar("train/adv_std", metrics["adv_std"], gstep)
                        tb_writer.add_scalar("train/n_resp_tokens", metrics["n_resp_tokens_mean"], gstep)
                        # saturation proxy: seq_logp_mean → 0 means the policy is becoming
                        # deterministic on its own samples (confidence saturating).
                        tb_writer.add_scalar("train/seq_logp_mean", metrics["seq_logp_mean"], gstep)
                        tb_writer.flush()
                    if not math.isfinite(metrics["loss"]):
                        if is_main:
                            print("[grpo] non-finite loss — aborting")
                        abort = True
                        break

    if tb_writer is not None:
        tb_writer.close()
    _save_adapter(accelerator, model, tokenizer, args.output_dir, progress)


def _save_adapter(accelerator, model, tokenizer, output_dir: Path, progress: list[dict] | None = None) -> None:
    """Save PEFT adapter under FSDP.

    Pre-gather state_dict (all ranks participate), then filter to LoRA-only
    keys and save on rank 0 only. The explicit LoRA filter is required
    because PEFT 0.18 + FSDP `ignored_modules` causes get_peft_model_state_dict
    to also include the frozen `.base_layer.weight` of LoRA-wrapped modules
    we kept unsharded (e.g. lm_head). That bloats the adapter by the size of
    the base weight (~700 MB for lm_head at vocab=131k hidden=2688).
    """
    accelerator.wait_for_everyone()
    is_main = accelerator.is_main_process
    output_dir = Path(output_dir)
    # All ranks participate in the FULL_STATE_DICT all-gather.
    state_dict = accelerator.get_state_dict(model)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Explicit LoRA-only filter: keep keys with "lora_" in them. Drops the
        # frozen base_layer / embedding weights that snuck in via ignored_modules.
        lora_state = {k: v for k, v in state_dict.items() if "lora_" in k}
        n_kept = len(lora_state)
        n_dropped = len(state_dict) - n_kept
        size_mb = sum(v.numel() * v.element_size() for v in lora_state.values()) / (1024 * 1024)
        print(f"[grpo] saving adapter to {output_dir}  "
              f"(kept {n_kept} LoRA keys, dropped {n_dropped} non-LoRA, {size_mb:.0f} MB)")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(output_dir), state_dict=lora_state, safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))
        if progress is not None:
            with open(output_dir / "progress.json", "w") as f:
                json.dump(progress, f, indent=2)
        print(f"[grpo] done. {len(progress) if progress else 0} optimizer steps logged.")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
