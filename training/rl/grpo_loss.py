"""GRPO loss + Schulman k3 KL on token-level logprobs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning per-token logprobs over response positions.

    Memory-critical: uses F.cross_entropy (fused log_softmax + gather) instead of
    explicit `logits.log_softmax(-1).gather(-1, ...)`. The fused kernel avoids
    materializing the [B, T-1, V=131k] log_softmax intermediate (~2 GB at T=8192)
    in the autograd graph. With base softmax this OOMs on 48 GB L40s; with fused
    cross_entropy it fits at seq=8192 comfortably.

    Shapes:
        input_ids:      [B, T]
        attention_mask: [B, T]
        response_mask:  [B, T]  -- 1 where input_ids[t] is a response token, 0 otherwise

    Returns:
        logp:       [B, T-1]  -- log p(input_ids[t+1] | input_ids[<= t]), masked
                                 by shifted response_mask. Zero elsewhere.
        shift_mask: [B, T-1]  -- response_mask shifted to align with logp.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, T, V]
    B, T, V = logits.shape
    shift_labels = input_ids[:, 1:].contiguous()       # [B, T-1]
    shift_mask = response_mask[:, 1:].contiguous()     # [B, T-1]

    # Memory: AVOID .contiguous() copy of shift_logits ([B, T-1, V] = up to
    # 1 GB at seq=4096). When B=1 we can squeeze + slice → contiguous view.
    # For B>1 we still incur a copy in .reshape(-1, V).
    if B == 1:
        # logits.shape = (1, T, V), strides (T*V, V, 1)
        # squeeze(0)[:-1].shape = (T-1, V), strides (V, 1) — CONTIGUOUS view
        shift_logits_flat = logits.squeeze(0)[:-1]      # [T-1, V] contig view
        shift_labels_flat = shift_labels.squeeze(0)     # [T-1]
    else:
        shift_logits_flat = logits[:, :-1, :].reshape(-1, V)  # may copy
        shift_labels_flat = shift_labels.reshape(-1)

    # Fused cross_entropy: avoids materializing log_softmax in the autograd graph.
    nll = F.cross_entropy(
        shift_logits_flat,
        shift_labels_flat,
        reduction="none",
    ).view(B, T - 1)                                   # [B, T-1]
    logp = -nll
    shift_mask_f = shift_mask.to(logp.dtype)
    logp = logp * shift_mask_f
    return logp, shift_mask_f


def grpo_loss(
    logp_new: torch.Tensor,
    logp_ref: torch.Tensor,
    shift_mask: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """GRPO objective with per-token mean reduction + Schulman k3 token-level KL.

    Args:
        logp_new:    [B, T-1]  policy logprobs (gradient flows here)
        logp_ref:    [B, T-1]  reference logprobs (detached internally)
        shift_mask:  [B, T-1]  per-token response mask aligned with the above
        advantages:  [B]       per-sequence group-relative advantage (precomputed)
        beta:        KL penalty strength

    Returns:
        (loss, metrics_dict)
    """
    n_resp = shift_mask.sum(dim=-1).clamp(min=1.0)  # [B]

    seq_logp = logp_new.sum(dim=-1) / n_resp                       # [B]
    pg_loss = -(advantages.to(seq_logp.dtype) * seq_logp).mean()

    # KL term — skipped when beta=0 to avoid the ref-model forward overhead
    if beta != 0.0:
        logp_ref = logp_ref.detach()
        delta = (logp_ref - logp_new) * shift_mask
        # Numerically stable k3: exp(delta) - delta - 1 ≥ 0
        delta = delta.clamp(min=-20.0, max=20.0)
        kl_token = (torch.exp(delta) - delta - 1.0) * shift_mask
        seq_kl = kl_token.sum(dim=-1) / n_resp
        kl_loss = seq_kl.mean()
        total = pg_loss + beta * kl_loss
    else:
        kl_loss = torch.tensor(0.0, device=pg_loss.device)
        total = pg_loss

    with torch.no_grad():
        adv_std = advantages.float().std().item() if advantages.numel() > 1 else 0.0
        metrics = {
            "loss": float(total.detach()),
            "pg_loss": float(pg_loss.detach()),
            "kl_loss": float(kl_loss.detach()),
            "seq_logp_mean": float(seq_logp.detach().mean()),
            "adv_mean": float(advantages.float().mean()),
            "adv_std": adv_std,
            "n_resp_tokens_mean": float(n_resp.float().mean()),
        }
    return total, metrics
