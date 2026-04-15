"""
distillation_training/weighted_trainer.py — Trainer and DataCollator with per-token loss weighting.

Enables different loss weights for reasoning tokens vs \\boxed{answer} tokens,
so the model prioritises getting correct final answers over merely imitating
reasoning patterns.
"""

import torch
from transformers import Trainer, DataCollatorForSeq2Seq


class WeightedDataCollator(DataCollatorForSeq2Seq):
    """Extends DataCollatorForSeq2Seq to handle the extra `loss_weights` field.

    Pops `loss_weights` before the parent collates (so padding/truncation of
    input_ids / labels works normally), then pads the weights to match.
    """

    def __call__(self, features):
        # Extract extra fields before parent processes features
        weight_lists = [f.pop("loss_weights") for f in features]
        boxed_lists = [f.pop("is_boxed") for f in features]

        batch = super().__call__(features)

        # Pad weights and is_boxed to the same length as labels
        max_len = batch["labels"].shape[1]
        padded_weights = []
        padded_boxed = []
        for w, b in zip(weight_lists, boxed_lists):
            pad_len = max_len - len(w)
            padded_weights.append(w + [0.0] * pad_len)
            padded_boxed.append(b + [False] * pad_len)

        batch["loss_weights"] = torch.tensor(padded_weights, dtype=torch.bfloat16)
        batch["is_boxed"] = torch.tensor(padded_boxed, dtype=torch.bool)
        return batch


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-token loss weights.

    Each sample's loss_weights are pre-normalised to sum to 1.0 (in dataset.py),
    so every sample contributes equally to the batch loss regardless of length.

    Within each sample, the relative weight ratio (e.g. 5× for boxed tokens
    vs 1× for reasoning tokens) is preserved by the normalisation.

    The batch loss is: sum of per-sample weighted losses / number of samples.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weights = inputs.pop("loss_weights")  # (B, seq_len)
        is_boxed = inputs.pop("is_boxed")          # (B, seq_len)

        outputs = model(**inputs)
        logits = outputs.logits  # (B, seq_len, vocab)

        labels = inputs["labels"]  # (B, seq_len)

        # Shift for causal LM: predict token at position t from position t-1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()
        shift_boxed = is_boxed[..., 1:].contiguous()

        # Per-token cross-entropy (no reduction, ignore_index=-100 → loss=0)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Each sample's weights sum to ~1.0, so total weight ≈ batch_size.
        # Dividing by weight sum gives per-sample average loss.
        flat_weights = shift_weights.view(-1)
        weighted_loss = (per_token_loss * flat_weights).sum() / flat_weights.sum().clamp(min=1e-8)

        # ── Log decomposed losses to TensorBoard ───────────────────────
        # Unweighted CE averages for boxed vs reasoning tokens (for monitoring only)
        if self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                flat_loss = per_token_loss
                flat_boxed = shift_boxed.view(-1)
                # supervised = tokens with non-zero weight
                supervised = flat_weights > 0

                boxed_mask = flat_boxed & supervised
                reasoning_mask = (~flat_boxed) & supervised

                if boxed_mask.any():
                    boxed_loss = flat_loss[boxed_mask].mean().item()
                    self._boxed_loss = boxed_loss
                if reasoning_mask.any():
                    reasoning_loss = flat_loss[reasoning_mask].mean().item()
                    self._reasoning_loss = reasoning_loss

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def log(self, logs, start_time=None, **kwargs):
        """Inject decomposed losses into the log dict before TensorBoard write."""
        if hasattr(self, "_boxed_loss"):
            logs["train/boxed_loss"] = self._boxed_loss
            del self._boxed_loss
        if hasattr(self, "_reasoning_loss"):
            logs["train/reasoning_loss"] = self._reasoning_loss
            del self._reasoning_loss
        super().log(logs, start_time=start_time, **kwargs)
