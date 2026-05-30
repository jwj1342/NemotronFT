"""disable_adapter context manager for FSDP+PEFT models.

Routes through whichever attribute exposes PEFT's disable_adapter() — works
on raw PeftModel, accelerate-prepared FSDP-wrapped PeftModel, or accelerator
.unwrap_model() result.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext


def _find_disable_adapter(model):
    """Walk a small list of likely attributes to find disable_adapter()."""
    for obj in (model, getattr(model, "module", None), getattr(model, "base_model", None)):
        if obj is None:
            continue
        if hasattr(obj, "disable_adapter") and callable(obj.disable_adapter):
            return obj.disable_adapter
    return None


@contextmanager
def disabled_adapter(model):
    """Yield with LoRA disabled (i.e. base-only forward).

    Falls back to a no-op context if disable_adapter is not available, with a
    warning — so the smoke test surfaces the issue at G0 rather than silently
    producing identical policy/ref logprobs.
    """
    fn = _find_disable_adapter(model)
    if fn is None:
        import warnings
        warnings.warn("[ref_policy] model has no disable_adapter(); using no-op context")
        with nullcontext():
            yield
        return
    with fn():
        yield
