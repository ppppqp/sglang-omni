# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for upstream qwen-tts."""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Callable
from typing import Any

import torch

_APPLY_LOCK = threading.Lock()
_PATCHED_FLAG = "_sglang_omni_qwen_tts_compat_patched"
_MASK_PATCHED_FLAG = "_sglang_omni_qwen_tts_mask_patched"


@functools.cache
def _accepted_params(func: Callable[..., Any]) -> frozenset[str]:
    return frozenset(inspect.signature(func).parameters)


def _adapt_mask_kwargs(
    func: Callable[..., Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Adapt qwen-tts mask-builder kwargs to the installed Transformers API."""

    accepted = _accepted_params(func)
    adapted = dict(kwargs)
    if "input_embeds" in adapted and "input_embeds" not in accepted:
        value = adapted.pop("input_embeds")
        if "inputs_embeds" in accepted:
            adapted.setdefault("inputs_embeds", value)
    return {key: value for key, value in adapted.items() if key in accepted}


def _patch_qwen_tts_mask_builders() -> None:
    """Reconcile qwen-tts mask builders with Transformers 5.x."""

    try:
        from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as mod
    except Exception:
        return

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(mod, name, None)
        if original is None or getattr(original, _MASK_PATCHED_FLAG, False):
            continue

        def make_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return orig(*args, **_adapt_mask_kwargs(orig, kwargs))

            setattr(wrapper, _MASK_PATCHED_FLAG, True)
            return wrapper

        setattr(mod, name, make_wrapper(original))


def _compute_default_rope_parameters(
    config: Any,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    del seq_len, layer_type
    base = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


def apply_qwen_tts_transformers_compatibility_patches() -> None:
    """Patch Transformers APIs expected by qwen-tts."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.utils import generic

    with _APPLY_LOCK:
        ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)
        _patch_qwen_tts_mask_builders()

        current = generic.check_model_inputs
        if getattr(current, _PATCHED_FLAG, False):
            return

        try:
            signature = inspect.signature(current)
        except (TypeError, ValueError):
            return

        params = list(signature.parameters.values())
        needs_func_arg = (
            len(params) == 1
            and params[0].default is inspect.Parameter.empty
            and params[0].kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if not needs_func_arg:
            return

        original = current

        def check_model_inputs_compat(
            func: Callable[..., Any] | None = None,
        ) -> Callable[..., Any]:
            if func is None:

                def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
                    return original(inner)

                return decorator
            return original(func)

        check_model_inputs_compat.__name__ = getattr(
            original, "__name__", "check_model_inputs"
        )
        check_model_inputs_compat.__doc__ = getattr(original, "__doc__", None)
        setattr(check_model_inputs_compat, _PATCHED_FLAG, True)
        generic.check_model_inputs = check_model_inputs_compat
