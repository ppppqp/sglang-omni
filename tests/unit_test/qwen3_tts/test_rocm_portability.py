# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from scripts.rocm.verify_tts_results import validate_tts_results
from sglang_omni.models.qwen3_tts import compat, sampling_kernels


def test_transformers_mask_kwargs_are_adapted_by_signature() -> None:
    def mask_builder(*, inputs_embeds, attention_mask=None):
        return inputs_embeds, attention_mask

    marker = object()
    adapted = compat._adapt_mask_kwargs(
        mask_builder,
        {
            "input_embeds": marker,
            "attention_mask": "mask",
            "cache_position": "unsupported",
        },
    )

    assert adapted == {"inputs_embeds": marker, "attention_mask": "mask"}


def test_rocm_uses_portable_sampling_fallback(monkeypatch) -> None:
    monkeypatch.setattr(sampling_kernels, "is_rocm", lambda: True)
    tensor = torch.ones((1, 1))

    assert (
        sampling_kernels.sample_from_sorted_probs_with_seed_small_k(
            tensor,
            torch.zeros_like(tensor, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
        is None
    )


def test_tts_gate_accepts_nonempty_audio_and_low_wer() -> None:
    speed = {
        "per_request": [{"id": "one", "is_success": True, "audio_duration_s": 1.25}]
    }
    wer = {
        "summary": {"wer_corpus": 0.05},
        "per_sample": [{"is_success": True}],
    }

    assert not validate_tts_results(speed, wer=wer, max_corpus_wer=0.10)


def test_tts_gate_reports_generation_and_wer_failures() -> None:
    speed = {
        "per_request": [
            {
                "id": "bad",
                "is_success": False,
                "audio_duration_s": 0,
                "error": "request failed",
            }
        ]
    }
    wer = {
        "summary": {"wer_corpus": 0.25},
        "per_sample": [{"is_success": False}],
    }

    failures = validate_tts_results(speed, wer=wer, max_corpus_wer=0.10)

    assert len(failures) == 4
