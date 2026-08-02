# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from scripts.rocm.verify_omni_results import validate_omni_results


def _results(*, success: bool = True, duration: float = 1.0, wer: float = 0.1):
    return {
        "generation": {
            "per_request": [
                {
                    "id": "one",
                    "is_success": success,
                    "audio_duration_s": duration,
                    "error": None if success else "failed",
                }
            ]
        },
        "accuracy": {"wer": {"wer_corpus": wer}},
    }


def test_rocm_omni_gate_accepts_complete_low_wer_run() -> None:
    assert not validate_omni_results(_results(), require_wer=True, max_corpus_wer=0.15)


def test_rocm_omni_gate_reports_generation_and_quality_failures() -> None:
    failures = validate_omni_results(
        _results(success=False, duration=0, wer=0.3),
        require_wer=True,
        max_corpus_wer=0.15,
    )

    assert len(failures) == 3


def test_rocm_omni_gate_requires_wer_when_requested() -> None:
    payload = _results()
    payload.pop("accuracy")

    assert "required accuracy.wer result is missing" in validate_omni_results(
        payload, require_wer=True, max_corpus_wer=0.15
    )
