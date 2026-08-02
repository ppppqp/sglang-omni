# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from scripts.rocm.verify_asr_results import validate_asr_results


def _payload(*, evaluated: int = 20, total: int = 20, wer: float = 0.02) -> dict:
    return {
        "results": [
            {
                "concurrency": 2,
                "evaluated": evaluated,
                "total": total,
                "corpus_wer": {"max": wer, "n": 1},
            }
        ]
    }


def test_rocm_asr_gate_accepts_complete_low_wer_run() -> None:
    assert not validate_asr_results(
        _payload(), max_corpus_wer=0.03, min_completion_rate=1.0
    )


def test_rocm_asr_gate_reports_quality_and_completion_failures() -> None:
    failures = validate_asr_results(
        _payload(evaluated=18, wer=0.04),
        max_corpus_wer=0.03,
        min_completion_rate=1.0,
    )

    assert any("completion rate" in failure for failure in failures)
    assert any("corpus WER" in failure for failure in failures)


def test_rocm_asr_gate_rejects_missing_results() -> None:
    assert validate_asr_results({}, max_corpus_wer=0.03, min_completion_rate=1.0) == [
        "benchmark output contains no result aggregates"
    ]
