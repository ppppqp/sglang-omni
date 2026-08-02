#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate Qwen3-ASR benchmark output on completion and transcription quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_asr_results(
    payload: dict[str, Any],
    *,
    max_corpus_wer: float,
    min_completion_rate: float,
) -> list[str]:
    """Return human-readable failures for every benchmark aggregate."""

    failures: list[str] = []
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return ["benchmark output contains no result aggregates"]

    for result in results:
        concurrency = result.get("concurrency", "unknown")
        total = int(result.get("total", 0))
        evaluated = int(result.get("evaluated", 0))
        completion_rate = evaluated / total if total else 0.0
        if completion_rate < min_completion_rate:
            failures.append(
                f"concurrency={concurrency}: completion rate {completion_rate:.4f} "
                f"is below {min_completion_rate:.4f} ({evaluated}/{total})"
            )

        wer_stats = result.get("corpus_wer") or {}
        worst_wer = wer_stats.get("max")
        sample_count = int(wer_stats.get("n", 0))
        if worst_wer is None or sample_count == 0:
            failures.append(f"concurrency={concurrency}: corpus WER is unavailable")
        elif float(worst_wer) > max_corpus_wer:
            failures.append(
                f"concurrency={concurrency}: corpus WER {float(worst_wer):.4f} "
                f"exceeds {max_corpus_wer:.4f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--max-corpus-wer", type=float, default=0.03)
    parser.add_argument("--min-completion-rate", type=float, default=1.0)
    args = parser.parse_args()

    with args.results.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    failures = validate_asr_results(
        payload,
        max_corpus_wer=args.max_corpus_wer,
        min_completion_rate=args.min_completion_rate,
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("Qwen3-ASR correctness gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
