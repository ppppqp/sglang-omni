#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate Qwen3-TTS generation and optional WER benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_tts_results(
    speed: dict[str, Any],
    *,
    wer: dict[str, Any] | None = None,
    max_corpus_wer: float = 0.10,
) -> list[str]:
    failures: list[str] = []
    requests = speed.get("per_request")
    if not isinstance(requests, list) or not requests:
        return ["speed results contain no requests"]
    for request in requests:
        request_id = request.get("id", "unknown")
        if not request.get("is_success"):
            failures.append(
                f"request={request_id}: generation failed: {request.get('error')}"
            )
        if float(request.get("audio_duration_s") or 0.0) <= 0:
            failures.append(f"request={request_id}: generated audio is empty")

    if wer is not None:
        summary = wer.get("summary") or {}
        corpus_wer = summary.get("wer_corpus", summary.get("corpus_wer"))
        if corpus_wer is None:
            failures.append("WER results do not contain corpus WER")
        elif float(corpus_wer) > max_corpus_wer:
            failures.append(
                f"corpus WER {float(corpus_wer):.4f} exceeds {max_corpus_wer:.4f}"
            )
        samples = wer.get("per_sample") or []
        if any(not sample.get("is_success") for sample in samples):
            failures.append("one or more WER transcription requests failed")
    return failures


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-wer", action="store_true")
    parser.add_argument("--max-corpus-wer", type=float, default=0.10)
    args = parser.parse_args()

    speed = _load(args.output_dir / "speed_results.json")
    wer_path = args.output_dir / "wer_results.json"
    wer = _load(wer_path) if wer_path.exists() else None
    if args.require_wer and wer is None:
        failures = [f"required WER artifact is missing: {wer_path}"]
    else:
        failures = validate_tts_results(
            speed,
            wer=wer,
            max_corpus_wer=args.max_corpus_wer,
        )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("Qwen3-TTS correctness gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
