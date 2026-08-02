#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate Qwen3-Omni speech benchmark artifacts on completion and WER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_omni_results(
    payload: dict[str, Any],
    *,
    require_wer: bool,
    max_corpus_wer: float,
) -> list[str]:
    failures: list[str] = []
    generation = payload.get("generation", payload)
    requests = generation.get("per_request")
    if not isinstance(requests, list) or not requests:
        return ["Omni results contain no generation requests"]
    for request in requests:
        request_id = request.get("id", "unknown")
        if not request.get("is_success"):
            failures.append(
                f"request={request_id}: generation failed: {request.get('error')}"
            )
        if float(request.get("audio_duration_s") or 0.0) <= 0:
            failures.append(f"request={request_id}: generated audio is empty")

    accuracy = payload.get("accuracy") or {}
    wer = accuracy.get("wer")
    if require_wer and not isinstance(wer, dict):
        failures.append("required accuracy.wer result is missing")
    elif isinstance(wer, dict):
        corpus_wer = wer.get("wer_corpus", wer.get("corpus_wer"))
        if corpus_wer is None:
            failures.append("accuracy.wer does not contain corpus WER")
        elif float(corpus_wer) > max_corpus_wer:
            failures.append(
                f"corpus WER {float(corpus_wer):.4f} exceeds {max_corpus_wer:.4f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--require-wer", action="store_true")
    parser.add_argument("--max-corpus-wer", type=float, default=0.15)
    args = parser.parse_args()

    with args.results.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    failures = validate_omni_results(
        payload,
        require_wer=args.require_wer,
        max_corpus_wer=args.max_corpus_wer,
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("Qwen3-Omni correctness gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
