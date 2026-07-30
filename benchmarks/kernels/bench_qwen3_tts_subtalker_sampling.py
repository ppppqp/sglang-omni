# SPDX-License-Identifier: Apache-2.0
"""Benchmark the fused Qwen3-TTS bounded subtalker sampling pipeline."""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    sample_from_sorted_probs_with_seed_small_k,
    sample_from_sorted_scores_with_seed_small_k,
)


def _eager_post_topk(
    sorted_scores: torch.Tensor,
    sorted_idx: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    width = sorted_scores.shape[1]
    rank = torch.arange(width, device=sorted_scores.device).unsqueeze(0)
    keep_top_k = rank < top_ks.unsqueeze(1)
    masked_scores = sorted_scores.masked_fill(~keep_top_k, -float("inf"))
    probs = torch.softmax(masked_scores, dim=-1)
    active_top_p = (top_ps > 0.0) & (top_ps < 1.0)
    remove = (torch.cumsum(probs, dim=-1) > top_ps.unsqueeze(1)) & (
        active_top_p.unsqueeze(1)
    )
    remove[:, 0] = False
    probs = probs.masked_fill(remove | ~keep_top_k, -float("inf"))
    sampled = sample_from_sorted_probs_with_seed_small_k(
        probs, sorted_idx, seeds, positions
    )
    if sampled is None:
        raise RuntimeError("The baseline Triton sampling kernel is unavailable")
    return sampled


def _time_us(fn, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--widths", default="50,64")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("batch width eager_us fused_us speedup mismatches")
    for batch_size in _parse_int_list(args.batch_sizes):
        for width in _parse_int_list(args.widths):
            generator = torch.Generator(device="cuda").manual_seed(
                batch_size * 1000 + width
            )
            scores = torch.randn(
                batch_size,
                width,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            sorted_scores, sorted_idx = torch.sort(scores, dim=-1, descending=True)
            top_ks = torch.randint(
                1,
                width + 1,
                (batch_size,),
                generator=generator,
                device="cuda",
                dtype=torch.long,
            )
            top_ps = (
                torch.rand(
                    batch_size,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                .mul_(0.9)
                .add_(0.05)
            )
            seeds = torch.randint(
                0,
                1 << 31,
                (batch_size,),
                generator=generator,
                device="cuda",
                dtype=torch.long,
            )
            positions = torch.arange(batch_size, device="cuda", dtype=torch.long)

            def eager() -> torch.Tensor:
                return _eager_post_topk(
                    sorted_scores, sorted_idx, top_ks, top_ps, seeds, positions
                )

            def fused() -> torch.Tensor:
                result = sample_from_sorted_scores_with_seed_small_k(
                    sorted_scores,
                    sorted_idx,
                    top_ks,
                    top_ps,
                    seeds,
                    positions,
                    has_top_p=True,
                )
                if result is None:
                    raise RuntimeError("The fused Triton kernel is unavailable")
                return result

            expected = eager()
            actual = fused()
            mismatches = int((actual != expected).sum().item())
            eager_us = statistics.median(_time_us(eager, args.warmup, args.repetitions))
            fused_us = statistics.median(_time_us(fused, args.warmup, args.repetitions))
            print(
                f"{batch_size:5d} {width:5d} {eager_us:8.2f} {fused_us:8.2f} "
                f"{eager_us / fused_us:7.2f}x {mismatches:10d}"
            )


if __name__ == "__main__":
    main()
