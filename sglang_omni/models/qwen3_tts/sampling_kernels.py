# SPDX-License-Identifier: Apache-2.0
"""Optional sampling kernels for Qwen3-TTS."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rotl32(x, r: tl.constexpr) -> tl.uint32:
        x = x.to(tl.uint64)
        return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF

    @triton.jit
    def _fmix32(h: tl.uint32) -> tl.uint32:
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
        return h

    @triton.jit
    def _murmur3_mix(h: tl.uint32, k: tl.uint32) -> tl.uint32:
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = _rotl32(k, 15)
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
        h = _rotl32(h, 13)
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
        return h

    @triton.jit
    def _seeded_gumbel_sample_sorted_kernel(
        probs,
        sorted_idx,
        seeds,
        positions,
        out,
        num_cols: tl.constexpr,
        probs_stride_b: tl.constexpr,
        probs_stride_k: tl.constexpr,
        idx_stride_b: tl.constexpr,
        idx_stride_k: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < num_cols

        seed = tl.load(seeds + row).to(tl.uint64)
        pos = tl.load(positions + row).to(tl.uint32)
        col = offsets.to(tl.uint32)

        h: tl.uint32 = 0
        h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, pos)
        h = _murmur3_mix(h, col)
        h ^= 16
        h = _fmix32(h)

        u = h.to(tl.float64) / 4294967295.0
        u = tl.maximum(u, 2.2250738585072014e-308)
        gumbel = -tl.log(-tl.log(u))
        weights = tl.load(
            probs + row * probs_stride_b + offsets * probs_stride_k,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float64)
        scores = tl.where(mask, weights + gumbel, -float("inf"))
        max_score = tl.max(scores, axis=0)
        candidates = tl.where(scores == max_score, offsets, num_cols)
        rank = tl.min(candidates, axis=0)
        token = tl.load(sorted_idx + row * idx_stride_b + rank * idx_stride_k)
        tl.store(out + row, token)

    @triton.jit
    def _filter_and_seeded_gumbel_sample_sorted_kernel(
        sorted_scores,
        sorted_idx,
        top_ks,
        top_ps,
        seeds,
        positions,
        out,
        num_cols: tl.constexpr,
        scores_stride_b: tl.constexpr,
        scores_stride_k: tl.constexpr,
        idx_stride_b: tl.constexpr,
        idx_stride_k: tl.constexpr,
        block_size: tl.constexpr,
        has_top_p: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        valid_col = offsets < num_cols
        top_k = tl.load(top_ks + row)
        keep_top_k = valid_col & ((top_k <= 0) | (offsets < top_k))

        scores = tl.load(
            sorted_scores + row * scores_stride_b + offsets * scores_stride_k,
            mask=valid_col,
            other=-float("inf"),
        ).to(tl.float32)
        scores = tl.where(keep_top_k, scores, -float("inf"))
        max_score = tl.max(scores, axis=0)
        exp_scores = tl.exp(scores - max_score)
        probs = exp_scores / tl.sum(exp_scores, axis=0)

        keep_top_p = keep_top_k
        if has_top_p:
            top_p = tl.load(top_ps + row).to(tl.float32)
            active_top_p = (top_p > 0.0) & (top_p < 1.0)
            cumulative = tl.cumsum(probs, axis=0)
            keep_top_p = keep_top_p & (
                (~active_top_p) | (offsets == 0) | (cumulative <= top_p)
            )

        seed = tl.load(seeds + row).to(tl.uint64)
        pos = tl.load(positions + row).to(tl.uint32)
        col = offsets.to(tl.uint32)

        h: tl.uint32 = 0
        h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, pos)
        h = _murmur3_mix(h, col)
        h ^= 16
        h = _fmix32(h)

        u = h.to(tl.float64) / 4294967295.0
        u = tl.maximum(u, 2.2250738585072014e-308)
        gumbel = -tl.log(-tl.log(u))
        sampled_scores = tl.where(
            keep_top_p, probs.to(tl.float64) + gumbel, -float("inf")
        )
        max_sampled_score = tl.max(sampled_scores, axis=0)
        candidates = tl.where(sampled_scores == max_sampled_score, offsets, num_cols)
        rank = tl.min(candidates, axis=0)
        token = tl.load(sorted_idx + row * idx_stride_b + rank * idx_stride_k)
        tl.store(out + row, token)

else:
    _seeded_gumbel_sample_sorted_kernel = None
    _filter_and_seeded_gumbel_sample_sorted_kernel = None


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def sample_from_sorted_probs_with_seed_small_k(
    probs: torch.Tensor,
    sorted_idx: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    if (
        _seeded_gumbel_sample_sorted_kernel is None
        or not probs.is_cuda
        or not sorted_idx.is_cuda
        or not seeds.is_cuda
        or not positions.is_cuda
    ):
        return None
    if probs.ndim != 2 or sorted_idx.shape != probs.shape:
        return None
    if seeds.ndim != 1 or positions.ndim != 1:
        return None
    batch_size, num_cols = probs.shape
    if batch_size == 0:
        return torch.empty((0,), device=probs.device, dtype=torch.long)
    if seeds.shape[0] != batch_size or positions.shape[0] != batch_size:
        return None
    if num_cols <= 0 or num_cols > 1024:
        return None

    block_size = _next_power_of_2(num_cols)
    out = torch.empty((batch_size,), device=probs.device, dtype=torch.long)
    _seeded_gumbel_sample_sorted_kernel[(batch_size,)](
        probs,
        sorted_idx,
        seeds,
        positions,
        out,
        int(num_cols),
        probs.stride(0),
        probs.stride(1),
        sorted_idx.stride(0),
        sorted_idx.stride(1),
        block_size,
    )
    return out


def sample_from_sorted_scores_with_seed_small_k(
    sorted_scores: torch.Tensor,
    sorted_idx: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    *,
    has_top_p: bool,
) -> torch.Tensor | None:
    """Filter and sample sorted bounded-top-k scores in one Triton kernel."""

    tensors = (sorted_scores, sorted_idx, top_ks, top_ps, seeds, positions)
    if _filter_and_seeded_gumbel_sample_sorted_kernel is None or any(
        not tensor.is_cuda for tensor in tensors
    ):
        return None
    if sorted_scores.ndim != 2 or sorted_idx.shape != sorted_scores.shape:
        return None
    batch_size, num_cols = sorted_scores.shape
    if any(tensor.ndim != 1 for tensor in tensors[2:]):
        return None
    if any(int(tensor.shape[0]) != batch_size for tensor in tensors[2:]):
        return None
    if batch_size == 0:
        return torch.empty((0,), device=sorted_scores.device, dtype=torch.long)
    if num_cols <= 0 or num_cols > 1024:
        return None
    if any(tensor.device != sorted_scores.device for tensor in tensors[1:]):
        return None
    if any(not tensor.is_contiguous() for tensor in tensors):
        return None

    block_size = _next_power_of_2(num_cols)
    out = torch.empty((batch_size,), device=sorted_scores.device, dtype=torch.long)
    _filter_and_seeded_gumbel_sample_sorted_kernel[(batch_size,)](
        sorted_scores,
        sorted_idx,
        top_ks,
        top_ps,
        seeds,
        positions,
        out,
        int(num_cols),
        sorted_scores.stride(0),
        sorted_scores.stride(1),
        sorted_idx.stride(0),
        sorted_idx.stride(1),
        block_size,
        has_top_p=bool(has_top_p),
    )
    return out
