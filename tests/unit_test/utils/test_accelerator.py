# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.utils.accelerator import (
    AcceleratorPlatform,
    detect_accelerator_platform,
    is_rocm,
    resolve_gpu_visibility,
    supports_nvidia_cuda_ipc,
    visibility_env_keys,
)


def _torch_build(*, cuda: str | None = None, hip: str | None = None):
    return SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=hip))


def test_detects_rocm_before_cuda_compatibility_surface() -> None:
    # Some ROCm builds may expose CUDA compatibility metadata. HIP is the
    # authoritative discriminator when both attributes are present.
    torch = _torch_build(cuda="compat", hip="7.1.0")

    assert detect_accelerator_platform(torch) is AcceleratorPlatform.AMD
    assert is_rocm(torch)
    assert not supports_nvidia_cuda_ipc(torch)


def test_detects_nvidia_cuda_build() -> None:
    torch = _torch_build(cuda="13.0")

    assert detect_accelerator_platform(torch) is AcceleratorPlatform.NVIDIA
    assert not is_rocm(torch)
    assert supports_nvidia_cuda_ipc(torch)


def test_detects_device_agnostic_build() -> None:
    torch = _torch_build()

    assert detect_accelerator_platform(torch) is AcceleratorPlatform.NONE
    assert not supports_nvidia_cuda_ipc(torch)


def test_rocm_visibility_prefers_rocr_and_accepts_matching_aliases() -> None:
    platform = AcceleratorPlatform.AMD

    assert visibility_env_keys(platform)[0] == "ROCR_VISIBLE_DEVICES"
    assert resolve_gpu_visibility(
        platform,
        {"ROCR_VISIBLE_DEVICES": " 2,3 ", "CUDA_VISIBLE_DEVICES": "2,3"},
    ) == ("ROCR_VISIBLE_DEVICES", "2,3")
