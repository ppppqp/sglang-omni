#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify ROCm transport and optional multi-GPU collective correctness."""

from __future__ import annotations

import asyncio
import os

import torch

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter
from sglang_omni.comm.stage_io import read_payload, write_payload
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.accelerator import AcceleratorPlatform, is_rocm


async def _verify_shm_round_trip(rank: int) -> None:
    router = CommRouter(
        stage_name=f"source-{rank}",
        gpu_id=rank,
        same_process_targets=set(),
        gpu_stage_names={"target"},
        accelerator_platform=AcceleratorPlatform.AMD,
    )
    expected = torch.arange(64, dtype=torch.float32, device="cuda:0") + rank
    payload = StagePayload(
        request_id=f"rocm-runtime-{rank}",
        request=OmniRequest(inputs=None),
        data={"tensor": expected},
    )
    kind, relay = router.relay_for_payload("target", payload)
    if kind is not TransportKind.SHM:
        raise AssertionError(f"ROCm GPU edge selected {kind.value}, expected shm")

    data_ref, put_op = await write_payload(
        relay,
        payload.request_id,
        payload,
        transport=kind,
        from_stage=router.stage_name,
        to_stage="target",
    )
    restored = await read_payload(relay, payload.request_id, data_ref)
    put_op.mark_receiver_done()
    await put_op.wait_for_completion()
    actual = restored.data["tensor"]
    if actual.device.type != "cuda" or not torch.equal(actual, expected):
        raise AssertionError("GPU -> SHM -> GPU payload round trip changed the tensor")
    router.close()


def main() -> int:
    if not is_rocm():
        raise RuntimeError("verify_runtime.py requires a ROCm PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("no AMD GPU is visible")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(value)
        expected_sum = world_size * (world_size + 1) / 2
        if value.item() != expected_sum:
            raise AssertionError(
                f"RCCL all-reduce returned {value.item()}, expected {expected_sum}"
            )

    asyncio.run(_verify_shm_round_trip(rank))
    if world_size > 1:
        torch.distributed.destroy_process_group()
    print(f"rank={rank}: ROCm runtime transport probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
