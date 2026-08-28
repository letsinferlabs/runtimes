#!/usr/bin/env python3
"""Validate the packaged deterministic HC kernel on GB10/SM121."""

from __future__ import annotations

import torch

from sglang.kernels.ops.qwen4_hc_mix_sm121 import (
    BLOCK_J,
    BLOCK_K,
    BLOCK_N,
    BLOCK_R,
    SPLITS,
    _deterministic_hc_mix_kernel,
    deterministic_hc_mix_sm121,
)
from sglang.srt.layers.hc_mix_triton import _hc_mix_persistent_kernel


HC = 4
HS = 2560
K = HC * HS
LOWRANK = 320
ROWS_PAD = 16
NUM_CTAS = 20


# Execute the former atomic kernel for a controlled output and timing reference.
def baseline(hidden, weight_down, weight_up, output, scratch, counters):
    _hc_mix_persistent_kernel[(NUM_CTAS,)](
        hidden,
        weight_down,
        weight_up,
        scratch,
        output,
        counters,
        K,
        LOWRANK,
        HS,
        hidden.shape[0],
        NUM_CTAS,
        1.0 / HC,
        ROWS=ROWS_PAD,
        HC=HC,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
    )
    return output


# Return a synchronized event-timed mean in microseconds.
def microseconds(operation) -> float:
    for _ in range(4):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(32):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / 32


# Require repeat stability, bounded reference delta, and a real timing win.
def main() -> int:
    torch.manual_seed(42042)
    device = torch.device("cuda")
    weight_down = torch.randn(
        (LOWRANK, K), dtype=torch.bfloat16, device=device
    )
    weight_up = torch.randn(
        (HC * HS, LOWRANK), dtype=torch.bfloat16, device=device
    )
    for rows in (1, 4):
        hidden = torch.randn((rows, K), dtype=torch.bfloat16, device=device)
        reference_output = torch.empty(
            (rows, HS), dtype=torch.bfloat16, device=device
        )
        scratch = torch.empty(
            (ROWS_PAD, LOWRANK), dtype=torch.float32, device=device
        )
        counters = torch.zeros(3, dtype=torch.int32, device=device)
        reference_operation = lambda: baseline(
            hidden,
            weight_down,
            weight_up,
            reference_output,
            scratch,
            counters,
        )
        candidate_operation = lambda: deterministic_hc_mix_sm121(
            hidden, weight_down, weight_up, HC, HS
        )
        reference = reference_operation().clone()
        candidate = candidate_operation().clone()
        torch.cuda.synchronize()
        for _ in range(8):
            torch.testing.assert_close(
                candidate_operation(), candidate, rtol=0, atol=0
            )
        difference = (candidate.float() - reference.float()).abs().max().item()
        if difference > 0.00390625:
            raise RuntimeError(f"deterministic HC reference delta changed: {difference}")
        candidate_output = torch.empty_like(reference_output)
        partials = torch.empty(
            (SPLITS, ROWS_PAD, LOWRANK), dtype=torch.float32, device=device
        )
        candidate_counters = torch.zeros(2, dtype=torch.int32, device=device)

        def candidate_kernel_operation():
            _deterministic_hc_mix_kernel[(NUM_CTAS,)](
                hidden,
                weight_down,
                weight_up,
                partials,
                candidate_output,
                candidate_counters,
                K,
                LOWRANK,
                HS,
                rows,
                NUM_CTAS,
                1.0 / HC,
                ROWS_CONST=ROWS_PAD,
                HC_CONST=HC,
                SPLITS_CONST=SPLITS,
                BLOCK_N_CONST=BLOCK_N,
                BLOCK_K_CONST=BLOCK_K,
                BLOCK_J_CONST=BLOCK_J,
                BLOCK_R_CONST=BLOCK_R,
                num_warps=8,
            )
            return candidate_output

        reference_us = microseconds(reference_operation)
        candidate_us = microseconds(candidate_kernel_operation)
        if candidate_us >= reference_us * 0.9:
            raise RuntimeError(
                f"deterministic HC timing regressed: {candidate_us} >= {reference_us}"
            )
        print(
            f"rows={rows} reference_us={reference_us:.6f} "
            f"candidate_us={candidate_us:.6f} max_abs={difference:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
