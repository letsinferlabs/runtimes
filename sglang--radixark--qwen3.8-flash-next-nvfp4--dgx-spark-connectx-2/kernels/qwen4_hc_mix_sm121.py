# SPDX-FileCopyrightText: 2026 Let's Infer contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic two-way split-K HC mixing for Qwen4-Exp on GB10/SM121."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS = 16
HC = 4
HS = 2560
K = HC * HS
LOWRANK = 320
SPLITS = 2
NUM_CTAS = 20
BLOCK_N = 32
BLOCK_K = 256
BLOCK_J = 64
BLOCK_R = 64


# Synchronize every resident CTA with one device-scope ticket counter.
@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


# Compute two fixed K partials and reduce them in a fixed order before HC up-mix.
@triton.jit
def _deterministic_hc_mix_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    partial_ptr,
    out_ptr,
    counters_ptr,
    K_RUNTIME,
    LOWRANK_RUNTIME,
    HS_RUNTIME,
    num_rows,
    num_ctas,
    inv_hc,
    ROWS_CONST: tl.constexpr,
    HC_CONST: tl.constexpr,
    SPLITS_CONST: tl.constexpr,
    BLOCK_N_CONST: tl.constexpr,
    BLOCK_K_CONST: tl.constexpr,
    BLOCK_J_CONST: tl.constexpr,
    BLOCK_R_CONST: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS_CONST)
    mask_m = offs_m < num_rows
    offs_n = tl.arange(0, BLOCK_N_CONST)
    offs_k = tl.arange(0, BLOCK_K_CONST)
    n_blocks = tl.cdiv(LOWRANK_RUNTIME, BLOCK_N_CONST)
    split = pid // n_blocks
    nb = pid % n_blocks
    n = nb * BLOCK_N_CONST + offs_n
    mask_n = n < LOWRANK_RUNTIME
    k_chunks = tl.cdiv(K_RUNTIME, BLOCK_K_CONST)
    down = tl.zeros((ROWS_CONST, BLOCK_N_CONST), dtype=tl.float32)
    for kc in range(split, k_chunks, SPLITS_CONST):
        k = kc * BLOCK_K_CONST + offs_k
        hidden = tl.load(
            x_ptr + offs_m[:, None] * K_RUNTIME + k[None, :],
            mask=mask_m[:, None] & (k[None, :] < K_RUNTIME),
            other=0.0,
        )
        weight = tl.load(
            w_down_ptr + n[:, None] * K_RUNTIME + k[None, :],
            mask=mask_n[:, None] & (k[None, :] < K_RUNTIME),
            other=0.0,
        )
        down = tl.dot(hidden, tl.trans(weight), down)
    partial_stride = ROWS_CONST * LOWRANK_RUNTIME
    tl.store(
        partial_ptr
        + split * partial_stride
        + offs_m[:, None] * LOWRANK_RUNTIME
        + n[None, :],
        down,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    _grid_barrier(counters_ptr, num_ctas)

    offs_j = tl.arange(0, BLOCK_J_CONST)
    offs_r = tl.arange(0, BLOCK_R_CONST)
    offs_g = tl.arange(0, HC_CONST)
    j_blocks = tl.cdiv(HS_RUNTIME, BLOCK_J_CONST)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J_CONST + offs_j
        mask_j = j < HS_RUNTIME
        gj = offs_g[:, None] * HS_RUNTIME + j[None, :]
        gj_flat = tl.reshape(gj, (HC_CONST * BLOCK_J_CONST,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC_CONST, BLOCK_J_CONST)),
            (HC_CONST * BLOCK_J_CONST,),
        )
        up = tl.zeros((ROWS_CONST, HC_CONST * BLOCK_J_CONST), dtype=tl.float32)
        for r0 in range(0, LOWRANK_RUNTIME, BLOCK_R_CONST):
            r = r0 + offs_r
            mask_r = r < LOWRANK_RUNTIME
            partial0 = tl.load(
                partial_ptr + offs_m[:, None] * LOWRANK_RUNTIME + r[None, :],
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0,
            )
            partial1 = tl.load(
                partial_ptr
                + partial_stride
                + offs_m[:, None] * LOWRANK_RUNTIME
                + r[None, :],
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0,
            )
            reduced = (partial0 + partial1) * inv_hc
            activated = (reduced * tl.sigmoid(reduced)).to(
                x_ptr.dtype.element_ty
            )
            weight = tl.load(
                w_up_ptr
                + gj_flat[:, None] * LOWRANK_RUNTIME
                + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            up = tl.dot(activated, tl.trans(weight), up)
        gate = tl.sigmoid(
            tl.reshape(up, (ROWS_CONST, HC_CONST, BLOCK_J_CONST))
        )
        grouped = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC_CONST * HS_RUNTIME)
            + offs_g[None, :, None] * HS_RUNTIME
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        output = tl.sum(gate * grouped, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS_RUNTIME + j[None, :],
            output.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 1, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr, 0)
        tl.store(counters_ptr + 1, 0)


_counters = {}


# Return one graph-stable counter pair for this CUDA device.
def _counters_for(device: torch.device) -> torch.Tensor:
    value = _counters.get(device)
    if value is None:
        value = torch.zeros(2, dtype=torch.int32, device=device)
        _counters[device] = value
    return value


# Execute the exact Qwen4-Exp HC shape or reject any drift.
def deterministic_hc_mix_sm121(
    hidden: torch.Tensor,
    weight_down: torch.Tensor,
    weight_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    if (
        not hidden.is_cuda
        or hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or tuple(hidden.shape[1:]) != (K,)
        or hidden.shape[0] not in range(0, ROWS + 1)
        or tuple(weight_down.shape) != (LOWRANK, K)
        or tuple(weight_up.shape) != (HC * HS, LOWRANK)
        or weight_down.dtype != hidden.dtype
        or weight_up.dtype != hidden.dtype
        or not weight_down.is_contiguous()
        or not weight_up.is_contiguous()
        or hc != HC
        or hs != HS
    ):
        raise RuntimeError("SM121 deterministic HC-mix shape changed")
    properties = torch.cuda.get_device_properties(hidden.device)
    if properties.major != 12 or properties.minor != 1:
        raise RuntimeError("SM121 deterministic HC mix requires compute capability 12.1")
    if properties.multi_processor_count < NUM_CTAS:
        raise RuntimeError("SM121 deterministic HC mix has too few resident CTAs")
    output = torch.empty(
        (hidden.shape[0], HS), dtype=hidden.dtype, device=hidden.device
    )
    if hidden.shape[0] == 0:
        return output
    partials = torch.empty(
        (SPLITS, ROWS, LOWRANK), dtype=torch.float32, device=hidden.device
    )
    _deterministic_hc_mix_kernel[(NUM_CTAS,)](
        hidden,
        weight_down,
        weight_up,
        partials,
        output,
        _counters_for(hidden.device),
        K,
        LOWRANK,
        HS,
        hidden.shape[0],
        NUM_CTAS,
        1.0 / HC,
        ROWS_CONST=ROWS,
        HC_CONST=HC,
        SPLITS_CONST=SPLITS,
        BLOCK_N_CONST=BLOCK_N,
        BLOCK_K_CONST=BLOCK_K,
        BLOCK_J_CONST=BLOCK_J,
        BLOCK_R_CONST=BLOCK_R,
        num_warps=8,
    )
    return output
