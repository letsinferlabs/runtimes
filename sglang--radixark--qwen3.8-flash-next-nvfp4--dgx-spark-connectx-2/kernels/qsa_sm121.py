# SPDX-License-Identifier: Apache-2.0
"""Direct-gather sparse GQA decode for NVIDIA SM120/SM121.

The upstream QSA fallback materializes selected K/V tensors for every query
row, repeats the single KV head for every GQA head, and then launches Torch
einsums. This kernel consumes either a physical slot table or graph-stable
logical indices plus the live request-to-token table. It performs an online
softmax, so selected K/V data and graph-time slots are never materialized or
repeated.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_sm121_sparse_attention_kernel(
    q,
    k_cache,
    v_cache,
    token_indices,
    token_to_sequence,
    row_req_pool_indices,
    row_sequence_lengths,
    req_to_token,
    output,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    indices_stride_row: tl.constexpr,
    indices_stride_token: tl.constexpr,
    token_to_sequence_stride: tl.constexpr,
    row_req_pool_stride: tl.constexpr,
    row_sequence_length_stride: tl.constexpr,
    req_to_token_stride_row: tl.constexpr,
    req_to_token_stride_token: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    softmax_scale,
    CACHE_TOKENS: tl.constexpr,
    METADATA_ROWS: tl.constexpr,
    REQUEST_ROWS: tl.constexpr,
    REQUEST_WIDTH: tl.constexpr,
    QUERY_HEADS_PER_KV: tl.constexpr,
    SELECTED_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_LOGICAL_INDICES: tl.constexpr,
):
    row = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // QUERY_HEADS_PER_KV
    dims = tl.arange(0, BLOCK_D)

    query = tl.load(
        q
        + row * q_stride_row
        + query_head * q_stride_head
        + dims * q_stride_dim,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)

    if USE_LOGICAL_INDICES:
        sequence = tl.load(
            token_to_sequence + row * token_to_sequence_stride
        ).to(tl.int64)
        valid_sequence = (sequence >= 0) & (sequence < METADATA_ROWS)
        safe_sequence = tl.where(valid_sequence, sequence, 0)
        request = tl.load(
            row_req_pool_indices + safe_sequence * row_req_pool_stride
        ).to(tl.int64)
        sequence_length = tl.load(
            row_sequence_lengths
            + safe_sequence * row_sequence_length_stride
        ).to(tl.int64)
        valid_request = valid_sequence & (request >= 0) & (request < REQUEST_ROWS)
        safe_request = tl.where(valid_request, request, 0)

    # Finite sentinels keep an all-invalid chunk from evaluating -inf - -inf.
    running_max = tl.full((), -1.0e30, tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)

    for start_n in tl.range(0, SELECTED_TOKENS, BLOCK_N, num_stages=2):
        selected = start_n + tl.arange(0, BLOCK_N)
        indices = tl.load(
            token_indices
            + row * indices_stride_row
            + selected * indices_stride_token,
            mask=selected < SELECTED_TOKENS,
            other=-1,
        ).to(tl.int64)

        if USE_LOGICAL_INDICES:
            valid = (
                (selected < SELECTED_TOKENS)
                & valid_request
                & (indices >= 0)
                & (indices < sequence_length)
                & (indices < REQUEST_WIDTH)
            )
            safe_indices = tl.where(valid, indices, 0)
            slots = tl.load(
                req_to_token
                + safe_request * req_to_token_stride_row
                + safe_indices * req_to_token_stride_token,
                mask=valid,
                other=-1,
            ).to(tl.int64)
            valid = valid & (slots >= 0) & (slots < CACHE_TOKENS)
        else:
            slots = indices
            valid = (
                (selected < SELECTED_TOKENS)
                & (slots >= 0)
                & (slots < CACHE_TOKENS)
            )
        safe_slots = tl.where(valid, slots, 0)

        keys = tl.load(
            k_cache
            + safe_slots[:, None] * k_stride_token
            + kv_head * k_stride_head
            + dims[None, :] * k_stride_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * softmax_scale
        scores = tl.where(valid, scores, -1.0e30)

        chunk_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, chunk_max)
        old_scale = tl.exp2((running_max - next_max) * 1.4426950408889634)
        probabilities = tl.where(
            valid,
            tl.exp2((scores - next_max) * 1.4426950408889634),
            0.0,
        )

        values = tl.load(
            v_cache
            + safe_slots[:, None] * v_stride_token
            + kv_head * v_stride_head
            + dims[None, :] * v_stride_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = next_max

    result = tl.where(running_sum > 0.0, accumulator / running_sum, 0.0)
    tl.store(
        output
        + row * out_stride_row
        + query_head * out_stride_head
        + dims * out_stride_dim,
        result,
        mask=dims < HEAD_DIM,
    )


def _validate_common(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_indices: torch.Tensor,
) -> int:
    if not all(tensor.is_cuda for tensor in (q, k_cache, v_cache, token_indices)):
        raise ValueError("SM121 QSA attention requires CUDA tensors")
    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    if token_indices.ndim != 2 or token_indices.shape[0] != q.shape[0]:
        raise ValueError("token indices must be [query_tokens, selected_tokens]")
    if k_cache.shape != v_cache.shape or q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("Q/K/V cache shapes are incompatible")
    if q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if q.dtype not in (torch.float16, torch.bfloat16) or q.dtype != k_cache.dtype:
        raise ValueError("SM121 QSA requires matching FP16 or BF16 Q/K/V")
    if v_cache.dtype != q.dtype or token_indices.dtype != torch.int32:
        raise ValueError("SM121 QSA requires matching Q/K/V and int32 indices")
    head_dim = q.shape[-1]
    if head_dim <= 0 or head_dim > 256:
        raise ValueError(f"unsupported SM121 QSA head dimension: {head_dim}")
    return head_dim


def qsa_sm121_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run direct-gather sparse GQA attention for physical token slots."""

    head_dim = _validate_common(q, k_cache, v_cache, token_slots)
    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    scale = float(softmax_scale) if softmax_scale is not None else head_dim**-0.5
    _qsa_sm121_sparse_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        k_cache,
        v_cache,
        token_slots,
        token_slots,
        token_slots,
        token_slots,
        token_slots,
        output,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *token_slots.stride(),
        token_slots.stride(0),
        token_slots.stride(0),
        token_slots.stride(0),
        token_slots.stride(0),
        token_slots.stride(1),
        *output.stride(),
        scale,
        CACHE_TOKENS=k_cache.shape[0],
        METADATA_ROWS=1,
        REQUEST_ROWS=1,
        REQUEST_WIDTH=1,
        QUERY_HEADS_PER_KV=q.shape[1] // k_cache.shape[1],
        SELECTED_TOKENS=token_slots.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_N=64,
        BLOCK_D=triton.next_power_of_2(head_dim),
        USE_LOGICAL_INDICES=False,
        num_warps=4,
    )
    return output


def qsa_sm121_sparse_attention_graph(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    token_to_sequence: torch.Tensor,
    row_req_pool_indices: torch.Tensor,
    row_sequence_lengths: torch.Tensor,
    req_to_token: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run graph-stable QSA attention without materializing physical slots."""

    head_dim = _validate_common(q, k_cache, v_cache, logical_indices)
    metadata = (
        token_to_sequence,
        row_req_pool_indices,
        row_sequence_lengths,
        req_to_token,
    )
    if not all(tensor.is_cuda for tensor in metadata):
        raise ValueError("SM121 graph QSA metadata must be CUDA tensors")
    if token_to_sequence.ndim != 1 or token_to_sequence.shape[0] != q.shape[0]:
        raise ValueError("token-to-sequence must contain one entry per query")
    if row_req_pool_indices.ndim != 1 or row_sequence_lengths.ndim != 1:
        raise ValueError("graph row metadata must be rank-1")
    if row_req_pool_indices.shape != row_sequence_lengths.shape:
        raise ValueError("request rows and sequence lengths must have matching shapes")
    if req_to_token.ndim != 2:
        raise ValueError("request-to-token table must be rank-2")
    if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in metadata):
        raise ValueError("SM121 graph QSA metadata must be int32 or int64")

    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    scale = float(softmax_scale) if softmax_scale is not None else head_dim**-0.5
    _qsa_sm121_sparse_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        k_cache,
        v_cache,
        logical_indices,
        token_to_sequence,
        row_req_pool_indices,
        row_sequence_lengths,
        req_to_token,
        output,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *logical_indices.stride(),
        token_to_sequence.stride(0),
        row_req_pool_indices.stride(0),
        row_sequence_lengths.stride(0),
        *req_to_token.stride(),
        *output.stride(),
        scale,
        CACHE_TOKENS=k_cache.shape[0],
        METADATA_ROWS=row_sequence_lengths.shape[0],
        REQUEST_ROWS=req_to_token.shape[0],
        REQUEST_WIDTH=req_to_token.shape[1],
        QUERY_HEADS_PER_KV=q.shape[1] // k_cache.shape[1],
        SELECTED_TOKENS=logical_indices.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_N=64,
        BLOCK_D=triton.next_power_of_2(head_dim),
        USE_LOGICAL_INDICES=True,
        num_warps=4,
    )
    return output


__all__ = ["qsa_sm121_sparse_attention", "qsa_sm121_sparse_attention_graph"]
