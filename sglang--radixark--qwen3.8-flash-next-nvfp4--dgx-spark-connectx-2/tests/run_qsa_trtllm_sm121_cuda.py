#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused SM121 probe for SGLang's TRT-LLM sparse QSA decode kernel."""

from __future__ import annotations

import json
import math

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    _resolve_trtllm_sparse_decode,
)


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if k.shape[2] == 1 and q.shape[1] != 1:
        k = k.expand(k.shape[0], k.shape[1], q.shape[1], k.shape[3])
        v = v.expand(v.shape[0], v.shape[1], q.shape[1], v.shape[3])
    scores = torch.einsum("bhd,bkhd->bhk", q.float(), k.float()) / math.sqrt(
        q.shape[-1]
    )
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhk,bkhd->bhd", probabilities, v.float()).to(q.dtype)


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise SystemExit("SM120/SM121 CUDA device is required")
    torch.manual_seed(42042)
    device = torch.device("cuda")
    batch, query_heads, kv_heads, page, head_dim = 1, 12, 1, 64, 256
    query = torch.randn(batch, query_heads, head_dim, device=device, dtype=torch.bfloat16)
    keys = torch.randn(batch, page, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    kv_cache = (
        keys.view(-1, page, kv_heads, head_dim).permute(0, 2, 1, 3),
        values.view(-1, page, kv_heads, head_dim).permute(0, 2, 1, 3),
    )
    block_tables = torch.zeros((batch, 1), dtype=torch.int32, device=device)
    sequence_lengths = torch.full((batch,), page, dtype=torch.int32, device=device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    decode = _resolve_trtllm_sparse_decode()
    if decode is None:
        raise SystemExit("SM121 TRT-LLM sparse decode did not resolve")

    def run() -> torch.Tensor:
        return decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=sequence_lengths,
            max_seq_len=page,
            bmm1_scale=head_dim**-0.5,
            bmm2_scale=1.0,
        )

    for _ in range(3):
        output = run()
    torch.cuda.synchronize()
    expected = reference(query, keys, values)
    direct_error = float((output - expected).abs().max())
    if direct_error > 0.04:
        raise SystemExit(f"TRT-LLM QSA error is too large: {direct_error}")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    query.copy_(torch.randn_like(query))
    graph.replay()
    torch.cuda.synchronize()
    graph_expected = reference(query, keys, values)
    graph_error = float((graph_output - graph_expected).abs().max())
    if graph_error > 0.04:
        raise SystemExit(f"TRT-LLM QSA graph replay error is too large: {graph_error}")
    print(
        json.dumps(
            {
                "direct_max_abs_error": direct_error,
                "graph_max_abs_error": graph_error,
                "shape": [batch, query_heads, page, head_dim],
                "trtllm_sparse_decode": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
