#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused SM121 compile, correctness, and latency check for QSA decode."""

from __future__ import annotations

import argparse
import json
import pathlib
import types

import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference


def load_kernel(path: pathlib.Path, block_n: int, num_warps: int):
    source = path.read_text()
    source = source.replace("BLOCK_N=64,", f"BLOCK_N={block_n},")
    source = source.replace("num_warps=4,", f"num_warps={num_warps},")
    module = types.ModuleType(f"letsinfer_qsa_sm121_{block_n}_{num_warps}")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def elapsed_ms(function, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", type=pathlib.Path)
    parser.add_argument("--block-n", type=int, default=64, choices=(16, 32, 64))
    parser.add_argument("--num-warps", type=int, default=4, choices=(4, 8))
    args = parser.parse_args()
    if torch.cuda.get_device_capability()[0] != 12:
        raise SystemExit("focused kernel check requires an SM12x GPU")

    torch.manual_seed(42042)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn((1, 12, 256), device=device, dtype=dtype)
    k = torch.randn((4096, 1, 256), device=device, dtype=dtype)
    v = torch.randn((4096, 1, 256), device=device, dtype=dtype)
    slots = torch.randperm(4096, device=device, dtype=torch.int64)[:2051].to(torch.int32)
    slots[-3:] = -1
    slots = slots.unsqueeze(0).contiguous()
    scale = 256**-0.5

    kernel = load_kernel(args.kernel, args.block_n, args.num_warps)
    actual = kernel.qsa_sm121_sparse_attention(q, k, v, slots, scale)
    expected = qsa_sparse_attention_reference(q, k, v, slots, scale)
    torch.cuda.synchronize()
    difference = (actual.float() - expected.float()).abs()
    if not torch.allclose(actual.float(), expected.float(), rtol=0.025, atol=0.02):
        raise AssertionError(
            f"SM121 QSA mismatch: max={difference.max().item()}, "
            f"mean={difference.mean().item()}"
        )

    invalid = torch.full((1, 2051), -1, dtype=torch.int32, device=device)
    invalid[0, 0] = k.shape[0]
    invalid_output = kernel.qsa_sm121_sparse_attention(q, k, v, invalid, scale)
    torch.cuda.synchronize()
    if torch.count_nonzero(invalid_output).item() != 0:
        raise AssertionError("invalid physical slots must produce an exact zero row")

    # The CUDA-graph path must resolve logical indices through replay-refreshed
    # request metadata. SGLang's graph metadata deliberately carries only a
    # one-column dummy token_slot_table, so using the eager mapper would turn
    # every selected index into slot zero.
    req_to_token = torch.stack(
        [
            torch.arange(4096, device=device, dtype=torch.int32),
            torch.randperm(4096, device=device, dtype=torch.int64).to(torch.int32),
            torch.randperm(4096, device=device, dtype=torch.int64).to(torch.int32),
        ]
    )
    logical = torch.arange(2051, device=device, dtype=torch.int32).unsqueeze(0)
    logical[0, -3:] = -1
    token_to_sequence = torch.zeros(1, device=device, dtype=torch.int32)
    row_req_pool_indices = torch.ones(1, device=device, dtype=torch.int32)
    row_sequence_lengths = torch.full(
        (1,), 2051, device=device, dtype=torch.int32
    )
    graph_actual = kernel.qsa_sm121_sparse_attention_graph(
        q,
        k,
        v,
        logical,
        token_to_sequence,
        row_req_pool_indices,
        row_sequence_lengths,
        req_to_token,
        scale,
    )
    graph_slots = req_to_token[1, logical.clamp_min(0)].clone()
    graph_slots[logical < 0] = -1
    graph_expected = qsa_sparse_attention_reference(q, k, v, graph_slots, scale)
    torch.cuda.synchronize()
    graph_difference = (graph_actual.float() - graph_expected.float()).abs()
    if not torch.allclose(
        graph_actual.float(), graph_expected.float(), rtol=0.025, atol=0.02
    ):
        raise AssertionError(
            f"SM121 logical QSA mismatch: max={graph_difference.max().item()}, "
            f"mean={graph_difference.mean().item()}"
        )

    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        replay_output = kernel.qsa_sm121_sparse_attention_graph(
            q,
            k,
            v,
            logical,
            token_to_sequence,
            row_req_pool_indices,
            row_sequence_lengths,
            req_to_token,
            scale,
        )
    row_req_pool_indices.fill_(2)
    capture.replay()
    torch.cuda.synchronize()
    replay_slots = req_to_token[2, logical.clamp_min(0)].clone()
    replay_slots[logical < 0] = -1
    replay_expected = qsa_sparse_attention_reference(q, k, v, replay_slots, scale)
    replay_difference = (replay_output.float() - replay_expected.float()).abs()
    if not torch.allclose(
        replay_output.float(), replay_expected.float(), rtol=0.025, atol=0.02
    ):
        raise AssertionError(
            "SM121 graph replay did not observe the updated request row: "
            f"max={replay_difference.max().item()}, "
            f"mean={replay_difference.mean().item()}"
        )

    for _ in range(5):
        kernel.qsa_sm121_sparse_attention(q, k, v, slots, scale)
    torch.cuda.synchronize()
    direct_ms = elapsed_ms(
        lambda: kernel.qsa_sm121_sparse_attention(q, k, v, slots, scale), 50
    )
    graph_direct_ms = elapsed_ms(
        lambda: kernel.qsa_sm121_sparse_attention_graph(
            q,
            k,
            v,
            logical,
            token_to_sequence,
            row_req_pool_indices,
            row_sequence_lengths,
            req_to_token,
            scale,
        ),
        50,
    )
    reference_ms = elapsed_ms(
        lambda: qsa_sparse_attention_reference(q, k, v, slots, scale), 10
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "shape": {
                    "q": list(q.shape),
                    "cache": list(k.shape),
                    "selected": slots.shape[1],
                },
                "block_n": args.block_n,
                "num_warps": args.num_warps,
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "graph_max_abs_error": graph_difference.max().item(),
                "graph_replay_max_abs_error": replay_difference.max().item(),
                "direct_ms": direct_ms,
                "graph_direct_ms": graph_direct_ms,
                "reference_ms": reference_ms,
                "speedup": reference_ms / direct_ms,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
