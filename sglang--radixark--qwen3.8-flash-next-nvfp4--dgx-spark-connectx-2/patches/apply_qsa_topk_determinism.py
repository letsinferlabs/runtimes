#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Canonicalize QSA radix top-k output before sparse-attention reduction."""

from __future__ import annotations

import hashlib
import pathlib


TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/kernels/ops/elementwise/fast_topk.py"
)
EXPECTED_SHA256 = "77780478c7b48517fbe9240d62d8a71371203a1acea42d27d44022cc1e9863be"
PATCHED_SHA256 = "fb2b79ed0f8d41d17846e93f8e6df5100982fb1932951da478ec1689ed47cc76"

IMPORTS_OLD = """import torch

from sglang.kernels.jit.utils import (
"""
IMPORTS_NEW = """import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
"""
KERNEL_ANCHOR = """_FAST_TOPK_SUPPORTED_K = (512, 2048)


@cache_once
"""
KERNEL_INSERT = """_FAST_TOPK_SUPPORTED_K = (512, 2048)


@triton.jit
def _canonicalize_fast_topk_kernel(
    indices,
    row_stride: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, TOPK)
    values = tl.load(indices + row * row_stride + columns)
    values = tl.where(values >= 0, values, 0x7FFFFFFF)
    values = tl.sort(values)
    values = tl.where(values == 0x7FFFFFFF, -1, values)
    tl.store(indices + row * row_stride + columns, values)


@cache_once
"""
BODY_OLD = """    module = _jit_fast_topk_module(topk)
    module.fast_topk(score, row_starts, indices, lengths)
    return indices
"""
BODY_NEW = """    module = _jit_fast_topk_module(topk)
    module.fast_topk(score, row_starts, indices, lengths)
    # Atomic collection returns a stable selected set in unspecified order.
    # Sparse attention consumes that order as a reduction sequence, so make it
    # canonical before graph replay or an eager consumer observes the indices.
    _canonicalize_fast_topk_kernel[(batch,)](
        indices,
        indices.stride(0),
        TOPK=topk,
        num_warps=8,
    )
    return indices
"""


def main() -> int:
    if TARGET.is_symlink() or not TARGET.is_file():
        raise SystemExit("pinned fast_topk Python source is unavailable or unsafe")
    content = TARGET.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"pinned fast_topk source drifted: expected {EXPECTED_SHA256}, got {actual}"
        )
    text = content.decode("utf-8")
    replacements = (
        (IMPORTS_OLD, IMPORTS_NEW),
        (KERNEL_ANCHOR, KERNEL_INSERT),
        (BODY_OLD, BODY_NEW),
    )
    for old, new in replacements:
        if text.count(old) != 1 or new in text:
            raise SystemExit("pinned fast_topk patch context is not unique")
        text = text.replace(old, new)
    TARGET.write_text(text, encoding="utf-8")
    patched_sha256 = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if patched_sha256 != PATCHED_SHA256:
        raise SystemExit(
            f"patched fast_topk source drifted: expected {PATCHED_SHA256}, "
            f"got {patched_sha256}"
        )
    print(patched_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
