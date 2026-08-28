#!/usr/bin/env python3
"""Delegate Qwen's fused HC mix to the deterministic SM121 implementation."""

from __future__ import annotations

import hashlib
import pathlib


PATH = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/hc_mix_triton.py"
)
EXPECTED_SHA256 = "ce38cce8c5eaafc6141e2a6cb296bf788a5cffa2bce0e358cc68145a827ac95c"
PATCHED_SHA256 = "6a425e0b0e8f526a45b8b47bd555353ee00996c3e0e534b72083636a5be032e9"
OLD = '''    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty(
        (rows, hs), dtype=hyper_input_normed.dtype, device=device
    )
    if rows == 0:
        return out
    _hc_mix_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        t_raw,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=64,
        BLOCK_K=256,
        BLOCK_J=64,
        BLOCK_R=64,
        num_warps=8,
    )
    return out
'''
NEW = '''    from sglang.kernels.ops.qwen4_hc_mix_sm121 import (
        deterministic_hc_mix_sm121,
    )

    return deterministic_hc_mix_sm121(
        hyper_input_normed, w_down, w_up, hc, hs
    )
'''


# Return the exact byte identity of the upstream module.
def sha256() -> str:
    return hashlib.sha256(PATH.read_bytes()).hexdigest()


# Replace one exact function body and reject all source or result drift.
def apply() -> None:
    if sha256() != EXPECTED_SHA256:
        raise SystemExit("deterministic HC patch preimage changed")
    source = PATH.read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise SystemExit("deterministic HC patch anchor changed")
    PATH.write_text(source.replace(OLD, NEW), encoding="utf-8")
    if sha256() != PATCHED_SHA256:
        raise SystemExit("deterministic HC patch result changed")


if __name__ == "__main__":
    apply()
