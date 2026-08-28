#!/usr/bin/env python3
"""Apply the verified SM120 live-M=9 cuBLASLt tactic correction."""

from __future__ import annotations

import hashlib
import pathlib
import sys


PREIMAGE_SHA256 = "6c17e99e93f34fb7d51908d87719fca5cf5a89ebbe88af594d2ccc1773db7213"
POSTIMAGE_SHA256 = "0b70932864f6d0225388b51742f6b8324d0e7c0b99d431b4bb2467dbeacb26c2"

BEFORE = """                a, b, scale_a, scale_b, out, workspace_buffer = inputs
                with torch.cuda.device(a.device):
                    cublas_handle = torch.cuda.current_blas_handle()
                # The cuBLASLt algo list is enumerated per-shape, so a tactic
"""

AFTER = """                a, b, scale_a, scale_b, out, workspace_buffer = inputs
                with torch.cuda.device(a.device):
                    cublas_handle = torch.cuda.current_blas_handle()

                # The generic tuner profiles dynamic M at a power-of-two bucket,
                # but cuBLASLt enumerates a shape-specific algorithm list.  Reusing
                # the bucket's integer index for live M=9 therefore selects the
                # wrong algorithms on SM120.  These exact live-shape choices were
                # exhaustively screened from that M=9 list; keep K=6144/N=5120 on
                # tactic 0 because tactic 1 changes near-tied greedy logits.
                if (
                    torch.cuda.get_device_capability(a.device)[0] == 12
                    and a.shape[-2] == 9
                ):
                    tactic = {
                        (5120, 14336): 0,
                        (5120, 16384): 1,
                        (6144, 5120): 0,
                    }.get((a.shape[-1], b.shape[-1]), tactic)

                # The cuBLASLt algo list is enumerated per-shape, so a tactic
"""


def digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def patch_source(source: str) -> str:
    if source.count(BEFORE) != 1:
        raise RuntimeError(
            f"expected one FlashInfer FP8 tactic anchor, found {source.count(BEFORE)}"
        )
    return source.replace(BEFORE, AFTER)


def apply(path: pathlib.Path) -> None:
    source = path.read_text()
    if digest(source) != PREIMAGE_SHA256:
        raise RuntimeError(
            f"FlashInfer FP8 preimage mismatch: {digest(source)} != {PREIMAGE_SHA256}"
        )
    patched = patch_source(source)
    if digest(patched) != POSTIMAGE_SHA256:
        raise RuntimeError(
            f"FlashInfer FP8 postimage mismatch: {digest(patched)} != {POSTIMAGE_SHA256}"
        )
    path.write_text(patched)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_flashinfer_fp8_m9.py GEMM_BASE_PATH")
    apply(pathlib.Path(sys.argv[1]))
