#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Enable SGLang's graph-safe TRT-LLM sparse QSA decode on SM120/SM121."""

from __future__ import annotations

import hashlib
import pathlib


TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
    "qwen_sparse_attn_backend.py"
)
EXPECTED_SHA256 = "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2"
PATCHED_SHA256 = "a6b003ed21b3be8ba763e8627aee39baee3d84184f5bf0fc650a1a6b853119d3"
OLD = """    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
"""
NEW = """    from sglang.srt.utils import is_sm100_supported, is_sm120_supported

    if not (is_sm100_supported() or is_sm120_supported()):
"""


def main() -> int:
    if TARGET.is_symlink() or not TARGET.is_file():
        raise SystemExit(f"pinned QSA source is unavailable or unsafe: {TARGET}")
    content = TARGET.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"pinned QSA source drifted: expected {EXPECTED_SHA256}, got {actual}"
        )
    source = content.decode("utf-8")
    if source.count(OLD) != 1 or NEW in source:
        raise SystemExit("TRT-LLM SM121 patch context is not unique")
    TARGET.write_text(source.replace(OLD, NEW), encoding="utf-8")
    actual = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if actual != PATCHED_SHA256:
        raise SystemExit(
            f"patched QSA source drifted: expected {PATCHED_SHA256}, got {actual}"
        )
    print(actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
