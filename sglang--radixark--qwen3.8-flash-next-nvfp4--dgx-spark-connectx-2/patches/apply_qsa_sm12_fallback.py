#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bind the exact SM120/SM121 direct-gather QSA kernel into SGLang."""

from __future__ import annotations

import hashlib
import pathlib


TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
    "qwen_sparse_attn_backend.py"
)
EXPECTED_SHA256 = "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2"
PATCHED_SHA256 = "8fe9277fe3a9ed41b5eced73283ecc5914fa0ecf67d4243b77923bf90e597b70"
KERNEL = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/qsa_sm121.py"
)
KERNEL_SHA256 = "5f012fb8d2fd8ef00cb411e1ebea228bc4c74b56e423b23a8fdfee17eda48907"
IMPORT_OLD = (
    "from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention\n"
)
IMPORT_NEW = IMPORT_OLD + (
    "from sglang.srt.layers.attention.qsa.qsa_sm121 import (\n"
    "    qsa_sm121_sparse_attention,\n"
    "    qsa_sm121_sparse_attention_graph,\n"
    ")\n"
)
OLD = """        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        trtllm_decode = _resolve_trtllm_sparse_decode()
"""
NEW = """        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        # FA4 CuTe cannot compile this shape on SM120/SM121. Consume physical
        # slots directly without materializing or repeating selected K/V. CUDA
        # graph metadata intentionally carries a one-column dummy slot table, so
        # resolve logical indices through the replay-refreshed request rows and
        # live req_to_token table inside the attention kernel instead.
        if torch.cuda.get_device_capability(q.device)[0] == 12:
            if metadata.is_cuda_graph:
                if metadata.row_req_pool_indices is None:
                    raise RuntimeError("QSA CUDA graph request rows are missing")
                output = qsa_sm121_sparse_attention_graph(
                    q,
                    k_buffer,
                    v_buffer,
                    topk_indices,
                    metadata.token_to_batch_idx,
                    metadata.row_req_pool_indices,
                    metadata.sequence_lengths,
                    self.req_to_token_pool.req_to_token,
                    layer.scaling,
                )
            else:
                slots = self._logical_to_physical(topk_indices, metadata)
                output = qsa_sm121_sparse_attention(
                    q, k_buffer, v_buffer, slots, layer.scaling
                )
            return output.reshape(q.shape[0], -1)
        trtllm_decode = _resolve_trtllm_sparse_decode()
"""


def main() -> int:
    if TARGET.is_symlink() or not TARGET.is_file():
        raise SystemExit("pinned QSA source is unavailable or unsafe")
    content = TARGET.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"pinned QSA source drifted: expected {EXPECTED_SHA256}, got {actual}"
        )
    if KERNEL.is_symlink() or not KERNEL.is_file():
        raise SystemExit("SM121 QSA kernel is unavailable or unsafe")
    kernel_actual = hashlib.sha256(KERNEL.read_bytes()).hexdigest()
    if kernel_actual != KERNEL_SHA256:
        raise SystemExit(
            f"SM121 QSA kernel drifted: expected {KERNEL_SHA256}, got {kernel_actual}"
        )
    text = content.decode("utf-8")
    if text.count(IMPORT_OLD) != 1 or IMPORT_NEW in text:
        raise SystemExit("pinned QSA import patch context is not unique")
    if text.count(OLD) != 1 or NEW in text:
        raise SystemExit("pinned QSA patch context is not unique")
    TARGET.write_text(
        text.replace(IMPORT_OLD, IMPORT_NEW).replace(OLD, NEW), encoding="utf-8"
    )
    patched = TARGET.read_text(encoding="utf-8")
    if patched.count(IMPORT_NEW) != 1 or patched.count(NEW) != 1 or OLD in patched:
        raise SystemExit("QSA SM12x fallback patch did not apply exactly")
    patched_sha256 = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if patched_sha256 != PATCHED_SHA256:
        raise SystemExit(
            f"patched QSA source drifted: expected {PATCHED_SHA256}, "
            f"got {patched_sha256}"
        )
    print(patched_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
