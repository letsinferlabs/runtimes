#!/usr/bin/env python3
"""Tune exact HC mixing and add an opt-in padded BF16 M=1 dispatch on SM121."""

from __future__ import annotations

import hashlib
import pathlib


ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt")
PATCHES = {
    ROOT / "layers/hc_mix_triton.py": {
        "before": "4cca5abd2a9c343373d4abf851c0759aa6bd81d24f70830e46113ccaca1f8a4d",
        "after": "ce38cce8c5eaafc6141e2a6cb296bf788a5cffa2bce0e358cc68145a827ac95c",
        "replacements": (
            ("        BLOCK_N=32,\n", "        BLOCK_N=64,\n"),
            ("        BLOCK_J=32,\n", "        BLOCK_J=64,\n"),
        ),
    },
    ROOT / "layers/quantization/unquant.py": {
        "before": "aa6fc33c179fc1ea7db5ea1ec2e132c3aca7cbff6d55c436dfd817b3c61bd1c1",
        "after": "e73f9ad48375266b85d498bed684dd8f0ef780fe69ca1b4570045e0a2bb8cb8a",
        "replacements": (
            (
                '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip\n',
                '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip\n'
                '_pad_sm121_bf16_m1 = get_bool_env_var("SGLANG_SM121_PAD_BF16_M1")\n'
                '_padded_sm121_bf16_shapes = {(8320, 2560), (10240, 2560)}\n',
            ),
            (
                "    return F.linear(x, weight, bias)\n\n\n"
                "def get_bf16_gemm_backend() -> Bf16GemmBackend:\n",
                "    if (\n"
                "        _pad_sm121_bf16_m1\n"
                "        and m == 1\n"
                "        and x.ndim == 2\n"
                "        and x.dtype == torch.bfloat16\n"
                "        and weight.dtype == torch.bfloat16\n"
                "        and tuple(weight.shape) in _padded_sm121_bf16_shapes\n"
                "    ):\n"
                "        padded = torch.cat((x, torch.zeros_like(x)), dim=0)\n"
                "        return F.linear(padded, weight, bias)[:1]\n"
                "    return F.linear(x, weight, bias)\n\n\n"
                "def get_bf16_gemm_backend() -> Bf16GemmBackend:\n",
            ),
        ),
    },
    ROOT / "layers/logits_processor.py": {
        "before": "8df0b86bcf2dfc2bea4f2bae82433860ed88233616c923a5443602bca9378a11",
        "after": "4cf6da3c414a78a6d9c0f63a7e9eb04b9812f30846d86751816c54118d8ded2a",
        "replacements": (
            (
                "from sglang.srt.utils.common import (\n    is_cpu,\n",
                "from sglang.srt.utils.common import (\n"
                "    get_bool_env_var,\n"
                "    is_cpu,\n",
            ),
            (
                "_is_npu = is_npu()\n_is_cpu = is_cpu()\n",
                "_is_npu = is_npu()\n"
                "_is_cpu = is_cpu()\n"
                '_pad_sm121_bf16_m1 = get_bool_env_var("SGLANG_SM121_PAD_BF16_M1")\n',
            ),
            (
                "            elif self.rl_on_policy_target is not None:\n"
                "                # Due to tie-weight, we may not be able to change lm_head's weight dtype\n"
                "                logits = torch.matmul(\n"
                "                    hidden_states.bfloat16(), lm_head.weight.T.bfloat16()\n"
                "                )\n"
                "            else:\n",
                "            elif self.rl_on_policy_target is not None:\n"
                "                # Due to tie-weight, we may not be able to change lm_head's weight dtype\n"
                "                logits = torch.matmul(\n"
                "                    hidden_states.bfloat16(), lm_head.weight.T.bfloat16()\n"
                "                )\n"
                "            elif (\n"
                "                _pad_sm121_bf16_m1\n"
                "                and hidden_states.ndim == 2\n"
                "                and hidden_states.shape == (1, 2560)\n"
                "                and hidden_states.dtype == torch.bfloat16\n"
                "                and lm_head.weight.shape == (124160, 2560)\n"
                "                and lm_head.weight.dtype == torch.bfloat16\n"
                "            ):\n"
                "                padded = torch.cat(\n"
                "                    (hidden_states, torch.zeros_like(hidden_states)), dim=0\n"
                "                )\n"
                "                logits = torch.matmul(padded, lm_head.weight.T)[:1]\n"
                "            else:\n",
            ),
        ),
    },
}


# Return one file's exact byte identity.
def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Apply every unique anchor and reject source or result drift.
def apply() -> None:
    for path, contract in PATCHES.items():
        if sha256(path) != contract["before"]:
            raise SystemExit(f"SM121 BF16 patch preimage changed: {path}")
        source = path.read_text(encoding="utf-8")
        for before, after in contract["replacements"]:
            if source.count(before) != 1:
                raise SystemExit(f"SM121 BF16 patch anchor changed: {path}")
            source = source.replace(before, after)
        path.write_text(source, encoding="utf-8")
        if sha256(path) != contract["after"]:
            raise SystemExit(f"SM121 BF16 patch result changed: {path}")


if __name__ == "__main__":
    apply()
