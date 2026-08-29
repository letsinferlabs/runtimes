#!/usr/bin/env python3
"""Focused source and protocol-boundary tests for the RTX 5090 Engine."""

from __future__ import annotations

import ast
import importlib.util
from importlib.machinery import SourceFileLoader
import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


CANDIDATE = pathlib.Path(__file__).resolve().parents[1]
ADAPTER = CANDIDATE / "adapter"
AMD64_BASE = "sha256:7d81a16a2ebbe104be37263b49f1bfc0a6a95c1d57e500875dbe7f7c41301e45"
TRITON_WHEEL_SHA256 = (
    "74caf5e34b66d9f3a429af689c1c7128daba1d8208df60e81106b115c00d6fca"
)
FLASHINFER_FP8_PREIMAGE_SHA256 = (
    "6c17e99e93f34fb7d51908d87719fca5cf5a89ebbe88af594d2ccc1773db7213"
)
FLASHINFER_FP8_POSTIMAGE_SHA256 = (
    "0b70932864f6d0225388b51742f6b8324d0e7c0b99d431b4bb2467dbeacb26c2"
)


def load_adapter():
    sys.path.insert(0, str(ADAPTER))
    try:
        name = "qwen38_rtx5090_engine_adapter"
        loader = SourceFileLoader(name, str(ADAPTER / "engine-adapter"))
        spec = importlib.util.spec_from_loader(name, loader)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, importlib.import_module("engine_frontend")
    finally:
        sys.path.remove(str(ADAPTER))


class Rtx5090EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = json.loads((CANDIDATE / "runtime.json").read_text())
        cls.adapter, cls.frontend = load_adapter()

    def test_source_overlay_contains_no_compiled_extension(self) -> None:
        generated = [
            path.relative_to(CANDIDATE).as_posix()
            for path in (CANDIDATE / "engine").rglob("*")
            if path.is_file()
            and (path.suffix in {".pyc", ".so"} or ".so." in path.name)
        ]
        self.assertEqual(generated, [])

    def test_dockerfile_uses_exact_amd64_base(self) -> None:
        dockerfile = (CANDIDATE / "image" / "Dockerfile").read_text()
        self.assertEqual(dockerfile.count(AMD64_BASE), 2)
        self.assertNotIn("aarch64-linux-gnu", dockerfile)
        self.assertIn('sysconfig.get_path("platlib")', dockerfile)
        self.assertNotIn("/usr/local/lib/python3.12/dist-packages", dockerfile)
        self.assertIn(f"--checksum=sha256:{TRITON_WHEEL_SHA256}", dockerfile)
        self.assertIn("triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64", dockerfile)
        self.assertIn('triton.__version__ == "3.6.0"', dockerfile)
        self.assertIn("apply-flashinfer-fp8-m9.py", dockerfile)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1 python3 -c", dockerfile)
        self.assertIn("--invalidation-mode checked-hash -q", dockerfile)

    def test_flashinfer_fp8_patch_is_fail_closed_and_shape_bound(self) -> None:
        patch_path = (
            CANDIDATE
            / "engine"
            / "patches"
            / "apply_flashinfer_fp8_m9.py"
        )
        spec = importlib.util.spec_from_file_location(
            "qwen38_flashinfer_fp8_patch", patch_path
        )
        assert spec is not None and spec.loader is not None
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)
        self.assertEqual(patch.PREIMAGE_SHA256, FLASHINFER_FP8_PREIMAGE_SHA256)
        self.assertEqual(patch.POSTIMAGE_SHA256, FLASHINFER_FP8_POSTIMAGE_SHA256)
        patched = patch.patch_source(patch.BEFORE)
        self.assertIn("a.shape[-2] == 9", patched)
        self.assertIn("(5120, 14336): 0", patched)
        self.assertIn("(5120, 16384): 1", patched)
        self.assertIn("(6144, 5120): 0", patched)

    def test_sm120_linear_attention_tiles_preserve_the_measured_c1_shape(self) -> None:
        kernel_root = CANDIDATE / "engine/sglang/python/sglang/kernels/ops"
        recurrent = (
            kernel_root
            / "attention/fla/fused_sigmoid_gating_recurrent.py"
        ).read_text(encoding="utf-8")
        convolution = (
            kernel_root / "mamba/causal_conv1d_triton.py"
        ).read_text(encoding="utf-8")
        self.assertIn("min(triton.next_power_of_2(V), 8)", recurrent)
        self.assertEqual(convolution.count("BLOCK_N=128"), 1)
        self.assertIn(
            "BLOCK_N=128,\n        SAVE_INTERMEDIATE=", convolution
        )

    def test_flashinfer_unit_scale_fp8_write_is_strictly_guarded(self) -> None:
        path = (
            CANDIDATE
            / "engine/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py"
        )
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "FlashInferAttnBackend"
        )
        methods = {
            node.name: node
            for node in backend.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        helper = methods["_try_fused_unit_scale_fp8_kv_write"]
        helper_source = ast.get_source_segment(source, helper)
        assert helper_source is not None
        self.assertIn("layer.k_scale_float != 1.0", helper_source)
        self.assertIn("layer.v_scale_float != 1.0", helper_source)
        self.assertIn("self.prefill_uses_dequant_workspace", helper_source)
        self.assertIn(
            'getattr(self.token_to_kv_pool, "use_hnd", False)', helper_source
        )
        self.assertIn("fused_fp8_qkv_kv_cache(", helper_source)
        self.assertIn("return True", helper_source)

        extend_source = ast.get_source_segment(source, methods["forward_extend"])
        assert extend_source is not None
        self.assertIn(
            "if not self._try_fused_unit_scale_fp8_kv_write(", extend_source
        )
        self.assertIn("self.token_to_kv_pool.set_kv_buffer(", extend_source)

    def test_qwen35_language_model_only_is_explicitly_whitelisted(self) -> None:
        server_args = (
            CANDIDATE
            / "engine/sglang/python/sglang/srt/server_args.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"Qwen3_5ForConditionalGeneration",', server_args)

    def test_host_mapped_weights_are_exact_and_fail_closed(self) -> None:
        eagle = (
            CANDIDATE
            / "engine/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _host_mapped_bf16_parameter(", eagle)
        self.assertIn("cudaHostGetDevicePointer", eagle)
        self.assertIn("cupy.cuda.UnownedMemory", eagle)
        self.assertIn("mapped_weight.data_ptr() != device_pointer.value", eagle)
        self.assertIn('SGLANG_OPT_HOST_MAPPED_EMBEDDING") == "1"', eagle)
        self.assertIn('SGLANG_OPT_HOST_MAPPED_VISION") == "1"', eagle)
        self.assertLess(
            eagle.index("self.init_lm_head()"),
            eagle.index("def alloc_memory_pool"),
        )

        dense_mlp = (
            CANDIDATE
            / "engine/sglang/python/sglang/srt/models/qwen2_moe.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _forward_chunk(self, x):", dense_mlp)
        self.assertIn("SGLANG_OPT_DENSE_MLP_CHUNK_SIZE", dense_mlp)
        self.assertIn("output[start:end].copy_", dense_mlp)

    def test_runtime_command_preserves_protocol_owned_values(self) -> None:
        command = list(self.adapter.build_command(self.runtime, 18000))
        self.assertEqual(command[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("127.0.0.1", command)
        self.assertIn("18000", command)
        self.assertIn("qwen3.8-27b", command)
        self.assertNotIn("DFLASH", command)
        self.assertEqual(
            command[command.index("--speculative-algorithm") + 1], "NEXTN"
        )
        self.assertEqual(
            command[command.index("--speculative-draft-model-path") + 1],
            "/models/radixark--qwen3.8-27b-nvfp4/"
            "554ebba9b5f1b79dc11246341960360e6ef05ef4",
        )
        self.assertEqual(command[command.index("--speculative-num-steps") + 1], "8")
        self.assertEqual(
            command[command.index("--speculative-num-draft-tokens") + 1], "9"
        )
        self.assertIn("--enable-linear-replayssm-spec", command)
        self.assertNotIn("--disable-flashinfer-autotune", command)
        self.assertIn("--disable-prefill-cuda-graph", command)
        self.assertNotIn("--disable-decode-cuda-graph", command)
        self.assertNotIn("--cuda-graph-bs-decode", command)
        self.assertNotIn("--bf16-gemm-backend", command)
        self.assertNotIn("gemv", command)
        memory_index = command.index("--mem-fraction-static")
        self.assertEqual(command[memory_index + 1], "0.920")
        attention_index = command.index("--attention-backend")
        self.assertEqual(command[attention_index + 1], "trtllm_mha")
        prefill_index = command.index("--prefill-attention-backend")
        self.assertEqual(command[prefill_index + 1], "flashinfer")
        page_index = command.index("--page-size")
        self.assertEqual(command[page_index + 1], "32")
        cache_index = command.index("--max-mamba-cache-size")
        self.assertEqual(command[cache_index + 1], "16")
        strategy_index = command.index("--mamba-radix-cache-strategy")
        self.assertEqual(command[strategy_index + 1], "extra_buffer")
        running_index = command.index("--max-running-requests")
        self.assertEqual(command[running_index + 1], "4")
        total_tokens_index = command.index("--max-total-tokens")
        self.assertEqual(command[total_tokens_index + 1], "131200")
        self.assertEqual(self.runtime["serving"]["max_active_requests"], 4)
        self.assertEqual(
            self.runtime["engine"]["environment"][
                "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"
            ],
            "1",
        )
        context_index = command.index("--context-length")
        self.assertEqual(command[context_index + 1], "131200")
        self.assertNotIn("--language-model-only", command)
        self.assertEqual(self.runtime["serving"]["max_context_tokens"], 131200)
        self.assertEqual(
            self.runtime["engine"]["environment"][
                "SGLANG_OPT_DENSE_MLP_CHUNK_SIZE"
            ],
            "1024",
        )
        self.assertEqual(
            self.runtime["engine"]["environment"][
                "SGLANG_OPT_HOST_MAPPED_EMBEDDING"
            ],
            "1",
        )
        self.assertEqual(
            self.runtime["engine"]["environment"][
                "SGLANG_OPT_HOST_MAPPED_VISION"
            ],
            "1",
        )
        contract = self.runtime["benchmark"]["contract"]
        self.assertEqual(contract["schema_version"], 8)
        self.assertEqual(contract["generator"]["version"], 8)
        self.assertEqual(
            [case["id"] for case in contract["cases"]],
            ["32k", "64k", "128k"],
        )
        self.assertEqual(contract["execution"]["isolation"], "fresh-context")
        self.assertEqual(contract["ttft_cache"]["prompt_tokens"], 64000)
        self.assertEqual(
            contract["tokenizer"]["engine_payload_sha256"],
            self.runtime["engine"]["distribution"]["payload_id"].removeprefix(
                "sha256:"
            ),
        )

    def test_frontend_accepts_only_current_runtime_source_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime_path = pathlib.Path(temporary) / "runtime.json"
            runtime_path.write_text(json.dumps(self.runtime), encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "LETSINFER_ENGINE_PROTOCOL": "2",
                    "LETSINFER_RUNTIME_CONFIG": str(runtime_path),
                },
            ):
                loaded = self.frontend.load_runtime("sglang")
            self.assertEqual(loaded["schema_version"], 6)

            legacy = json.loads(json.dumps(self.runtime))
            legacy["schema_version"] = 5
            runtime_path.write_text(json.dumps(legacy), encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "LETSINFER_ENGINE_PROTOCOL": "2",
                        "LETSINFER_RUNTIME_CONFIG": str(runtime_path),
                    },
                ),
                self.assertRaisesRegex(
                    self.frontend.AdapterError,
                    "runtime does not match this Engine OCI protocol identity",
                ),
            ):
                self.frontend.load_runtime("sglang")

    def test_runtime_cannot_supply_reserved_environment(self) -> None:
        runtime = json.loads(json.dumps(self.runtime))
        runtime["engine"]["environment"]["LETSINFER_ESCAPE"] = "1"
        with self.assertRaisesRegex(
            self.frontend.AdapterError, "runtime engine environment is invalid"
        ):
            self.frontend.child_environment(runtime)


if __name__ == "__main__":
    unittest.main()
