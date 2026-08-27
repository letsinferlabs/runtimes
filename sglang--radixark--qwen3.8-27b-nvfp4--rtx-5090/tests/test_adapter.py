#!/usr/bin/env python3
"""Focused source and protocol-boundary tests for the RTX 5090 Engine."""

from __future__ import annotations

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

    def test_runtime_command_preserves_protocol_owned_values(self) -> None:
        command = list(self.adapter.build_command(self.runtime, 18000))
        self.assertEqual(command[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("127.0.0.1", command)
        self.assertIn("18000", command)
        self.assertIn("qwen3.8-27b", command)
        self.assertNotIn("DFLASH", command)
        self.assertIn("--disable-flashinfer-autotune", command)
        self.assertIn("--disable-decode-cuda-graph", command)
        self.assertNotIn("--bf16-gemm-backend", command)
        self.assertNotIn("gemv", command)
        cache_index = command.index("--max-mamba-cache-size")
        self.assertEqual(command[cache_index + 1], "20")

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
