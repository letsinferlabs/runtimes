#!/usr/bin/env python3

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def frontend_module():
    path = ROOT / "adapter" / "engine_frontend.py"
    spec = importlib.util.spec_from_file_location("nemotron_vllm_frontend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Engine frontend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapter_module():
    path = ROOT / "adapter" / "engine-adapter"
    loader = importlib.machinery.SourceFileLoader(
        "nemotron_vllm_engine_adapter", str(path)
    )
    spec = importlib.util.spec_from_loader("nemotron_vllm_engine_adapter", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Engine adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdapterTests(unittest.TestCase):
    def test_command_binds_protocol_values_and_expands_draft(self) -> None:
        runtime = json.loads((ROOT / "runtime.json").read_text())
        command = list(adapter_module().build_command(runtime, 18000))
        self.assertEqual(command[:2], ["vllm", "serve"])
        self.assertEqual(
            command[2],
            "/models/nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4/"
            "cc84af2fe71647d87f4486c064f320e1e7535243",
        )
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18000")
        self.assertEqual(
            command[command.index("--served-model-name") + 1],
            "nemotron-3.5-lightning",
        )
        draft = command[command.index("--speculative-config.model") + 1]
        self.assertEqual(
            draft,
            "/models/nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4-dspark/"
            "d10c6ff40d6e69d1f92e407e027de3eafdb77645",
        )
        self.assertNotIn("${artifact:draft}", command)
        self.assertIn("--async-scheduling", command)
        self.assertEqual(
            runtime["engine"]["environment"]["VLLM_MARLIN_USE_ATOMIC_ADD"],
            "1",
        )

    def test_vllm_reasoning_is_normalized(self) -> None:
        payload = (
            b'data: {"choices":[{"delta":{"reasoning":"plan",'
            b'"content":""}}]}\n\n'
        )
        self.assertEqual(
            frontend_module().normalize_vllm_response(payload),
            (
                b'data: {"choices":[{"delta":{"reasoning_content":"plan",'
                b'"content":""}}]}\n\n'
            ),
        )

    def test_reasoning_text_is_not_rewritten(self) -> None:
        payload = b'{"content":"the reasoning: remains text"}'
        self.assertEqual(frontend_module().normalize_vllm_response(payload), payload)

    def test_exact_token_count_contract(self) -> None:
        module = adapter_module()
        module.backend_json = lambda *_args, **_kwargs: (
            200,
            {"count": 3, "tokens": [11, 12, 13]},
        )
        self.assertEqual(module.count_tokens("127.0.0.1", 1, b"{}", "model"), 3)

    def test_invalid_token_count_is_rejected(self) -> None:
        module = adapter_module()
        module.backend_json = lambda *_args, **_kwargs: (
            200,
            {"count": 2, "tokens": [11]},
        )
        with self.assertRaises(module.AdapterError):
            module.count_tokens("127.0.0.1", 1, b"{}", "model")


if __name__ == "__main__":
    unittest.main()
