#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import importlib.machinery
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def adapter_module():
    path = ROOT / "adapter" / "engine-adapter"
    loader = importlib.machinery.SourceFileLoader(
        "nemotron_engine_adapter", str(path)
    )
    spec = importlib.util.spec_from_loader("nemotron_engine_adapter", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Engine adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdapterTests(unittest.TestCase):
    def test_command_binds_protocol_owned_values_and_expands_draft(self) -> None:
        runtime = json.loads((ROOT / "runtime.json").read_text())
        command = list(adapter_module().build_command(runtime, 18000))
        self.assertEqual(command[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18000")
        self.assertEqual(
            command[command.index("--served-model-name") + 1],
            "nemotron-3.5-lightning",
        )
        draft = command[command.index("--speculative-draft-model-path") + 1]
        self.assertEqual(
            draft,
            "/models/nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4-dspark/"
            "d10c6ff40d6e69d1f92e407e027de3eafdb77645",
        )
        self.assertNotIn("${artifact:draft}", command)


if __name__ == "__main__":
    unittest.main()
