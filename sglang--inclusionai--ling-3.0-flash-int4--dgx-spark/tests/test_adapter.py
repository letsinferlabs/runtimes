#!/usr/bin/env python3

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def adapter_module():
    path = ROOT / "adapter" / "engine-adapter"
    loader = importlib.machinery.SourceFileLoader("ling_engine_adapter", str(path))
    spec = importlib.util.spec_from_loader("ling_engine_adapter", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Engine adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdapterTests(unittest.TestCase):
    def test_command_binds_protocol_values_and_expands_dspark_draft(self) -> None:
        runtime = json.loads((ROOT / "runtime.json").read_text())
        command = list(adapter_module().build_command(runtime, 18000))
        self.assertEqual(command[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18000")
        self.assertEqual(
            command[command.index("--served-model-name") + 1],
            "ling-3.0-flash",
        )
        self.assertEqual(
            command[command.index("--speculative-algorithm") + 1],
            "DSPARK",
        )
        draft = command[command.index("--speculative-draft-model-path") + 1]
        self.assertEqual(
            draft,
            "/models/inclusionai--ling-3.0-flash-dspark/"
            "8e5d9988c9b09de13f1f7c9d999ff2bfa533a149",
        )
        self.assertNotIn("${artifact:draft}", command)
        self.assertIn("--enable-linear-replayssm-spec", command)
        self.assertEqual(
            command[command.index("--linear-replayssm-cache-len") + 1],
            "32",
        )


if __name__ == "__main__":
    unittest.main()
