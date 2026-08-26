#!/usr/bin/env python3

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def frontend_module():
    path = ROOT / "adapter" / "engine_frontend.py"
    spec = importlib.util.spec_from_file_location("qwen_flash_frontend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Engine frontend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapter_module():
    path = ROOT / "adapter" / "engine-adapter"
    loader = importlib.machinery.SourceFileLoader(
        "qwen_flash_engine_adapter", str(path)
    )
    spec = importlib.util.spec_from_loader("qwen_flash_engine_adapter", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Engine adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(rank: int = 0) -> dict[str, object]:
    return {
        "group": {},
        "task_id": f"task-{rank}",
        "rank": rank,
        "interface": "enp1s0f0np0",
        "device": "mlx5_0",
        "local_address": f"169.254.145.{8 + rank}",
        "local_backend_port": 18002,
        "dist_init_addr": "169.254.145.8:18001",
    }


class AdapterTests(unittest.TestCase):
    def test_command_binds_exact_two_node_resources(self) -> None:
        module = adapter_module()
        runtime = json.loads((ROOT / "runtime.json").read_text())
        module._context = lambda _runtime: context(0)
        command = list(module.build_command(runtime, 18002))
        self.assertEqual(command[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertEqual(command[command.index("--tp-size") + 1], "2")
        self.assertEqual(command[command.index("--nnodes") + 1], "2")
        self.assertEqual(command[command.index("--node-rank") + 1], "0")
        self.assertEqual(
            command[command.index("--dist-init-addr") + 1],
            "169.254.145.8:18001",
        )
        self.assertEqual(command[command.index("--port") + 1], "18002")
        self.assertNotIn("--allow-auto-truncate", command)
        self.assertNotIn("--speculative-algorithm", command)

    def test_runtime_cannot_replace_distributed_arguments(self) -> None:
        module = adapter_module()
        runtime = json.loads((ROOT / "runtime.json").read_text())
        runtime["engine"]["arguments"].append("--node-rank")
        with self.assertRaisesRegex(module.AdapterError, "distributed resources"):
            module._engine_arguments(runtime)

    def test_rdma_environment_forces_ib_without_socket_fallback(self) -> None:
        module = adapter_module()
        runtime = json.loads((ROOT / "runtime.json").read_text())
        module._context = lambda _runtime: context(0)
        module._write_baseline = lambda _context: None
        with mock.patch.dict(
            os.environ,
            {"LETSINFER_GROUP_ID": "a" * 32},
            clear=False,
        ):
            environment = module.build_environment(runtime)
        self.assertEqual(environment["NCCL_NET"], "IB")
        self.assertEqual(environment["NCCL_IB_DISABLE"], "0")
        self.assertEqual(environment["NCCL_IB_HCA"], "=mlx5_0")
        self.assertEqual(environment["NCCL_SOCKET_IFNAME"], "=enp1s0f0np0")
        self.assertEqual(environment["NCCL_NET_GDR_LEVEL"], "SYS")

    def test_readiness_requires_ib_counters_and_endpoint(self) -> None:
        module = adapter_module()
        module.load_runtime = lambda _engine: {}
        module._context = lambda _runtime: context(0)
        module._nccl_log = lambda _context: "NCCL INFO NET/IB : Using mlx5_0"
        module._rdma_progressed = lambda _context: True
        module._endpoint_healthy = lambda: True
        self.assertEqual(module.ready(), 0)
        module._nccl_log = lambda _context: "NCCL INFO NET/Socket : Using eth0"
        self.assertEqual(module.ready(), 1)

    def test_worker_execs_rank_one_with_forced_environment(self) -> None:
        module = adapter_module()
        runtime = json.loads((ROOT / "runtime.json").read_text())
        module.load_runtime = lambda _engine: runtime
        module._context = lambda _runtime: context(1)
        module.build_environment = lambda _runtime: {"NCCL_NET": "IB"}
        with (
            mock.patch.dict(os.environ, {"LETSINFER_ENGINE_PORT": "-1"}),
            mock.patch.object(module.os, "execvpe", side_effect=RuntimeError("exec")) as execute,
            self.assertRaisesRegex(RuntimeError, "exec"),
        ):
            module.worker()
        self.assertEqual(execute.call_args.args[0], "python3")
        self.assertIn("--node-rank", execute.call_args.args[1])
        self.assertEqual(execute.call_args.args[2]["NCCL_NET"], "IB")

    def test_frontend_uses_core_assigned_backend_port(self) -> None:
        module = frontend_module()
        with mock.patch.dict(
            os.environ,
            {
                "LETSINFER_GROUP_CONFIG": "/run/letsinfer/group.json",
                "LETSINFER_PORT_BASE": "18000",
                "LETSINFER_PORT_COUNT": "4",
            },
            clear=False,
        ):
            self.assertEqual(module.allocated_backend_port("127.0.0.1"), 18002)


if __name__ == "__main__":
    unittest.main()
