#!/usr/bin/env python3

from __future__ import annotations

import importlib.machinery
import importlib.util
import hashlib
import json
import os
import pathlib
import tempfile
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


def patch_module():
    path = ROOT / "patches" / "apply_qsa_sm12_fallback.py"
    spec = importlib.util.spec_from_file_location("qwen_qsa_sm12_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load QSA SM12x patch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def topk_patch_module():
    path = ROOT / "patches" / "apply_qsa_topk_determinism.py"
    spec = importlib.util.spec_from_file_location("qwen_qsa_topk_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load QSA top-k patch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trtllm_patch_module():
    path = ROOT / "patches" / "apply_qsa_sm12_trtllm.py"
    spec = importlib.util.spec_from_file_location("qwen_qsa_trtllm_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load QSA TRT-LLM patch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sm121_bf16_patch_module():
    path = ROOT / "patches" / "apply_sm121_bf16_decode.py"
    spec = importlib.util.spec_from_file_location("qwen_sm121_bf16_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load SM121 BF16 patch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_hc_patch_module():
    path = ROOT / "patches" / "apply_hc_mix_deterministic_sm121.py"
    spec = importlib.util.spec_from_file_location("qwen_deterministic_hc_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load deterministic HC patch")
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
    def test_deterministic_hc_patch_is_exact_and_two_way_split_k(self) -> None:
        module = deterministic_hc_patch_module()
        self.assertEqual(len(module.EXPECTED_SHA256), 64)
        self.assertEqual(len(module.PATCHED_SHA256), 64)
        self.assertIn("deterministic_hc_mix_sm121", module.NEW)
        source = (ROOT / "kernels" / "qwen4_hc_mix_sm121.py").read_text()
        self.assertIn("SPLITS = 2", source)
        self.assertIn("partial0 + partial1", source)
        self.assertNotIn("tl.atomic_add(\n            partial_ptr", source)
        dockerfile = (ROOT / "image" / "Dockerfile").read_text()
        self.assertIn("qwen4_hc_mix_sm121.py", dockerfile)
        self.assertIn("apply_hc_mix_deterministic_sm121.py", dockerfile)

    def test_sm121_bf16_patch_is_hash_guarded_and_runtime_gated(self) -> None:
        module = sm121_bf16_patch_module()
        self.assertEqual(len(module.PATCHES), 3)
        replacement_text = "\n".join(
            after
            for contract in module.PATCHES.values()
            for _before, after in contract["replacements"]
        )
        for contract in module.PATCHES.values():
            self.assertEqual(len(contract["before"]), 64)
            self.assertEqual(len(contract["after"]), 64)
        self.assertIn("BLOCK_N=64", replacement_text)
        self.assertIn("BLOCK_J=64", replacement_text)
        self.assertIn("SGLANG_SM121_PAD_BF16_M1", replacement_text)
        self.assertIn("(8320, 2560)", replacement_text)
        self.assertIn("(124160, 2560)", replacement_text)
        dockerfile = (ROOT / "image" / "Dockerfile").read_text()
        self.assertIn("apply_sm121_bf16_decode.py", dockerfile)

    def test_adapter_accepts_only_the_sealed_placement_group_document(self) -> None:
        module = adapter_module()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "placement-group.json"
            placement_group_id = "a" * 32
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "placement_group_id": placement_group_id,
                        "endpoint_placement_id": "b" * 32,
                        "placements": [
                            {
                                "placement_id": "b" * 32,
                                "node_id": "c" * 32,
                                "task_id": "task-0",
                            },
                            {
                                "placement_id": "d" * 32,
                                "node_id": "e" * 32,
                                "task_id": "task-1",
                            },
                        ],
                        "connections": [],
                    }
                ),
                encoding="utf-8",
            )
            module.PLACEMENT_GROUP_PATH = path
            with mock.patch.dict(
                os.environ,
                {
                    "LETSINFER_PLACEMENT_GROUP_CONFIG": str(path),
                    "LETSINFER_PLACEMENT_GROUP_ID": placement_group_id,
                },
                clear=False,
            ):
                self.assertEqual(module._load_group()["placement_group_id"], placement_group_id)
            with mock.patch.dict(
                os.environ,
                {
                    "LETSINFER_PLACEMENT_GROUP_CONFIG": str(path),
                    "LETSINFER_PLACEMENT_GROUP_ID": "f" * 32,
                },
                clear=False,
            ), self.assertRaisesRegex(module.AdapterError, "two-node parallel"):
                module._load_group()

    def test_trtllm_patch_extends_only_the_blackwell_architecture_gate(self) -> None:
        module = trtllm_patch_module()
        self.assertIn("is_sm100_supported", module.OLD)
        self.assertNotIn("is_sm120_supported", module.OLD)
        self.assertIn("is_sm120_supported", module.NEW)
        self.assertIn(
            "is_sm100_supported() or is_sm120_supported()", module.NEW
        )
        self.assertEqual(len(module.EXPECTED_SHA256), 64)
        self.assertEqual(len(module.PATCHED_SHA256), 64)
        dockerfile = (ROOT / "image" / "Dockerfile").read_text()
        self.assertIn("apply_qsa_sm12_trtllm.py", dockerfile)
        cuda_test = (ROOT / "tests" / "run_qsa_trtllm_sm121_cuda.py").read_text()
        self.assertIn("_resolve_trtllm_sparse_decode", cuda_test)
        self.assertIn("torch.cuda.CUDAGraph", cuda_test)

    def test_qsa_patch_gates_direct_gather_path_to_sm12x(self) -> None:
        module = patch_module()
        source = (
            "def decode(self, forward_batch, topk_indices, torch, q, k_buffer, "
            "v_buffer, layer, qsa_sm121_sparse_attention, "
            "qsa_sm121_sparse_attention_graph, "
            "_resolve_trtllm_sparse_decode):\n"
            + module.OLD
            + '        return "fast"\n'
        )
        namespace: dict[str, object] = {}
        self.assertEqual(source.count(module.OLD), 1)
        patched = source.replace(module.OLD, module.NEW)
        self.assertEqual(patched.count(module.NEW), 1)
        exec(compile(patched, "<qsa-patch-test>", "exec"), namespace)

        class Value:
            device = "cuda:0"
            shape = (1, 2, 3)

            def to(self, _dtype):
                return self

            def contiguous(self):
                return self

            def reshape(self, *shape):
                return ("direct", shape)

        class Backend:
            class Pool:
                req_to_token = "req-to-token"

            req_to_token_pool = Pool()

            def __init__(self):
                self.metadata = type(
                    "Metadata",
                    (),
                    {
                        "is_cuda_graph": False,
                        "row_req_pool_indices": "request-rows",
                        "token_to_batch_idx": "token-to-sequence",
                        "sequence_lengths": "sequence-lengths",
                    },
                )()

            def _resolve_metadata(self, _forward_batch):
                return self.metadata

            def _logical_to_physical(self, _indices, _metadata):
                return "slots"

        class Cuda:
            capability = (12, 1)

            @classmethod
            def get_device_capability(cls, _device):
                return cls.capability

        class Torch:
            int32 = "int32"
            cuda = Cuda

        class Layer:
            scaling = 1.0

        calls = {"direct": 0, "graph": 0, "fast": 0}

        def direct(*_arguments):
            calls["direct"] += 1
            return Value()

        def graph(*arguments):
            calls["graph"] += 1
            self.assertEqual(arguments[4:8], (
                "token-to-sequence",
                "request-rows",
                "sequence-lengths",
                "req-to-token",
            ))
            return Value()

        def fast():
            calls["fast"] += 1
            return object()

        decode = namespace["decode"]
        value = Value()
        backend = Backend()
        self.assertEqual(
            decode(
                backend,
                None,
                value,
                Torch,
                value,
                None,
                None,
                Layer(),
                direct,
                graph,
                fast,
            ),
            ("direct", (1, -1)),
        )
        self.assertEqual(calls, {"direct": 1, "graph": 0, "fast": 0})

        backend.metadata.is_cuda_graph = True
        self.assertEqual(
            decode(
                backend,
                None,
                value,
                Torch,
                value,
                None,
                None,
                Layer(),
                direct,
                graph,
                fast,
            ),
            ("direct", (1, -1)),
        )
        self.assertEqual(calls, {"direct": 1, "graph": 1, "fast": 0})

        Cuda.capability = (11, 0)
        self.assertEqual(
            decode(
                backend,
                None,
                value,
                Torch,
                value,
                None,
                None,
                Layer(),
                direct,
                graph,
                fast,
            ),
            "fast",
        )
        self.assertEqual(calls, {"direct": 1, "graph": 1, "fast": 1})

    def test_sm121_kernel_is_exact_direct_gather_online_softmax(self) -> None:
        module = patch_module()
        path = ROOT / "kernels" / "qsa_sm121.py"
        source = path.read_text()
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), module.KERNEL_SHA256)
        self.assertIn("safe_slots = tl.where(valid, slots, 0)", source)
        self.assertIn("req_to_token", source)
        self.assertIn("USE_LOGICAL_INDICES=True", source)
        self.assertIn("slots < CACHE_TOKENS", source)
        self.assertIn("running_max", source)
        self.assertIn("running_sum", source)
        self.assertIn("accumulator", source)
        self.assertNotIn("index_select", source)
        self.assertNotIn("repeat_interleave", source)
        dockerfile = (ROOT / "image" / "Dockerfile").read_text()
        self.assertIn("kernels/qsa_sm121.py", dockerfile)

    def test_qsa_topk_patch_canonicalizes_the_selected_set_in_place(self) -> None:
        module = topk_patch_module()
        self.assertIn("values = tl.sort(values)", module.KERNEL_INSERT)
        self.assertIn("values >= 0", module.KERNEL_INSERT)
        self.assertIn("_canonicalize_fast_topk_kernel[(batch,)]", module.BODY_NEW)
        self.assertIn("num_warps=8", module.BODY_NEW)
        self.assertEqual(len(module.EXPECTED_SHA256), 64)
        self.assertEqual(len(module.PATCHED_SHA256), 64)
        dockerfile = (ROOT / "image" / "Dockerfile").read_text()
        self.assertIn("apply_qsa_topk_determinism.py", dockerfile)

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
        self.assertIn("--allow-auto-truncate", command)
        self.assertEqual(
            command[command.index("--speculative-algorithm") + 1], "NEXTN"
        )
        self.assertEqual(command[command.index("--speculative-num-steps") + 1], "3")
        self.assertEqual(
            command[command.index("--speculative-eagle-topk") + 1], "1"
        )
        self.assertEqual(
            command[command.index("--speculative-num-draft-tokens") + 1], "4"
        )
        self.assertIn("--enable-linear-replayssm-spec", command)
        self.assertEqual(
            command[command.index("--speculative-attention-mode") + 1], "decode"
        )
        self.assertNotIn("--default-chat-template-kwargs", command)
        self.assertNotIn("--disable-cuda-graph", command)
        self.assertNotIn("--disable-radix-cache", command)
        worker = list(module._native_command(runtime, context(1)))
        self.assertEqual(worker[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertNotIn("/opt/letsinfer/bin/nsys-tee", command)
        self.assertNotIn("/opt/letsinfer/bin/nsys-tee", worker)

    def test_public_recipe_declares_the_complete_262k_matrix(self) -> None:
        runtime = json.loads((ROOT / "runtime.json").read_text())
        self.assertEqual(runtime["version"], "0.1.0-rc.1")
        self.assertEqual(runtime["serving"]["max_context_tokens"], 262144)
        contract = runtime["benchmark"]["contract"]
        self.assertEqual(contract["execution"]["isolation"], "fresh-context")
        self.assertEqual(contract["short"]["domains"], ["code", "prose"])
        self.assertEqual(contract["short"]["concurrencies"], [1, 2, 4])
        self.assertEqual(
            [
                (case["id"], case["prompt_tokens"], case["concurrencies"])
                for case in contract["cases"]
            ],
            [
                ("32k", 32768, [1, 2, 4]),
                ("64k", 65536, [1, 2, 4]),
                ("128k", 131072, [1, 2, 4]),
                ("256k", 260000, [1, 2, 4]),
            ],
        )
        self.assertEqual(contract["ttft_cache"]["prompt_tokens"], 64000)
        self.assertEqual(contract["ttft_cache"]["repetitions"], 2)
        self.assertFalse(runtime["cache"]["persistent"])

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
            {"LETSINFER_PLACEMENT_GROUP_ID": "a" * 32},
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
                "LETSINFER_PLACEMENT_GROUP_CONFIG": (
                    "/run/letsinfer/placement-group.json"
                ),
                "LETSINFER_PORT_BASE": "18000",
                "LETSINFER_PORT_COUNT": "4",
            },
            clear=False,
        ):
            self.assertEqual(module.allocated_backend_port("127.0.0.1"), 18002)


if __name__ == "__main__":
    unittest.main()
