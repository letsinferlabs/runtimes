#!/usr/bin/env python3

from __future__ import annotations

import base64
import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import (
    benchmark_artifact,
    changed_candidates,
    generate_manifest,
    oci_artifact,
    oci_layout,
    pin_engine,
    readme_onboarding,
    set_publication_source,
    sign_manifest,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ManifestToolTests(unittest.TestCase):
    def test_changed_candidate_selection_is_flat_and_shared_changes_fan_out(self) -> None:
        qwen = "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
        ling = "sglang--inclusionai--ling-3.0-flash-int4--dgx-spark"
        nemotron = (
            "sglang--nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4--dgx-spark"
        )
        nemotron_vllm = (
            "vllm--nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4--dgx-spark"
        )
        deepseek = "dwarfstar--antirez--deepseek-v4-gguf--dgx-spark"
        sparkinfer = "sparkinfer--0xsero--deepseek-v4-flash-0731-spark--dgx-spark"
        self.assertEqual(
            changed_candidates.changed(ROOT, [f"{qwen}/runtime.json"]),
            [qwen],
        )
        self.assertEqual(
            changed_candidates.changed(ROOT, ["tools/generate_manifest.py"]),
            sorted(changed_candidates.candidates(ROOT)),
        )

    def test_runtime_oci_plan_is_deterministic_and_pull_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "runtime.letsinfer"
            artifact.write_bytes(b"deterministic runtime pack")
            first = oci_artifact.plan(
                artifact,
                repository="ghcr.io/letsinferlabs/runtimes/example",
                candidate="example--owner--model--target",
                version="1.2.3-rc.4",
            )
            second = oci_artifact.plan(
                artifact,
                repository="ghcr.io/letsinferlabs/runtimes/example",
                candidate="example--owner--model--target",
                version="1.2.3-rc.4",
            )
        self.assertEqual(first.manifest, second.manifest)
        self.assertEqual(first.source, second.source)
        manifest = json.loads(first.manifest)
        self.assertEqual(len(manifest["layers"]), 1)
        self.assertEqual(
            manifest["layers"][0]["mediaType"], oci_artifact.PACK_MEDIA_TYPE
        )
        self.assertEqual(
            oci_artifact.PACK_MEDIA_TYPE, oci_layout.RUNTIME_LAYER_MEDIA_TYPE
        )
        self.assertEqual(
            manifest["layers"][0]["annotations"]["org.opencontainers.image.title"],
            "runtime.letsinfer",
        )

    def test_committed_manifest_is_the_canonical_candidate_projection(self) -> None:
        manifest_path = ROOT / "manifest.json"
        sources = generate_manifest.sources_from_manifest(manifest_path)
        previous = generate_manifest.read_object(manifest_path)
        expected = generate_manifest.canonical_bytes(
            generate_manifest.generate(ROOT, sources, previous)
        )
        self.assertEqual(manifest_path.read_bytes(), expected)

    def test_unqualified_candidate_does_not_enter_catalog_projection(self) -> None:
        manifest_path = ROOT / "manifest.json"
        sources = generate_manifest.sources_from_manifest(manifest_path)
        previous = generate_manifest.read_object(manifest_path)
        items = generate_manifest.candidates(ROOT, sources)
        unqualified = copy.deepcopy(items[0])
        unqualified["runtime"]["id"] = "engine--owner--unqualified--target"
        unqualified["runtime"]["logical_model"] = "unqualified"
        unqualified["runtime"]["target"]["id"] = "unqualified-target"
        unqualified["qualified"] = False
        unqualified["consensus"] = None
        unqualified["source"] = None
        unqualified["release_metadata"]["provenance"] = None
        with mock.patch.object(
            generate_manifest, "candidates", return_value=[*items, unqualified]
        ):
            generated = generate_manifest.generate(ROOT, sources, previous)
        self.assertEqual(generated, previous)

    def test_materialized_qualification_remains_execution_bound(self) -> None:
        manifest_path = ROOT / "manifest.json"
        sources = generate_manifest.sources_from_manifest(manifest_path)
        previous = generate_manifest.read_object(manifest_path)
        target = previous["models"]["qwen3.8-27b"]["targets"]["dgx-spark"]
        candidate = target["recommended"]["candidate"]
        version = target["recommended"]["version"]
        target["candidates"][candidate]["releases"][version]["verification"][
            "execution_contract_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            generate_manifest.ManifestError,
            "runtime execution contract changed",
        ):
            generate_manifest.generate(ROOT, sources, previous)

    def test_scored_dgx_candidates_remain_recommended(self) -> None:
        manifest = generate_manifest.read_object(ROOT / "manifest.json")
        expected = {
            "deepseek-v4-flash": (
                "sparkinfer--0xsero--deepseek-v4-flash-0731-spark--dgx-spark",
                "0.1.0-rc.36",
            ),
            "ling-3.0-flash": (
                "sglang--inclusionai--ling-3.0-flash-int4--dgx-spark",
                "0.1.0-rc.3",
            ),
            "nemotron-3.5-lightning": (
                "sglang--nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4--dgx-spark",
                "0.1.0-rc.4",
            ),
            "qwen3.8-27b": (
                "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark",
                "0.1.0-rc.18",
            ),
        }
        for model, identity in expected.items():
            recommended = manifest["models"][model]["targets"]["dgx-spark"][
                "recommended"
            ]
            self.assertEqual(
                (recommended["candidate"], recommended["version"]), identity
            )
        qwen = manifest["models"]["qwen3-0.6b"]["targets"]
        self.assertIsNone(qwen["ios-apple-gpu"]["recommended"])
        self.assertIsNone(qwen["macos-apple-silicon"]["recommended"])

    def test_carried_qualification_remains_revocable(self) -> None:
        manifest_path = ROOT / "manifest.json"
        sources = generate_manifest.sources_from_manifest(manifest_path)
        previous = generate_manifest.read_object(manifest_path)
        target = previous["models"]["qwen3.8-27b"]["targets"]["dgx-spark"]
        recommendation = target["recommended"]
        release = target["candidates"][recommendation["candidate"]]["releases"][
            recommendation["version"]
        ]
        identity = (
            release["source"].rsplit("@", 1)[-1],
            release["verification"]["consensus_sha256"],
        )
        with mock.patch.object(
            generate_manifest, "revocation_identities", return_value={identity}
        ):
            generated = generate_manifest.generate(ROOT, sources, previous)
        self.assertNotIn("qwen3.8-27b", generated["models"])

    def test_release_metadata_preserves_multiple_runtime_authors(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        qwen = manifest["models"]["qwen3.8-27b"]["targets"]["dgx-spark"]
        candidate = next(iter(qwen["candidates"].values()))
        release = candidate["releases"][candidate["latest"]]
        self.assertEqual(
            release["authors"],
            [
                {"github_login": "MiaAI-Lab", "github_id": 83042094, "github_type": "User"},
                {"github_login": "letsinferlabs", "github_id": 317451145, "github_type": "Organization"},
            ],
        )
        self.assertEqual(release["license"], "AGPL-3.0-only")

    def test_scored_maintainer_bypasses_mark_author_benchmark_source(self) -> None:
        manifest = generate_manifest.read_object(ROOT / "manifest.json")
        expected: set[tuple[str, str, str]] = set()
        for consensus_path in ROOT.glob("*/benchmark.consensus.json"):
            consensus = generate_manifest.read_object(consensus_path)
            if consensus["score"]["policy"] != "letsinfer-throughput-geomean-of-author-run-v1":
                continue
            runtime = generate_manifest.read_object(consensus_path.with_name("runtime.json"))
            expected.add(
                (
                    runtime["logical_model"],
                    runtime["id"],
                    runtime["version"],
                )
            )
        observed: set[tuple[str, str, str]] = set()
        for model_id, model in manifest["models"].items():
            for target in model["targets"].values():
                for candidate_id, candidate in target["candidates"].items():
                    for version, release in candidate["releases"].items():
                        source = release["verification"].get("benchmark_source")
                        if source is not None:
                            self.assertEqual(source, "author-benchmark-v1")
                            self.assertEqual(
                                release["verification"]["method"],
                                "allowlisted-maintainer-bypass-v1",
                            )
                            self.assertEqual(release["verification"]["verifiers"], [])
                            self.assertIsNotNone(release["benchmark"])
                            observed.add((model_id, candidate_id, version))
        # Historical scored releases remain in the append-only catalog after
        # their current candidate moves to a newer unscored schema release.
        self.assertLessEqual(expected, observed)

    def test_every_candidate_validates_without_a_publication_source(self) -> None:
        records = generate_manifest.candidates(ROOT, {}, require_sources=False)
        self.assertEqual(
            {record["runtime"]["id"] for record in records},
            changed_candidates.candidates(ROOT),
        )

    def test_schema_six_catalog_history_is_retired_at_cutover(self) -> None:
        self.assertEqual(
            generate_manifest._previous_releases(
                {
                    "schema_version": 6,
                    "models": {"legacy": {"targets": {}}},
                }
            ),
            {},
        )

    def test_parallel_runtime_contract_keeps_engine_semantics_out_of_core(self) -> None:
        runtime = json.loads(
            (
                ROOT
                / "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
                / "runtime.json"
            ).read_text(encoding="utf-8")
        )
        runtime["target"]["placement"].update(
            {
                "strategy": "parallel",
                "node_count": 2,
                "interconnect": {
                    "kind": "connectx",
                    "rdma_required": True,
                    "minimum_speed_mbps": 100000,
                    "minimum_mtu": 9000,
                },
            }
        )
        runtime["orchestration"] = {
            "schema_version": 3,
            "failure_policy": "whole-group",
            "endpoint_owner": "task-0",
            "startup_order": [["task-1"], ["task-0"]],
            "tasks": [
                {
                    "task_id": f"task-{index}",
                    "launcher": "runtime-command",
                    "port_count": 4,
                    "command": ["/opt/runtime/launch", f"task-{index}"],
                    "environment": {},
                    "readiness": {
                        "kind": "exec",
                        "command": ["/opt/runtime/ready"],
                        "interval_seconds": 2,
                        "timeout_seconds": 3,
                        "retries": 90,
                    },
                }
                for index in range(2)
            ],
        }
        generate_manifest.validate_runtime_execution_contract(runtime)
        runtime["orchestration"]["tasks"][0]["rank"] = 0
        with self.assertRaisesRegex(generate_manifest.ManifestError, "unsupported fields"):
            generate_manifest.validate_runtime_execution_contract(runtime)

    def test_single_runtime_cannot_smuggle_a_parallel_contract(self) -> None:
        runtime = json.loads(
            (
                ROOT
                / "dwarfstar--antirez--deepseek-v4-gguf--dgx-spark"
                / "runtime.json"
            ).read_text(encoding="utf-8")
        )
        runtime["orchestration"] = {}
        with self.assertRaisesRegex(generate_manifest.ManifestError, "single-node"):
            generate_manifest.validate_runtime_execution_contract(runtime)

    def test_every_hugging_face_artifact_requires_its_readme_link(self) -> None:
        runtime = {
            "artifacts": [
                {"uri": "hf://Owner/Primary"},
                {"uri": "hf://Owner/Drafter"},
            ]
        }
        generate_manifest.validate_model_links(
            runtime,
            "https://huggingface.co/Owner/Primary\n"
            "https://huggingface.co/Owner/Drafter\n",
        )
        with self.assertRaisesRegex(
            generate_manifest.ManifestError, "Owner/Drafter"
        ):
            generate_manifest.validate_model_links(
                runtime, "https://huggingface.co/Owner/Primary\n"
            )

    def test_runtime_readme_onboarding_uses_the_logical_model(self) -> None:
        block = readme_onboarding.launch_block("qwen3.8-27b")
        self.assertTrue(block.startswith("> **Run this model with [Let's Infer]"))
        self.assertIn("https://letsinfer.ai/", block)
        self.assertIn("curl -fsSL https://letsinfer.ai/install.sh | sh", block)
        self.assertIn("letsinfer model install qwen3.8-27b", block)
        readme_onboarding.validate(block + "\n# Existing README\n", "qwen3.8-27b")
        with self.assertRaisesRegex(
            readme_onboarding.ReadmeError, "logical model deepseek-v4-flash"
        ):
            readme_onboarding.validate(
                block + "\n# Existing README\n", "deepseek-v4-flash"
            )

    def test_runtime_readme_onboarding_preserves_existing_content(self) -> None:
        existing = "# Upstream runtime\n\nOriginal documentation.\n"
        updated = readme_onboarding.prepend(existing, "model")
        self.assertTrue(updated.endswith(existing))
        self.assertEqual(
            readme_onboarding.prepend(updated, "model"),
            updated,
        )
        misplaced = existing + readme_onboarding.launch_block("model")
        with self.assertRaisesRegex(readme_onboarding.ReadmeError, "misplaced"):
            readme_onboarding.prepend(misplaced, "model")

    def test_runtime_readme_onboarding_supports_documented_script_entrypoint(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/readme_onboarding.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--candidate", completed.stdout)

    def test_signature_document_is_ed25519_and_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            private = root / "private.pem"
            public = root / "public.pem"
            subprocess.run(
                ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "openssl",
                    "pkey",
                    "-in",
                    str(private),
                    "-pubout",
                    "-out",
                    str(public),
                ],
                check=True,
                capture_output=True,
            )
            document = sign_manifest.sign(ROOT / "manifest.json", private, public)
            self.assertEqual(document["algorithm"], "ed25519")
            self.assertEqual(
                document["catalog_sha256"],
                hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest(),
            )
            signature = root / "signature.bin"
            signature.write_bytes(base64.b64decode(document["signature_base64"]))
            subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(public),
                    "-rawin",
                    "-in",
                    str(ROOT / "manifest.json"),
                    "-sigfile",
                    str(signature),
                ],
                check=True,
                capture_output=True,
            )

    def test_new_engine_identity_invalidates_old_qualification(self) -> None:
        runtime = json.loads(
            (
                ROOT
                / "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark/runtime.json"
            ).read_text(encoding="utf-8")
        )
        changed, execution_changed = pin_engine.update(
            runtime,
            "ghcr.io/letsinferlabs/engines/example@sha256:" + "9" * 64,
            "sha256:" + "8" * 64,
            "sha256:" + "7" * 64,
        )
        self.assertTrue(changed)
        self.assertTrue(execution_changed)
        self.assertNotIn("status", runtime)
        self.assertNotIn("qualified", runtime["serving"])
        self.assertNotIn("record", runtime["benchmark"])
        self.assertEqual(
            runtime["benchmark"]["contract"]["tokenizer"]["engine_payload_sha256"],
            "7" * 64,
        )

    def test_schema_six_engine_distribution_is_pinned(self) -> None:
        runtime = json.loads(
            (
                ROOT
                / "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark/runtime.json"
            ).read_text(encoding="utf-8")
        )
        if "oci" in runtime["engine"]:
            runtime["schema_version"] = 6
            runtime["engine"]["distribution"] = {
                "kind": "oci-container",
                **runtime["engine"].pop("oci"),
            }
        changed, execution_changed = pin_engine.update(
            runtime,
            "ghcr.io/letsinferlabs/engine-images@sha256:" + "9" * 64,
            "sha256:" + "8" * 64,
            "sha256:" + "7" * 64,
        )
        self.assertTrue(changed)
        self.assertTrue(execution_changed)
        self.assertEqual(runtime["engine"]["distribution"]["kind"], "oci-container")
        self.assertEqual(
            runtime["engine"]["distribution"]["payload_id"], "sha256:" + "7" * 64
        )
        self.assertEqual(
            runtime["benchmark"]["contract"]["tokenizer"]["engine_payload_sha256"],
            "7" * 64,
        )

    def test_pinning_changed_engine_removes_stale_bound_benchmark(self) -> None:
        source = (
            ROOT
            / "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
            / "runtime.json"
        )
        runtime = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runtime_path = root / "runtime.json"
            benchmark_path = root / "benchmark.json"
            consensus_path = root / "benchmark.consensus.json"
            release_path = root / "release.json"
            runtime_bytes = generate_manifest.canonical_bytes(runtime)
            reference = (
                "ghcr.io/letsinferlabs/engines/example@sha256:" + "7" * 64
            )
            immutable_id = "sha256:" + "6" * 64
            payload_id = "sha256:" + "5" * 64
            _, _, _, expected_runtime_bytes = pin_engine.update_bytes(
                runtime_bytes, reference, immutable_id, payload_id
            )
            runtime_path.write_bytes(runtime_bytes)
            benchmark_path.write_text("sealed evidence\n", encoding="utf-8")
            consensus_path.write_text("sealed consensus\n", encoding="utf-8")
            release_path.write_text(
                json.dumps({"provenance": {"qualified": True}}), encoding="utf-8"
            )
            _, changed = pin_engine.pin_runtime(
                runtime_path,
                reference,
                immutable_id,
                payload_id,
            )
            self.assertTrue(changed)
            self.assertFalse(benchmark_path.exists())
            self.assertFalse(consensus_path.exists())
            self.assertEqual(runtime_path.read_bytes(), expected_runtime_bytes)
            pinned = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertIsNone(json.loads(release_path.read_text())["provenance"])
            self.assertEqual(
                pinned["benchmark"]["contract"]["tokenizer"]["engine_payload_sha256"],
                "5" * 64,
            )

    def test_packaging_only_engine_repin_preserves_evidence(self) -> None:
        source = (
            ROOT
            / "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
            / "runtime.json"
        )
        runtime = json.loads(source.read_text(encoding="utf-8"))
        distribution = runtime["engine"].get(
            "distribution", runtime["engine"].get("oci")
        )
        distribution["payload_id"] = "sha256:" + "5" * 64
        runtime["benchmark"]["contract"]["tokenizer"].pop(
            "engine_image_sha256", None
        )
        runtime["benchmark"]["contract"]["tokenizer"][
            "engine_payload_sha256"
        ] = "5" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runtime_path = root / "runtime.json"
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            benchmark = root / "benchmark.json"
            consensus = root / "benchmark.consensus.json"
            benchmark.write_text("evidence", encoding="utf-8")
            consensus.write_text("consensus", encoding="utf-8")
            _, changed = pin_engine.pin_runtime(
                runtime_path,
                "ghcr.io/letsinferlabs/engines/example@sha256:" + "9" * 64,
                "sha256:" + "8" * 64,
                "sha256:" + "5" * 64,
            )
            self.assertTrue(changed)
            self.assertTrue(benchmark.exists())
            self.assertTrue(consensus.exists())

    def test_runtime_publication_source_replacement_is_exactly_scoped(self) -> None:
        document = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        before = copy.deepcopy(document)
        candidate = "dwarfstar--antirez--deepseek-v4-gguf--dgx-spark"
        source = "ghcr.io/letsinferlabs/runtimes/example@sha256:" + "7" * 64
        set_publication_source.update(document, candidate, source)
        self.assertNotEqual(document, before)
        record = document["models"]["deepseek-v4-flash"]["targets"]["dgx-spark"][
            "candidates"
        ][candidate]
        self.assertEqual(
            record["releases"][record["latest"]]["source"],
            source,
        )
        self.assertEqual(
            document["models"]["qwen3.8-27b"],
            before["models"]["qwen3.8-27b"],
        )

    def test_benchmark_oci_plan_is_runtime_bound_and_deterministic(self) -> None:
        candidate = "example--owner--model--target"
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "benchmark.json"
            evidence.write_text(
                json.dumps({"id": "1" * 64}), encoding="utf-8"
            )
            artifact = root / "runtime.letsinfer"
            artifact.write_bytes(b"runtime")
            runtime = oci_artifact.plan(
                artifact,
                repository="ghcr.io/letsinferlabs/runtimes/example",
                candidate=candidate,
                version="1.2.3",
            )
            runtime_plan = root / "runtime-plan.json"
            runtime_plan.write_text(json.dumps(runtime.document()), encoding="utf-8")
            first = benchmark_artifact.plan(
                evidence,
                repository="ghcr.io/letsinferlabs/benchmarks/example",
                candidate=candidate,
                version="1.2.3",
                runtime_plan=runtime_plan,
            )
            second = benchmark_artifact.plan(
                evidence,
                repository="ghcr.io/letsinferlabs/benchmarks/example",
                candidate=candidate,
                version="1.2.3",
                runtime_plan=runtime_plan,
            )
        self.assertEqual(first.manifest, second.manifest)
        manifest = json.loads(first.manifest)
        self.assertEqual(manifest["subject"]["digest"], runtime.manifest_digest)
        self.assertEqual(
            manifest["layers"][0]["mediaType"],
            benchmark_artifact.BENCHMARK_MEDIA_TYPE,
        )

    def test_runtime_release_publishes_catalog_directly(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("contents: write", workflow)
        self.assertIn("gh release create", workflow)
        self.assertIn('tag="catalog-$GITHUB_SHA"', workflow)
        self.assertNotIn("catalog-v7-$GITHUB_SHA", workflow)
        self.assertIn(
            "from tools.generate_manifest import SCHEMA_VERSION; print(SCHEMA_VERSION)",
            workflow,
        )
        self.assertEqual(workflow.count("from tools.generate_manifest"), 1)
        self.assertIn("int(os.environ['CATALOG_SCHEMA_VERSION'])", workflow)
        self.assertNotIn("catalog['schema_version'] == 7", workflow)
        self.assertNotIn("LETSINFER_VERIFIER_BYPASS_GITHUB_IDS", workflow)
        self.assertIn("revocations.json.sig", workflow)
        self.assertNotIn("tools.benchmark_artifact push", workflow)
        self.assertIn("catalog-public-key.pem", workflow)
        self.assertNotIn("LETSINFER_CATALOG_TOKEN", workflow)
        self.assertNotIn("git push origin HEAD:main", workflow)


if __name__ == "__main__":
    unittest.main()
