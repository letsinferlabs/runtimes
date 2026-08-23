#!/usr/bin/env python3

from __future__ import annotations

import base64
import copy
import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest

from tools import (
    benchmark_artifact,
    changed_candidates,
    generate_manifest,
    oci_artifact,
    pin_engine,
    set_publication_source,
    sign_manifest,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ManifestToolTests(unittest.TestCase):
    def test_changed_candidate_selection_is_flat_and_shared_changes_fan_out(self) -> None:
        qwen = "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
        deepseek = "dwarfstar--antirez--deepseek-v4-gguf--dgx-spark"
        self.assertEqual(
            changed_candidates.changed(ROOT, [f"{qwen}/runtime.json"]),
            [qwen],
        )
        self.assertEqual(
            changed_candidates.changed(ROOT, ["tools/generate_manifest.py"]),
            sorted((deepseek, qwen)),
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
            manifest["layers"][0]["annotations"]["org.opencontainers.image.title"],
            "runtime.letsinfer",
        )

    def test_committed_manifest_is_the_canonical_candidate_projection(self) -> None:
        manifest_path = ROOT / "manifest.json"
        sources = generate_manifest.sources_from_manifest(manifest_path)
        evidence = generate_manifest.evidence_from_manifest(manifest_path)
        previous = generate_manifest.read_object(manifest_path)
        expected = generate_manifest.canonical_bytes(
            generate_manifest.generate(ROOT, sources, evidence, previous)
        )
        self.assertEqual(manifest_path.read_bytes(), expected)

    def test_release_metadata_preserves_multiple_runtime_authors(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        qwen = manifest["models"]["qwen3.8-27b"]["targets"]["dgx-spark"]
        release = next(iter(qwen["candidates"].values()))["releases"]["0.1.0-rc.12"]
        self.assertEqual(release["authors"], ["MiaAI-Lab", "Letsinfer"])
        self.assertEqual(release["license"], "AGPL-3.0-only")

    def test_every_candidate_validates_without_a_publication_source(self) -> None:
        records = generate_manifest.candidates(ROOT, {}, require_sources=False)
        self.assertEqual(
            {record["runtime"]["id"] for record in records},
            {
                "dwarfstar--antirez--deepseek-v4-gguf--dgx-spark",
                "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark",
            },
        )

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
        runtime["status"] = "qualified"
        runtime["serving"]["qualified"] = True
        runtime["serving"].pop("blocked_by", None)
        runtime["benchmark"]["record"] = {"path": "x", "sha256": "a", "id": "b"}
        changed = pin_engine.update(
            runtime,
            "ghcr.io/letsinferlabs/engines/example@sha256:" + "9" * 64,
            "sha256:" + "8" * 64,
        )
        self.assertTrue(changed)
        self.assertEqual(runtime["status"], "candidate")
        self.assertFalse(runtime["serving"]["qualified"])
        self.assertEqual(
            runtime["serving"]["blocked_by"], "engine-oci-requalification"
        )
        self.assertIsNone(runtime["benchmark"]["record"])
        self.assertEqual(
            runtime["benchmark"]["contract"]["tokenizer"]["engine_image_sha256"],
            "8" * 64,
        )

    def test_pinning_changed_engine_removes_stale_bound_benchmark(self) -> None:
        source = (
            ROOT
            / "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
            / "runtime.json"
        )
        runtime = json.loads(source.read_text(encoding="utf-8"))
        runtime["benchmark"]["record"] = {
            "path": "benchmark.json",
            "sha256": "a" * 64,
            "id": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runtime_path = root / "runtime.json"
            benchmark_path = root / "benchmark.json"
            runtime_path.write_bytes(generate_manifest.canonical_bytes(runtime))
            benchmark_path.write_text("sealed evidence\n", encoding="utf-8")
            _, changed = pin_engine.pin_runtime(
                runtime_path,
                "ghcr.io/letsinferlabs/engines/example@sha256:" + "7" * 64,
                "sha256:" + "6" * 64,
            )
            self.assertTrue(changed)
            self.assertFalse(benchmark_path.exists())
            self.assertTrue(runtime_path.read_text(encoding="utf-8").startswith("{\n  \""))
            pinned = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertIsNone(pinned["benchmark"]["record"])
            self.assertEqual(
                pinned["benchmark"]["contract"]["tokenizer"]["engine_image_sha256"],
                "6" * 64,
            )

    def test_runtime_publication_source_replacement_is_exactly_scoped(self) -> None:
        document = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        before = copy.deepcopy(document)
        candidate = "dwarfstar--antirez--deepseek-v4-gguf--dgx-spark"
        source = "ghcr.io/letsinferlabs/runtimes/example@sha256:" + "7" * 64
        set_publication_source.update(document, candidate, source)
        self.assertNotEqual(document, before)
        self.assertEqual(
            document["models"]["deepseek-v4-flash"]["targets"]["dgx-spark"]
            ["candidates"][candidate]["releases"]["0.11.0-rc.10"]["source"],
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
        self.assertIn("catalog-v6-$GITHUB_SHA", workflow)
        self.assertIn("python3 -m tools.benchmark_artifact push", workflow)
        self.assertIn("catalog-public-key.pem", workflow)
        self.assertNotIn("LETSINFER_CATALOG_TOKEN", workflow)
        self.assertNotIn("git push origin HEAD:main", workflow)


if __name__ == "__main__":
    unittest.main()
