from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import (
    candidate_policy,
    engine_sbom,
    generate_manifest,
    shipit,
    verification_bot,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PublicationPolicyTests(unittest.TestCase):
    def test_engine_mode_is_derived_from_changed_source(self) -> None:
        candidate = "engine--owner--model--target"
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / candidate).mkdir()
            self.assertEqual(
                candidate_policy.classify_paths(
                    candidate=candidate,
                    paths=[f"{candidate}/runtime.json"],
                    candidate_is_new=False,
                    root=root,
                ),
                "reuse-engine",
            )
            self.assertEqual(
                candidate_policy.classify_paths(
                    candidate=candidate,
                    paths=[f"{candidate}/engine/new.py"],
                    candidate_is_new=False,
                    root=root,
                ),
                "build-engine",
            )

    def test_new_candidates_use_shared_preprovisioned_packages(self) -> None:
        existing = "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
        self.assertEqual(
            candidate_policy.runtime_repository(ROOT, existing),
            f"ghcr.io/letsinferlabs/runtimes/{existing}",
        )
        repositories = candidate_policy.publication_repositories(
            ROOT, "newengine--owner--model--target"
        )
        self.assertEqual(
            repositories["runtime_repository"],
            "ghcr.io/letsinferlabs/runtime-artifacts",
        )
        self.assertEqual(
            repositories["engine_repository"],
            "ghcr.io/letsinferlabs/engine-images",
        )
        digest = "sha256:" + "7" * 64
        self.assertEqual(
            verification_bot.runtime_publication_source(
                ROOT, "newengine--owner--model--target", digest
            ),
            f"ghcr.io/letsinferlabs/runtime-artifacts@{digest}",
        )

    def test_engine_sbom_accepts_the_shared_engine_package(self) -> None:
        digest = "sha256:" + "7" * 64
        document = engine_sbom.spdx(
            {"schema_version": 1, "debian": [], "python": []},
            "newengine--owner--model--target",
            f"ghcr.io/letsinferlabs/engine-images@{digest}",
            "sha256:" + "8" * 64,
        )
        self.assertEqual(
            document["packages"][0]["downloadLocation"],
            f"ghcr.io/letsinferlabs/engine-images@{digest}",
        )

    def test_engine_sbom_inventory_is_strictly_typed_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "inventory.json"
            for records in (
                [{"name": "z", "version": "1"}, {"name": "a", "version": "1"}],
                [{"name": "a", "version": 1}],
            ):
                path.write_text(
                    json.dumps(
                        {"schema_version": 1, "debian": [], "python": records}
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(engine_sbom.SbomError):
                    engine_sbom.read_inventory(path)

    def test_shipit_command_and_bypass_reason_are_exact(self) -> None:
        self.assertEqual(shipit.parse_command("/shipit\n"), (False, None))
        self.assertEqual(
            shipit.parse_command(
                "/shipit --bypass-verifiers\nReason: urgent target enablement"
            ),
            (True, "urgent target enablement"),
        )
        for invalid in (
            "/shipit please",
            "/shipit --bypass-verifiers",
            "/shipit --bypass-verifiers\nreason: missing exact field",
            "/shipit --bypass-verifiers\nReason: ",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(shipit.ShipitError):
                shipit.parse_command(invalid)

    def test_shipit_verifies_every_bundle_file_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name in ("bundle.json", "runtime.letsinfer"):
                (root / name).write_bytes(name.encode())
            completed = [
                subprocess.CompletedProcess(
                    ["gh", "--version"], 0, b"gh version 2.97.0\n", b""
                ),
                subprocess.CompletedProcess([], 0, b"verified\n", b""),
                subprocess.CompletedProcess([], 0, b"verified\n", b""),
            ]
            with mock.patch.dict(
                os.environ, {"LETSINFER_ATTESTATION_TOKEN": "test-token"}
            ), mock.patch.object(
                shipit.subprocess, "run", side_effect=completed
            ) as run:
                shipit._verify_bundle_attestations(root)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list[1:]:
            command = call.args[0]
            self.assertEqual(command[:3], ["gh", "attestation", "verify"])
            self.assertIn(shipit.FINALIZER_CERT_IDENTITY, command)
            self.assertEqual(call.kwargs["env"]["GH_TOKEN"], "test-token")

    def test_shipit_bypass_requires_a_configured_immutable_maintainer_id(self) -> None:
        shipit.require_configured_bypass_actor(10000001, "10000001,10000002")
        for configured in (
            "",
            "999",
            "10000001,10000001",
            "010000001",
            "10000001, 10000002",
            "login",
            "0",
        ):
            with self.subTest(configured=configured), self.assertRaises(
                shipit.ShipitError
            ):
                shipit.require_configured_bypass_actor(10000001, configured)

    def test_shipit_permission_is_bound_to_the_same_immutable_actor_id(self) -> None:
        actor = {
            "github_login": "ConfiguredMaintainer",
            "github_id": 10000001,
            "github_type": "User",
        }
        response = {
            "permission": "maintain",
            "user": {
                "login": "ConfiguredMaintainer",
                "id": 10000001,
                "type": "User",
            },
        }
        with mock.patch.object(
            verification_bot, "api", return_value=response
        ) as api:
            self.assertEqual(shipit._permission(actor), "maintain")
        api.assert_called_once_with(
            "repos/letsinferlabs/runtimes/collaborators/ConfiguredMaintainer/permission"
        )
        response["user"]["id"] = 999
        with mock.patch.object(
            verification_bot, "api", return_value=response
        ), self.assertRaisesRegex(shipit.ShipitError, "permission is invalid"):
            shipit._permission(actor)

    def _waived_consensus(self, actor_id: int) -> tuple[dict, dict]:
        candidate = "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
        runtime = generate_manifest.read_object(ROOT / candidate / "runtime.json")
        verifier = {
            "github_login": "Verifier",
            "github_id": 99,
            "github_type": "User",
        }
        consensus = {
            "schema_version": 2,
            "candidate_id": candidate,
            "runtime_version": runtime["version"],
            "pull_request": 17,
            "subject": {
                "candidate_id": candidate,
                "runtime_version": runtime["version"],
                "engine_oci_manifest_digest": runtime["engine"]["oci"]["reference"].rsplit("@", 1)[-1],
                "benchmark_contract_sha256": hashlib.sha256(
                    generate_manifest.canonical_bytes(runtime["benchmark"]["contract"])
                ).hexdigest(),
                "target_contract_sha256": hashlib.sha256(
                    generate_manifest.canonical_bytes(runtime["target"])
                ).hexdigest(),
            },
            "policy": {"id": "letsinfer-two-independent-passes-v1"},
            "qualification": {
                "passed": True,
                "independent_verifiers": 1,
                "required_verifiers": 2,
                "safety_passed": True,
                "blocking_failures": [],
            },
            "verifications": [{}],
            "verifiers": [verifier],
            "waiver": {
                "schema_version": 1,
                "policy": "maintainer-one-independent-pass-v1",
                "actor": {
                    "github_login": "ConfiguredMaintainer",
                    "github_id": actor_id,
                    "github_type": "User",
                },
                "reason": "Documented maintainer exception",
                "comment_id": 123,
                "comment_url": "https://github.com/letsinferlabs/runtimes/pull/17#issuecomment-123",
                "issued_at": "2026-08-24T12:00:00Z",
            },
        }
        consensus["consensus_id"] = hashlib.sha256(
            generate_manifest.canonical_bytes(consensus)
        ).hexdigest()
        return runtime, consensus

    def test_verifier_waiver_is_bound_to_configured_immutable_account_id(self) -> None:
        runtime, consensus = self._waived_consensus(10000001)
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            generate_manifest.ManifestError, "IDs are not configured"
        ):
            generate_manifest.validate_consensus_binding(runtime, consensus)
        with mock.patch.dict(
            os.environ, {"LETSINFER_VERIFIER_BYPASS_GITHUB_IDS": "999"}
        ), self.assertRaisesRegex(generate_manifest.ManifestError, "unauthorized"):
            generate_manifest.validate_consensus_binding(runtime, consensus)
        with mock.patch.dict(
            os.environ, {
                "LETSINFER_VERIFIER_BYPASS_GITHUB_IDS": "10000001,10000002"
            }
        ):
            generate_manifest.validate_consensus_binding(runtime, consensus)


if __name__ == "__main__":
    unittest.main()
