from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import (
    candidate_policy,
    engine_reuse,
    engine_sbom,
    generate_manifest,
    shipit,
    verification_bot,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PublicationPolicyTests(unittest.TestCase):
    def test_engine_source_identity_ignores_runtime_only_metadata(self) -> None:
        candidate = "engine--owner--model--target"
        engine = {"path": "image/Dockerfile", "bytes": 4, "mode": 0o644, "sha256": "1" * 64}
        runtime = {"path": "runtime.json", "bytes": 4, "mode": 0o644, "sha256": "2" * 64}
        first = candidate_policy.engine_source_sha256(candidate, [engine, runtime])
        changed_runtime = dict(runtime, sha256="3" * 64)
        self.assertEqual(
            first,
            candidate_policy.engine_source_sha256(
                candidate, [engine, changed_runtime]
            ),
        )
        self.assertNotEqual(
            first,
            candidate_policy.engine_source_sha256(
                candidate, [dict(engine, sha256="4" * 64), runtime]
            ),
        )

    def test_engine_reuse_proof_binds_source_contract_and_inventory(self) -> None:
        candidate = "engine--owner--model--target"
        source = "1" * 64
        contract = "2" * 64
        audit = {
            "candidate": candidate,
            "engine_source_sha256": source,
            "engine_reference": (
                "ghcr.io/letsinferlabs/engine-images@sha256:" + "3" * 64
            ),
            "engine_config_digest": "sha256:" + "4" * 64,
            "target_platform": "linux/arm64",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inventory = root / "engine-inventory.json"
            inventory.write_bytes(
                engine_reuse.canonical_bytes(
                    {"schema_version": 1, "debian": [], "python": []}
                )
            )
            proof = {
                "schema_version": 1,
                "candidate": candidate,
                "engine_source_sha256": source,
                "builder_contract_sha256": contract,
                "candidate_audit_sha256": "5" * 64,
                "engine_reference": audit["engine_reference"],
                "engine_config_digest": audit["engine_config_digest"],
                "target_platform": audit["target_platform"],
                "engine_archive_sha256": "6" * 64,
                "engine_archive_bytes": 1,
                "engine_spdx_sha256": "7" * 64,
                "engine_spdx_bytes": 1,
                "inventory_sha256": engine_reuse.sha256_file(inventory),
                "bundle": {
                    "artifact_id": 10,
                    "artifact_digest": "sha256:" + "8" * 64,
                    "artifact_name": (
                        "verification-bundle-pr-31-" + "9" * 40
                    ),
                    "proposal_head_sha": "9" * 40,
                },
                "finalizer": {
                    "pull_request": 31,
                    "run_id": 11,
                    "workflow_sha": "a" * 40,
                },
            }
            (root / "engine-proof.json").write_bytes(
                engine_reuse.canonical_bytes(proof)
            )
            self.assertEqual(
                engine_reuse._validate_proof(
                    root, audit=audit, contract_sha256=contract
                ),
                proof,
            )
            proof["engine_source_sha256"] = "b" * 64
            (root / "engine-proof.json").write_bytes(
                engine_reuse.canonical_bytes(proof)
            )
            with self.assertRaisesRegex(engine_reuse.ReuseError, "differs"):
                engine_reuse._validate_proof(
                    root, audit=audit, contract_sha256=contract
                )

    def test_finalizer_revalidates_restored_engine_proof(self) -> None:
        candidate = "engine--owner--model--target"
        source = "1" * 64
        contract = "2" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            raw = root / "raw"
            proof_root = root / "proof-source"
            raw.mkdir()
            proof_root.mkdir()
            audit = {
                "candidate": candidate,
                "engine_source_sha256": source,
                "engine_reference": (
                    "ghcr.io/letsinferlabs/engine-images@sha256:" + "3" * 64
                ),
                "engine_config_digest": "sha256:" + "4" * 64,
                "target_platform": "linux/arm64",
            }
            (raw / "candidate-audit.json").write_bytes(
                engine_reuse.canonical_bytes(audit)
            )
            (raw / "engine.oci.tar").write_bytes(b"engine archive")
            inventory = {"schema_version": 1, "debian": [], "python": []}
            for destination in (
                raw / "engine-inventory.json",
                proof_root / "engine-inventory.json",
            ):
                destination.write_bytes(engine_reuse.canonical_bytes(inventory))
            proof = {
                "schema_version": 1,
                "candidate": candidate,
                "engine_source_sha256": source,
                "builder_contract_sha256": contract,
                "candidate_audit_sha256": "5" * 64,
                "engine_reference": audit["engine_reference"],
                "engine_config_digest": audit["engine_config_digest"],
                "target_platform": audit["target_platform"],
                "engine_archive_sha256": engine_reuse.sha256_file(
                    raw / "engine.oci.tar"
                ),
                "engine_archive_bytes": (raw / "engine.oci.tar").stat().st_size,
                "engine_spdx_sha256": "7" * 64,
                "engine_spdx_bytes": 1,
                "inventory_sha256": engine_reuse.sha256_file(
                    proof_root / "engine-inventory.json"
                ),
                "bundle": {
                    "artifact_id": 10,
                    "artifact_digest": "sha256:" + "8" * 64,
                    "artifact_name": "verification-bundle-pr-31-" + "9" * 40,
                    "proposal_head_sha": "9" * 40,
                },
                "finalizer": {
                    "pull_request": 31,
                    "run_id": 11,
                    "workflow_sha": "a" * 40,
                },
            }
            (proof_root / "engine-proof.json").write_bytes(
                engine_reuse.canonical_bytes(proof)
            )
            marker = {
                "schema_version": 1,
                "proof_artifact_id": 12,
                "proof_artifact_digest": "sha256:" + "b" * 64,
                "proof_artifact_name": f"engine-proof-{source}",
                "proof_sha256": engine_reuse.sha256_file(
                    proof_root / "engine-proof.json"
                ),
                "engine_source_sha256": source,
            }
            (raw / "engine-reuse.json").write_bytes(
                engine_reuse.canonical_bytes(marker)
            )

            def download(_artifact_id, output, *, token):
                del token
                shutil.copytree(proof_root, output)

            artifact = {
                "id": 12,
                "name": marker["proof_artifact_name"],
                "expired": False,
                "digest": marker["proof_artifact_digest"],
                "workflow_run": {"id": 11},
            }
            run = {
                "event": "workflow_run",
                "path": ".github/workflows/finalize-verifier.yml",
                "conclusion": "success",
                "head_branch": "main",
                "head_sha": "a" * 40,
            }
            with (
                mock.patch.object(engine_reuse, "builder_contract", return_value=contract),
                mock.patch.object(
                    engine_reuse,
                    "_gh_json",
                    side_effect=[artifact, run],
                ),
                mock.patch.object(engine_reuse, "_download_artifact", side_effect=download),
                mock.patch.object(engine_reuse, "_verify_attestations"),
            ):
                restored = engine_reuse.verify_restored(
                    trusted_root=ROOT, raw=raw, token="test-token"
                )
            self.assertEqual(restored, proof)
            self.assertFalse((raw / "engine-reuse.json").exists())

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
        existing_repositories = candidate_policy.publication_repositories(
            ROOT, existing
        )
        self.assertEqual(
            existing_repositories["engine_repository"],
            f"ghcr.io/letsinferlabs/engines/{existing}",
        )
        self.assertEqual(
            existing_repositories["engine_existing_reference"],
            generate_manifest.read_object(ROOT / existing / "runtime.json")["engine"][
                "oci"
            ]["reference"],
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

    def test_allowlisted_bypass_does_not_require_a_second_maintainer(self) -> None:
        with mock.patch.object(
            verification_bot, "api", return_value=[]
        ) as api:
            self.assertIsNone(
                shipit._approved_review(31, 10000001, required=False)
            )
        api.assert_called_once_with(
            "repos/letsinferlabs/runtimes/pulls/31/reviews?per_page=100",
            paginate=True,
        )

        requested_changes = [{
            "id": 1,
            "state": "CHANGES_REQUESTED",
            "user": {
                "login": "Reviewer",
                "id": 10000002,
                "type": "User",
            },
        }]
        with mock.patch.object(
            verification_bot, "api", return_value=requested_changes
        ), self.assertRaisesRegex(shipit.ShipitError, "requests changes"):
            shipit._approved_review(31, 10000001, required=False)

    def test_bypass_ignores_only_the_superseded_community_wrapper(self) -> None:
        head = "a" * 40
        runs = [
            {
                "id": 1,
                "name": "validate",
                "status": "completed",
                "conclusion": "success",
            },
            {
                "id": 2,
                "name": verification_bot.CHECK_NAME,
                "status": "completed",
                "conclusion": "success",
            },
            {
                "id": 3,
                "name": "process",
                "status": "completed",
                "conclusion": "failure",
                "details_url": (
                    "https://github.com/letsinferlabs/runtimes/actions/runs/"
                    "123/job/456"
                ),
                "app": {"slug": "github-actions"},
            },
        ]
        workflow = {
            "path": ".github/workflows/community-verification.yml",
            "head_sha": head,
            "event": "pull_request_target",
        }
        with (
            mock.patch.object(shipit, "_check_runs", return_value=runs),
            mock.patch.object(
                verification_bot, "api", return_value=workflow
            ) as api,
        ):
            shipit.require_checks(head, bypass=True)
        api.assert_called_once_with("repos/letsinferlabs/runtimes/actions/runs/123")

        workflow["path"] = ".github/workflows/validate.yml"
        with (
            mock.patch.object(shipit, "_check_runs", return_value=runs),
            mock.patch.object(verification_bot, "api", return_value=workflow),
            self.assertRaisesRegex(shipit.ShipitError, "blocking checks failed"),
        ):
            shipit.require_checks(head, bypass=True)

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

    def test_maintainer_bypass_accepts_zero_independent_verifiers(self) -> None:
        runtime, consensus = self._waived_consensus(10000001)
        consensus["qualification"]["independent_verifiers"] = 0
        consensus["verifications"] = []
        consensus["verifiers"] = []
        consensus["results"] = []
        consensus["score"] = {
            "policy": "letsinfer-throughput-geomean-of-verifier-means-v1",
            "aggregate_tps": None,
        }
        consensus["waiver"]["policy"] = "allowlisted-maintainer-bypass-v1"
        consensus.pop("consensus_id")
        consensus["consensus_id"] = hashlib.sha256(
            generate_manifest.canonical_bytes(consensus)
        ).hexdigest()

        with mock.patch.dict(
            os.environ,
            {"LETSINFER_VERIFIER_BYPASS_GITHUB_IDS": "10000001"},
            clear=True,
        ):
            generate_manifest.validate_consensus_binding(runtime, consensus)

        consensus["score"]["aggregate_tps"] = 1.0
        consensus.pop("consensus_id")
        consensus["consensus_id"] = hashlib.sha256(
            generate_manifest.canonical_bytes(consensus)
        ).hexdigest()
        with mock.patch.dict(
            os.environ,
            {"LETSINFER_VERIFIER_BYPASS_GITHUB_IDS": "10000001"},
            clear=True,
        ), self.assertRaisesRegex(
            generate_manifest.ManifestError, "bypass score is invalid"
        ):
            generate_manifest.validate_consensus_binding(runtime, consensus)

    def test_maintainer_bypass_materializes_without_verifier_comments(self) -> None:
        candidate = "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
        runtime = generate_manifest.read_object(ROOT / candidate / "runtime.json")
        release = generate_manifest.read_object(ROOT / candidate / "release.json")
        subject = {
            "candidate_id": candidate,
            "runtime_version": runtime["version"],
            "proposal_head_sha": "a" * 40,
            "execution_sha256": "b" * 64,
            "engine_oci_manifest_digest": runtime["engine"]["oci"][
                "reference"
            ].rsplit("@", 1)[-1],
            "benchmark_contract_sha256": hashlib.sha256(
                generate_manifest.canonical_bytes(runtime["benchmark"]["contract"])
            ).hexdigest(),
            "target_contract_sha256": hashlib.sha256(
                generate_manifest.canonical_bytes(runtime["target"])
            ).hexdigest(),
        }
        pull = {
            "number": 31,
            "html_url": "https://github.com/letsinferlabs/runtimes/pull/31",
            "user": {
                "login": "RuntimeAuthor",
                "id": 10000001,
                "type": "User",
            },
        }
        actor = {
            "github_login": "RuntimeAuthor",
            "github_id": 10000001,
            "github_type": "User",
        }
        comment = {
            "id": 123,
            "html_url": (
                "https://github.com/letsinferlabs/runtimes/pull/31"
                "#issuecomment-123"
            ),
        }
        with mock.patch.object(
            verification_bot, "accepted_submissions", return_value=[]
        ):
            consensus = shipit._bypass_consensus(
                pr=pull,
                candidate=candidate,
                subject=subject,
                root=ROOT,
                actor=actor,
                reason="Sole maintainer release decision",
                comment=comment,
            )

        self.assertTrue(consensus["qualification"]["passed"])
        self.assertEqual(consensus["qualification"]["independent_verifiers"], 0)
        self.assertEqual(consensus["verifiers"], [])
        self.assertIsNone(consensus["score"]["aggregate_tps"])
        self.assertEqual(
            consensus["waiver"]["policy"],
            "allowlisted-maintainer-bypass-v1",
        )
        self.assertEqual(consensus["runtime_authors"], release["authors"])
        with mock.patch.dict(
            os.environ,
            {"LETSINFER_VERIFIER_BYPASS_GITHUB_IDS": "10000001"},
            clear=True,
        ):
            generate_manifest.validate_consensus_binding(runtime, consensus)


if __name__ == "__main__":
    unittest.main()
