from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class EngineWorkflowTests(unittest.TestCase):
    def test_untrusted_pr_builder_has_no_publication_authority(self) -> None:
        workflow = (ROOT / ".github/workflows/build-verifier.yml").read_text(
            encoding="utf-8"
        )
        sentinel = (ROOT / ".github/workflows/request-verifier.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("pull_request:", sentinel)
        self.assertIn("'*--*--*--*/**'", sentinel)
        self.assertNotIn("actions/checkout@", sentinel)
        self.assertIn("workflow_run:", workflow)
        self.assertIn("workflows: [Request verifier artifact]", workflow)
        self.assertIn("contents: read", workflow)
        self.assertNotIn("packages: write", workflow)
        self.assertNotIn("pull_request_target", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertIn("raw-verifier-pr-", workflow)
        self.assertIn('head != identity["trigger_sha"]', workflow)
        self.assertNotIn(
            'pull.get("merge_commit_sha") != identity["trigger_sha"]', workflow
        )
        self.assertIn("path: trusted", workflow)
        self.assertIn("working-directory: trusted", workflow)

    def test_engine_builds_are_reproducible_and_verifier_loadable(self) -> None:
        workflow = (ROOT / ".github/workflows/build-verifier.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("dest=$raw/engine.oci.tar", workflow)
        self.assertIn("dest=$raw/engine-b.oci.tar", workflow)
        self.assertIn('cmp "$raw/engine-a-plan.json" "$raw/engine-b-plan.json"', workflow)
        self.assertNotIn("type=docker,dest=", workflow)
        self.assertIn("--build-arg SOURCE_DATE_EPOCH=0", workflow)
        self.assertIn("rewrite-timestamp=true", workflow)
        self.assertIn("--target letsinfer-engine-inventory", workflow)
        self.assertEqual(workflow.count("--build-context letsinfer-tools=."), 3)
        self.assertNotIn("--build-context letsinfer-tools=../proposal", workflow)

    def test_verification_bot_uses_a_read_only_attestation_token(self) -> None:
        workflow = (
            ROOT / ".github/workflows/community-verification.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("attestations: read", workflow)
        self.assertIn(
            "LETSINFER_ATTESTATION_TOKEN: ${{ github.token }}", workflow
        )

    def test_trusted_finalizer_never_executes_proposal_code(self) -> None:
        workflow = (ROOT / ".github/workflows/finalize-verifier.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("workflow_run:", workflow)
        self.assertIn("ref: ${{ github.workflow_sha }}", workflow)
        self.assertIn("path: proposal", workflow)
        self.assertIn("trusted-classification.json", workflow)
        self.assertIn("proposal_base_sha", workflow)
        self.assertIn("without executing proposal code", workflow)
        self.assertIn("tools/verifier_bundle.py finalize", workflow)
        self.assertIn("engine-pin-pr-", workflow)
        self.assertIn("runtime/verifier-bundle", workflow)
        self.assertIn('run.get("head_branch") != "main"', workflow)
        self.assertIn("actions/attest@", workflow)
        self.assertNotIn("working-directory: proposal", workflow)

    def test_shipit_is_the_only_engine_registry_publisher(self) -> None:
        self.assertFalse((ROOT / ".github/workflows/publish-engine.yml").exists())
        shipit = (ROOT / ".github/workflows/shipit.yml").read_text(encoding="utf-8")
        self.assertIn("issue_comment:", shipit)
        self.assertIn("production-runtime-publication", shipit)
        self.assertIn("packages: write", shipit)
        self.assertIn("attestations: read", shipit)
        self.assertIn("LETSINFER_ATTESTATION_TOKEN: ${{ github.token }}", shipit)
        self.assertIn("tools.shipit", shipit)
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertNotIn("packages: write", release)
        self.assertNotIn("oci_artifact.py push", release)
        self.assertIn("oci_layout.py verify", release)

    def test_every_engine_exports_the_same_inventory_contract(self) -> None:
        for dockerfile in ROOT.glob("*/image/Dockerfile"):
            source = dockerfile.read_text(encoding="utf-8")
            self.assertIn("AS letsinfer-engine-inventory-build", source)
            self.assertIn("FROM scratch AS letsinfer-engine-inventory", source)
            self.assertIn("tools/engine_sbom.py", source)

    def test_runtime_release_uses_the_current_attestation_action(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("artifact-metadata: write", workflow)
        self.assertIn("actions/attest@", workflow)
        self.assertNotIn("actions/attest-build-provenance@", workflow)


if __name__ == "__main__":
    unittest.main()
