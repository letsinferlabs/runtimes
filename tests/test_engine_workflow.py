from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class EngineWorkflowTests(unittest.TestCase):
    def test_workflows_use_the_exact_core_pack_library_not_product_cli(self) -> None:
        workflows = [
            ROOT / ".github/workflows/validate.yml",
            ROOT / ".github/workflows/build-verifier.yml",
            ROOT / ".github/workflows/finalize-verifier.yml",
            ROOT / ".github/workflows/release.yml",
        ]
        for path in workflows:
            source = path.read_text(encoding="utf-8")
            self.assertIn("tools/pack_runtime.py", source, path.name)
            self.assertNotIn("bin/letsinfer pack", source, path.name)

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

    def test_engine_build_is_canonical_single_pass_and_thin(self) -> None:
        workflow = (ROOT / ".github/workflows/build-verifier.yml").read_text(
            encoding="utf-8"
        )
        builder = (ROOT / "tools/build_engine.py").read_text(encoding="utf-8")
        self.assertIn("tools/build_engine.py", workflow)
        self.assertIn("Build the exact Engine OCI once", workflow)
        self.assertNotIn("engine-b.oci.tar", workflow)
        self.assertNotIn("engine-b-plan.json", workflow)
        self.assertNotIn("A second no-cache build", workflow)
        self.assertIn("moby/buildkit@sha256:", builder)
        self.assertIn("--existing-reference", builder)
        self.assertIn("type=oci,dest=-", builder)
        self.assertIn("rewrite-timestamp=true", builder)
        self.assertIn("--inventory-output", workflow)
        self.assertIn('kwargs.setdefault("stdout", sys.stderr)', builder)
        self.assertIn("engine_distribution.validate", builder)
        self.assertNotIn('runtime["engine"]["oci"]', builder)

    def test_unchanged_engine_uses_a_finalizer_attested_proof(self) -> None:
        build = (ROOT / ".github/workflows/build-verifier.yml").read_text(
            encoding="utf-8"
        )
        finalizer = (ROOT / ".github/workflows/finalize-verifier.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("attestations: read", build)
        self.assertIn("tools/engine_reuse.py restore", build)
        self.assertIn("steps.engine_reuse.outputs.reused != 'true'", build)
        self.assertIn("engine_build_contract_sha256", build)
        self.assertIn("tools/engine_reuse.py verify-restored", finalizer)
        self.assertIn("tools/engine_reuse.py create-proof", finalizer)
        self.assertIn("engine-proof-${{ steps.identity.outputs.engine_source_sha256 }}", finalizer)
        self.assertIn("Attest reusable unchanged-Engine proof", finalizer)

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
        self.assertNotIn("engine-pin-pr-", workflow)
        self.assertNotIn("engine-pin-request.json", workflow)
        self.assertIn("exact authored Engine identity", workflow)
        self.assertIn("runtime/verifier-bundle", workflow)
        self.assertIn('engine.get("distribution", engine.get("oci"))', workflow)
        self.assertIn('run.get("head_branch") != "main"', workflow)
        self.assertIn("actions/attest@", workflow)
        self.assertNotIn("working-directory: proposal", workflow)

    def test_engine_pin_bot_is_removed(self) -> None:
        self.assertFalse((ROOT / ".github/workflows/auto-pin-engine.yml").exists())
        self.assertFalse((ROOT / "tools/engine_pin_updater.py").exists())
        finalizer = (ROOT / ".github/workflows/finalize-verifier.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("Engine identity differs from the canonical trusted build", finalizer)
        self.assertNotIn("createCommitOnBranch", finalizer)

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
        self.assertIn('test "$engine_kind" = oci-container', release)
        self.assertIn('engine.get("distribution", engine.get("oci"))', release)

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

    def test_release_promotion_requires_the_exact_current_main_tree(self) -> None:
        workflow = (ROOT / ".github/workflows/validate.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('test "$BASE_REF" = release', workflow)
        self.assertIn("git fetch --no-tags --depth=1 origin main", workflow)
        self.assertIn('git rev-parse "${HEAD_SHA}^{tree}"', workflow)
        self.assertIn('git rev-parse "FETCH_HEAD^{tree}"', workflow)
        self.assertIn('git rev-parse "${HEAD_SHA}^"', workflow)
        self.assertIn('= "$BASE_SHA"', workflow)

    def test_new_runtime_version_may_only_clear_stale_bot_qualification(self) -> None:
        workflow = (ROOT / ".github/workflows/validate.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("'*/benchmark.consensus.json'", workflow)
        self.assertIn("set(changes) != {('D', deleted[0]), ('M', release_path)}", workflow)
        self.assertIn("reset_release['provenance'] = None", workflow)
        self.assertIn("old_runtime.get('version') == new_runtime.get('version')", workflow)
        self.assertIn("new_release != reset_release", workflow)


if __name__ == "__main__":
    unittest.main()
