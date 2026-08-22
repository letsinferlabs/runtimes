from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class EngineWorkflowTests(unittest.TestCase):
    def test_manual_dispatch_sboms_the_existing_pin_without_rebuilding(self) -> None:
        workflow = (ROOT / ".github/workflows/publish-engine.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("Resolve the pinned Engine for a manual SBOM retry", workflow)
        self.assertIn("rebuild:", workflow)
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && !inputs.rebuild",
            workflow,
        )
        self.assertIn(
            "if: github.event_name != 'workflow_dispatch' || inputs.rebuild",
            workflow,
        )
        self.assertIn("steps.pinned.outputs.platform_reference", workflow)

    def test_engine_build_normalizes_exported_layer_timestamps(self) -> None:
        workflow = (ROOT / ".github/workflows/publish-engine.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("--build-arg SOURCE_DATE_EPOCH=0", workflow)
        self.assertIn("rewrite-timestamp=true", workflow)

    def test_actions_publish_a_review_branch_without_creating_a_pr(self) -> None:
        workflow = (ROOT / ".github/workflows/publish-engine.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("Engine pin review branch", workflow)
        self.assertNotIn("gh pr create", workflow)

    def test_engine_sbom_uses_build_time_package_inventory(self) -> None:
        workflow = (ROOT / ".github/workflows/publish-engine.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("--target letsinfer-engine-inventory", workflow)
        self.assertIn("tools/engine_sbom.py spdx", workflow)
        self.assertIn("steps.engine.outputs.immutable_id", workflow)
        self.assertIn("actions/upload-artifact@", workflow)
        self.assertNotIn("anchore/sbom-action", workflow)

    def test_every_engine_exports_the_same_inventory_contract(self) -> None:
        for dockerfile in ROOT.glob("*/image/Dockerfile"):
            source = dockerfile.read_text(encoding="utf-8")
            self.assertIn("AS letsinfer-engine-inventory-build", source)
            self.assertIn("FROM scratch AS letsinfer-engine-inventory", source)
            self.assertIn("tools/engine_sbom.py", source)


if __name__ == "__main__":
    unittest.main()
