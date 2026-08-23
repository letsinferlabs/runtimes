from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CatalogReleaseWorkflowTests(unittest.TestCase):
    def test_release_packs_every_runtime_advertised_by_the_catalog(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("for runtime in */runtime.json", workflow)
        self.assertIn("version in record['releases']", workflow)
        self.assertNotIn("for consensus in */benchmark.consensus.json", workflow)
        self.assertNotIn("*/benchmark.consensus.json\n", workflow)

    def test_candidate_release_is_bound_to_commit_and_exact_asset_set(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("--json targetCommitish,assets", workflow)
        self.assertIn("names != expected", workflow)
        self.assertIn("value.get('targetCommitish') != sys.argv[2]", workflow)


if __name__ == "__main__":
    unittest.main()
