from __future__ import annotations

import pathlib
import tempfile
import unittest

from tools import audit_dockerfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DockerfileAuditTests(unittest.TestCase):
    def test_repository_dockerfiles_use_pinned_or_local_stages(self) -> None:
        for path in ROOT.glob("*/image/Dockerfile"):
            audit_dockerfile.audit(path)

    def test_local_stage_references_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "Dockerfile"
            path.write_text(
                "FROM example.invalid/base@sha256:" + "1" * 64 + " AS engine\n"
                "FROM engine AS inventory-build\n"
                "FROM scratch AS inventory\n"
                "FROM engine\n",
                encoding="utf-8",
            )
            audit_dockerfile.audit(path)

    def test_mutable_external_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "Dockerfile"
            path.write_text("FROM example.invalid/base:latest\n", encoding="utf-8")
            with self.assertRaisesRegex(
                audit_dockerfile.DockerfileAuditError, "unpinned FROM"
            ):
                audit_dockerfile.audit(path)


if __name__ == "__main__":
    unittest.main()
