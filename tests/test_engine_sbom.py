from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tools import engine_sbom


class EngineSbomTests(unittest.TestCase):
    def test_spdx_is_deterministic_and_bound_to_exact_engine(self) -> None:
        inventory = {
            "debian": [
                {"architecture": "arm64", "name": "libc6", "version": "2.39"}
            ],
            "python": [{"name": "SGLang", "version": "0.5.8"}],
            "schema_version": 1,
        }
        image = "ghcr.io/letsinferlabs/engines/example@sha256:" + "1" * 64
        config = "sha256:" + "2" * 64
        first = engine_sbom.spdx(inventory, "example", image, config)
        second = engine_sbom.spdx(inventory, "example", image, config)
        self.assertEqual(first, second)
        self.assertEqual(first["packages"][0]["downloadLocation"], image)
        self.assertEqual(first["packages"][0]["versionInfo"], config)
        self.assertEqual(len(first["packages"]), 3)
        self.assertEqual(len(first["relationships"]), 3)

    def test_inventory_reader_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "inventory.json"
            path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            with self.assertRaisesRegex(engine_sbom.SbomError, "unsupported"):
                engine_sbom.read_inventory(path)


if __name__ == "__main__":
    unittest.main()
