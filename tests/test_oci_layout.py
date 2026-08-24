from __future__ import annotations

import hashlib
import gzip
import io
import json
import pathlib
import tarfile
import tempfile
import unittest

from tools import oci_layout


def compact(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class OciLayoutTests(unittest.TestCase):
    def _archives(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, str, str]:
        layer_buffer = io.BytesIO()
        with tarfile.open(fileobj=layer_buffer, mode="w") as layer_archive:
            payload = b"synthetic layer file"
            record = tarfile.TarInfo("opt/engine.txt")
            record.size = len(payload)
            layer_archive.addfile(record, io.BytesIO(payload))
        uncompressed_layer = layer_buffer.getvalue()
        diff_id = oci_layout.digest(uncompressed_layer)
        config = compact({"architecture": "arm64", "os": "linux", "rootfs": {"type": "layers", "diff_ids": [diff_id]}})
        config_digest = oci_layout.digest(config)
        layer = gzip.compress(uncompressed_layer, mtime=0)
        layer_digest = oci_layout.digest(layer)
        manifest = compact(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": {
                    "mediaType": "application/vnd.oci.image.config.v1+json",
                    "digest": config_digest,
                    "size": len(config),
                },
                "layers": [
                    {
                        "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                        "digest": layer_digest,
                        "size": len(layer),
                    }
                ],
            }
        )
        manifest_digest = oci_layout.digest(manifest)
        index = compact(
            {
                "schemaVersion": 2,
                "manifests": [
                    {
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": manifest_digest,
                        "size": len(manifest),
                        "platform": {"os": "linux", "architecture": "arm64"},
                    }
                ],
            }
        )
        oci = root / "engine.oci.tar"
        with tarfile.open(oci, "w") as archive:
            for name, data in (
                ("oci-layout", compact({"imageLayoutVersion": "1.0.0"})),
                ("index.json", index),
                (f"blobs/sha256/{config_digest.removeprefix('sha256:')}", config),
                (f"blobs/sha256/{layer_digest.removeprefix('sha256:')}", layer),
                (f"blobs/sha256/{manifest_digest.removeprefix('sha256:')}", manifest),
            ):
                record = tarfile.TarInfo(name)
                record.size = len(data)
                archive.addfile(record, io.BytesIO(data))
        docker = root / "engine.docker.tar"
        config_name = config_digest.removeprefix("sha256:") + ".json"
        layer_name = "layer.tar"
        docker_manifest = compact(
            [{"Config": config_name, "RepoTags": ["test/image:head"], "Layers": [layer_name]}]
        )
        with tarfile.open(docker, "w") as archive:
            for name, data in (
                ("manifest.json", docker_manifest),
                (config_name, config),
                (layer_name, uncompressed_layer),
            ):
                record = tarfile.TarInfo(name)
                record.size = len(data)
                archive.addfile(record, io.BytesIO(data))
        return oci, docker, manifest_digest, config_digest

    def test_exact_platform_layout_and_docker_archive_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oci, docker, manifest_digest, config_digest = self._archives(pathlib.Path(temporary))
            value = oci_layout.inspect_archive(oci, "linux/arm64")
            docker_value = oci_layout.inspect_docker_archive(
                docker, expected_config=config_digest, expected_platform="linux/arm64"
            )
        self.assertEqual(value["manifest_digest"], manifest_digest)
        self.assertEqual(value["config_digest"], config_digest)
        self.assertEqual(docker_value["config_digest"], config_digest)

    def test_wrong_platform_or_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oci, docker, _manifest_digest, _config_digest = self._archives(pathlib.Path(temporary))
            with self.assertRaises(oci_layout.LayoutError):
                oci_layout.inspect_archive(oci, "linux/amd64")
            with self.assertRaises(oci_layout.LayoutError):
                oci_layout.inspect_docker_archive(
                    docker,
                    expected_config="sha256:" + "0" * 64,
                    expected_platform="linux/arm64",
                )


if __name__ == "__main__":
    unittest.main()
