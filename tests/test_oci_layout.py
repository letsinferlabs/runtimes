from __future__ import annotations

import hashlib
import gzip
import io
import json
import pathlib
import tarfile
import tempfile
import unittest
from unittest import mock

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

    def _remote_image(
        self, archive: pathlib.Path, reference: str
    ) -> oci_layout.RemoteImage:
        with tarfile.open(archive, "r") as source:
            index = json.load(source.extractfile("index.json"))
            manifest_descriptor = index["manifests"][0]
            manifest = json.load(
                source.extractfile(
                    "blobs/sha256/"
                    + manifest_descriptor["digest"].removeprefix("sha256:")
                )
            )
            config_descriptor = manifest["config"]
            config = json.load(
                source.extractfile(
                    "blobs/sha256/"
                    + config_descriptor["digest"].removeprefix("sha256:")
                )
            )
        return oci_layout.RemoteImage(
            reference=reference,
            repository=reference.rsplit("@", 1)[0],
            manifest_digest=manifest_descriptor["digest"],
            manifest=manifest,
            config_descriptor=config_descriptor,
            config=config,
            layers=tuple(manifest["layers"]),
            diff_ids=tuple(config["rootfs"]["diff_ids"]),
        )

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

    def test_thin_layout_externalizes_verified_existing_layers(self) -> None:
        repository = "ghcr.io/letsinferlabs/engines/example"
        reference = repository + "@sha256:" + "1" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            full, _docker, manifest_digest, config_digest = self._archives(root)
            remote = self._remote_image(full, reference)
            thin = root / "engine-thin.oci.tar"
            with mock.patch.object(oci_layout, "_remote_image", return_value=remote):
                document = oci_layout.thin_archive(
                    full,
                    thin,
                    platform="linux/arm64",
                    repository=repository,
                    existing_reference=reference,
                )
                inspected = oci_layout.inspect_archive(thin, "linux/arm64")
            with tarfile.open(thin, "r") as archive:
                names = {member.name for member in archive}
            layer_digest = remote.layers[0]["digest"].removeprefix("sha256:")
        self.assertEqual(document["manifest_digest"], manifest_digest)
        self.assertEqual(document["config_digest"], config_digest)
        self.assertEqual(document["external_reference"], reference)
        self.assertEqual(document["external_layer_count"], 1)
        self.assertEqual(document["local_layer_count"], 0)
        self.assertEqual(inspected, {key: value for key, value in document.items() if key != "reference"})
        self.assertIn(oci_layout.EXTERNAL_BLOBS_FILE, names)
        self.assertNotIn(f"blobs/sha256/{layer_digest}", names)

    def test_thin_layout_without_a_prior_engine_remains_self_contained(self) -> None:
        repository = "ghcr.io/letsinferlabs/engine-images"
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            full, _docker, manifest_digest, config_digest = self._archives(root)
            compacted = root / "engine-compacted.oci.tar"
            document = oci_layout.thin_archive(
                full,
                compacted,
                platform="linux/arm64",
                repository=repository,
            )
            inspected = oci_layout.inspect_archive(compacted, "linux/arm64")
        self.assertEqual(document["manifest_digest"], manifest_digest)
        self.assertEqual(document["config_digest"], config_digest)
        self.assertEqual(document["external_layer_count"], 0)
        self.assertEqual(document["local_layer_count"], 1)
        self.assertNotIn("external_reference", document)
        self.assertEqual(inspected, {key: value for key, value in document.items() if key != "reference"})

    def test_thin_publish_requires_external_blobs_and_uploads_only_local_data(self) -> None:
        repository = "ghcr.io/letsinferlabs/engines/example"
        reference = repository + "@sha256:" + "1" * 64
        uploaded: list[str] = []

        class Registry:
            def __init__(self, **_unused: object) -> None:
                pass

            def authenticate(self, _scope: str) -> None:
                pass

            def _blob_exists(self, _value: str) -> bool:
                return True

            def upload(self, _handle: object, _size: int, value: str) -> None:
                uploaded.append(value)

            def request(self, *_args: object, **_kwargs: object) -> tuple[int, dict[str, str], bytes]:
                return 201, {}, b""

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            full, _docker, manifest_digest, config_digest = self._archives(root)
            remote = self._remote_image(full, reference)
            thin = root / "engine-thin.oci.tar"
            with mock.patch.object(oci_layout, "_remote_image", return_value=remote):
                oci_layout.thin_archive(
                    full,
                    thin,
                    platform="linux/arm64",
                    repository=repository,
                    existing_reference=reference,
                )
                with mock.patch.object(oci_layout, "Registry", Registry):
                    published = oci_layout.publish(
                        thin,
                        repository=repository,
                        platform="linux/arm64",
                        tag="verified-test",
                        username="actor",
                        password="token",
                    )
        self.assertEqual(published["manifest_digest"], manifest_digest)
        self.assertEqual(published["config_digest"], config_digest)
        self.assertEqual(uploaded, [config_digest])


if __name__ == "__main__":
    unittest.main()
