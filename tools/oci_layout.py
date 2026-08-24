#!/usr/bin/env python3
"""Inspect, publish, and anonymously verify an exact OCI image layout.

The module treats layouts as untrusted data. It never executes an image or
trusts descriptor metadata until the named blob, size, and SHA-256 agree.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import pathlib
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO


DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
REFERENCE_RE = re.compile(
    r"(?P<registry>[A-Za-z0-9.-]+(?::[0-9]+)?)/"
    r"(?P<repository>[A-Za-z0-9._/-]+)@(?P<digest>sha256:[0-9a-f]{64})"
)
REPOSITORY_RE = re.compile(
    r"(?P<registry>[A-Za-z0-9.-]+(?::[0-9]+)?)/"
    r"(?P<repository>[A-Za-z0-9._/-]+)"
)
AUTH_PARAMETER_RE = re.compile(r'([a-z]+)="([^"]+)"')
MANIFEST_MEDIA_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
MAX_LAYOUT_BYTES = 16 << 30
MAX_LAYOUT_FILES = 100_000
MAX_GHCR_LAYER_BYTES = 10_000_000_000
CONFIG_MEDIA_TYPES = {
    "application/vnd.oci.image.config.v1+json",
    "application/vnd.docker.container.image.v1+json",
}
LAYER_MEDIA_TYPES = {
    "application/vnd.oci.image.layer.v1.tar",
    "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.oci.image.layer.v1.tar+zstd",
    "application/vnd.docker.image.rootfs.diff.tar",
    "application/vnd.docker.image.rootfs.diff.tar.gzip",
}
RUNTIME_CONFIG_MEDIA_TYPE = "application/vnd.letsinfer.runtime.config.v1+json"
RUNTIME_LAYER_MEDIA_TYPE = "application/vnd.letsinfer.runtime.v5+tar"


class LayoutError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _object(data: bytes, where: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LayoutError(f"{where} is not valid JSON") from error
    if not isinstance(value, dict):
        raise LayoutError(f"{where} must contain an object")
    return value


def _descriptor(value: Any, where: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("mediaType"), str)
        or DIGEST_RE.fullmatch(str(value.get("digest"))) is None
        or not isinstance(value.get("size"), int)
        or isinstance(value.get("size"), bool)
        or value["size"] < 0
    ):
        raise LayoutError(f"{where} has an invalid OCI descriptor")
    return dict(value)


def _platform(value: str) -> tuple[str, str]:
    parts = value.split("/")
    if len(parts) != 2 or not all(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", p) for p in parts):
        raise LayoutError("platform must use os/architecture syntax")
    return parts[0], parts[1]


def _extract(archive: pathlib.Path, destination: pathlib.Path) -> None:
    total = 0
    try:
        source = tarfile.open(archive, "r:*")
    except (OSError, tarfile.TarError) as error:
        raise LayoutError(f"cannot open OCI layout: {error}") from error
    with source:
        seen: set[str] = set()
        for count, member in enumerate(source, start=1):
            if count > MAX_LAYOUT_FILES:
                raise LayoutError("OCI layout contains too many entries")
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise LayoutError("OCI layout contains an unsafe path")
            name = path.as_posix()
            if name in seen:
                raise LayoutError("OCI layout contains duplicate entries")
            seen.add(name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise LayoutError("OCI layout contains a special entry")
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                continue
            if not member.isfile() or member.size < 0:
                raise LayoutError("OCI layout contains an unsupported entry")
            total += member.size
            if total > MAX_LAYOUT_BYTES:
                raise LayoutError("OCI layout exceeds the 16 GiB expansion limit")
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            extracted = source.extractfile(member)
            if extracted is None:
                raise LayoutError("OCI layout entry is unreadable")
            with target.open("xb") as output:
                shutil.copyfileobj(extracted, output, 1024 * 1024)


@dataclass(frozen=True)
class ImageLayout:
    root: pathlib.Path
    platform: str
    manifest_descriptor: dict[str, Any]
    manifest: bytes
    config_descriptor: dict[str, Any]
    reachable: tuple[dict[str, Any], ...]

    @property
    def manifest_digest(self) -> str:
        return str(self.manifest_descriptor["digest"])

    @property
    def config_digest(self) -> str:
        return str(self.config_descriptor["digest"])

    def blob(self, value: str) -> pathlib.Path:
        return self.root / "blobs" / "sha256" / value.removeprefix("sha256:")

    def document(self, repository: str | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "platform": self.platform,
            "manifest_digest": self.manifest_digest,
            "manifest_bytes": len(self.manifest),
            "config_digest": self.config_digest,
            "layer_digests": [
                item["digest"] for item in self.reachable[1:]
            ],
        }
        if repository is not None:
            if REPOSITORY_RE.fullmatch(repository) is None:
                raise LayoutError("repository must be registry/contained/path")
            value["reference"] = f"{repository}@{self.manifest_digest}"
        return value


class MaterializedLayout:
    def __init__(self, archive: pathlib.Path, platform: str) -> None:
        self.archive = archive.resolve(strict=True)
        self.platform = platform
        self.temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> ImageLayout:
        self.temporary = tempfile.TemporaryDirectory(prefix="letsinfer-oci-layout-")
        root = pathlib.Path(self.temporary.name)
        try:
            _extract(self.archive, root)
            return inspect_root(root, self.platform)
        except BaseException:
            self.temporary.cleanup()
            self.temporary = None
            raise

    def __exit__(self, *unused: object) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()


def _blob(root: pathlib.Path, descriptor: Mapping[str, Any], where: str) -> bytes:
    item = _descriptor(descriptor, where)
    path = root / "blobs" / "sha256" / str(item["digest"]).removeprefix("sha256:")
    if path.is_symlink() or not path.is_file():
        raise LayoutError(f"{where} blob is unavailable")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != item["size"]:
        raise LayoutError(f"{where} blob size differs")
    data = path.read_bytes()
    if digest(data) != item["digest"]:
        raise LayoutError(f"{where} blob digest differs")
    return data


def inspect_root(root: pathlib.Path, platform: str) -> ImageLayout:
    os_name, architecture = _platform(platform)
    layout = _object((root / "oci-layout").read_bytes(), "oci-layout")
    if layout != {"imageLayoutVersion": "1.0.0"}:
        raise LayoutError("unsupported OCI layout version")
    index_data = (root / "index.json").read_bytes()
    index = _object(index_data, "index.json")
    manifests = index.get("manifests")
    if index.get("schemaVersion") != 2 or not isinstance(manifests, list):
        raise LayoutError("OCI index is invalid")
    selected: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for offset, raw in enumerate(manifests):
        item = _descriptor(raw, f"index manifest {offset}")
        item_platform = item.get("platform")
        if isinstance(item_platform, Mapping) and (
            item_platform.get("os"), item_platform.get("architecture")
        ) == (os_name, architecture):
            candidates.append(item)
        elif (
            len(manifests) == 1
            and not isinstance(item_platform, Mapping)
            and item["mediaType"] in MANIFEST_MEDIA_TYPES
        ):
            candidates.append(item)
    if len(candidates) != 1:
        raise LayoutError("OCI layout does not contain exactly one requested platform")
    selected = candidates[0]
    selected_data = _blob(root, selected, "platform manifest")
    if selected["mediaType"] in INDEX_MEDIA_TYPES:
        nested = _object(selected_data, "platform index")
        nested_candidates = []
        for offset, raw in enumerate(nested.get("manifests", [])):
            item = _descriptor(raw, f"platform index manifest {offset}")
            item_platform = item.get("platform")
            if isinstance(item_platform, Mapping) and (
                item_platform.get("os"), item_platform.get("architecture")
            ) == (os_name, architecture):
                nested_candidates.append(item)
        if len(nested_candidates) != 1:
            raise LayoutError("nested OCI index platform selection is ambiguous")
        selected = nested_candidates[0]
        selected_data = _blob(root, selected, "platform manifest")
    if selected["mediaType"] not in MANIFEST_MEDIA_TYPES:
        raise LayoutError("selected OCI object is not an image manifest")
    manifest = _object(selected_data, "platform manifest")
    config = _descriptor(manifest.get("config"), "image config")
    layers = manifest.get("layers")
    if manifest.get("schemaVersion") != 2 or not isinstance(layers, list) or not layers:
        raise LayoutError("image manifest is invalid")
    if config["mediaType"] not in CONFIG_MEDIA_TYPES:
        raise LayoutError("image configuration media type is unsupported")
    config_data = _blob(root, config, "image config")
    image_config = _object(config_data, "image config")
    if (image_config.get("os"), image_config.get("architecture")) != (
        os_name,
        architecture,
    ):
        raise LayoutError("image configuration platform differs")
    rootfs = image_config.get("rootfs")
    diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, Mapping) else None
    if (
        not isinstance(rootfs, Mapping)
        or rootfs.get("type") != "layers"
        or not isinstance(diff_ids, list)
        or len(diff_ids) != len(layers)
        or any(DIGEST_RE.fullmatch(str(value)) is None for value in diff_ids)
    ):
        raise LayoutError("image configuration rootfs differs from manifest layers")
    reachable = [config]
    for offset, raw in enumerate(layers):
        layer = _descriptor(raw, f"image layer {offset}")
        if layer["mediaType"] not in LAYER_MEDIA_TYPES:
            raise LayoutError(f"image layer {offset} media type is unsupported")
        if layer["size"] > MAX_GHCR_LAYER_BYTES:
            raise LayoutError(f"image layer {offset} exceeds GHCR's 10 GB limit")
        _blob(root, layer, f"image layer {offset}")
        reachable.append(layer)
    return ImageLayout(
        root=root,
        platform=platform,
        manifest_descriptor=selected,
        manifest=selected_data,
        config_descriptor=config,
        reachable=tuple(reachable),
    )


def inspect_archive(archive: pathlib.Path, platform: str) -> dict[str, Any]:
    with MaterializedLayout(archive, platform) as layout:
        return layout.document()


def inspect_docker_archive(
    archive: pathlib.Path, *, expected_config: str, expected_platform: str
) -> dict[str, Any]:
    """Validate a Docker-load archive and bind it to the OCI config identity."""

    os_name, architecture = _platform(expected_platform)
    archive = archive.resolve(strict=True)
    try:
        source = tarfile.open(archive, "r:*")
    except (OSError, tarfile.TarError) as error:
        raise LayoutError(f"cannot open Docker image archive: {error}") from error
    with source:
        members = source.getmembers()
        if len(members) > MAX_LAYOUT_FILES:
            raise LayoutError("Docker image archive contains too many entries")
        names: set[str] = set()
        total = 0
        regular: dict[str, tarfile.TarInfo] = {}
        for member in members:
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise LayoutError("Docker image archive contains an unsafe path")
            name = path.as_posix()
            if name in names:
                raise LayoutError("Docker image archive contains duplicate entries")
            names.add(name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise LayoutError("Docker image archive contains a special entry")
            if member.isfile():
                total += member.size
                regular[name] = member
                if total > MAX_LAYOUT_BYTES:
                    raise LayoutError("Docker image archive exceeds the 16 GiB limit")

        def read(name: str, limit: int) -> bytes:
            member = regular.get(name)
            if member is None or member.size > limit:
                raise LayoutError(f"Docker image archive is missing bounded {name}")
            handle = source.extractfile(member)
            if handle is None:
                raise LayoutError(f"Docker image archive cannot read {name}")
            data = handle.read(limit + 1)
            if len(data) != member.size:
                raise LayoutError(f"Docker image archive {name} size differs")
            return data

        manifest_value = json.loads(read("manifest.json", 1 << 20))
        if not isinstance(manifest_value, list) or len(manifest_value) != 1:
            raise LayoutError("Docker image archive must contain exactly one image")
        record = manifest_value[0]
        if not isinstance(record, dict):
            raise LayoutError("Docker image archive manifest entry is invalid")
        config_name = record.get("Config")
        layers = record.get("Layers")
        if (
            not isinstance(config_name, str)
            or not isinstance(layers, list)
            or not layers
            or any(not isinstance(item, str) for item in layers)
        ):
            raise LayoutError("Docker image archive manifest is invalid")
        config_data = read(config_name, 4 << 20)
        config_digest = digest(config_data)
        if config_digest != expected_config:
            raise LayoutError("Docker archive image configuration differs from OCI layout")
        config = _object(config_data, "Docker image config")
        if (config.get("os"), config.get("architecture")) != (os_name, architecture):
            raise LayoutError("Docker archive image platform differs")
        rootfs = config.get("rootfs")
        diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, Mapping) else None
        if (
            not isinstance(rootfs, Mapping)
            or rootfs.get("type") != "layers"
            or not isinstance(diff_ids, list)
            or len(diff_ids) != len(layers)
            or any(DIGEST_RE.fullmatch(str(value)) is None for value in diff_ids)
        ):
            raise LayoutError("Docker image rootfs differs from archive layers")
        for offset, layer in enumerate(layers):
            layer_data = read(layer, MAX_LAYOUT_BYTES)
            if digest(layer_data) != diff_ids[offset]:
                raise LayoutError(f"Docker archive layer {offset} differs from image rootfs")
        return {
            "schema_version": 1,
            "config_digest": config_digest,
            "platform": expected_platform,
            "layer_count": len(layers),
            "archive_sha256": file_digest(archive),
            "archive_bytes": archive.stat().st_size,
        }


class Registry:
    def __init__(
        self,
        *,
        registry: str,
        repository: str,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.registry = registry
        self.repository = repository
        self.username = username
        self.password = password
        self.token: str | None = None

    def authenticate(self, scope: str) -> None:
        probe = urllib.request.Request(f"https://{self.registry}/v2/")
        try:
            urllib.request.urlopen(probe, timeout=20).close()
            return
        except urllib.error.HTTPError as error:
            if error.code != 401:
                raise LayoutError(f"registry probe returned HTTP {error.code}") from error
            challenge = error.headers.get("WWW-Authenticate", "")
        if not challenge.startswith("Bearer "):
            raise LayoutError("registry did not advertise bearer authentication")
        parameters = dict(AUTH_PARAMETER_RE.findall(challenge))
        realm, service = parameters.get("realm"), parameters.get("service")
        if not realm or not realm.startswith("https://") or not service:
            raise LayoutError("registry bearer challenge is invalid")
        query = urllib.parse.urlencode(
            {"service": service, "scope": f"repository:{self.repository}:{scope}"}
        )
        headers = {}
        if self.username is not None and self.password is not None:
            basic = base64.b64encode(
                f"{self.username}:{self.password}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {basic}"
        request = urllib.request.Request(f"{realm}?{query}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                value = json.loads(response.read(256 << 10))
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise LayoutError(f"cannot obtain registry token: {error}") from error
        token = value.get("token") if isinstance(value, dict) else None
        if not isinstance(token, str) or not token:
            raise LayoutError("registry token response is invalid")
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        accept: str | None = None,
        limit: int = 4 << 20,
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {}
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        if content_type:
            headers["Content-Type"] = content_type
        if accept:
            headers["Accept"] = accept
        request = urllib.request.Request(
            f"https://{self.registry}{path}", data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.status, dict(response.headers), response.read(limit + 1)
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read(limit + 1)

    def _blob_exists(self, value: str) -> bool:
        status, _headers, _body = self.request(
            "HEAD", f"/v2/{self.repository}/blobs/{value}"
        )
        if status not in {200, 404}:
            raise LayoutError(f"registry blob probe returned HTTP {status}")
        return status == 200

    def _upload_location(self) -> urllib.parse.SplitResult:
        status, headers, _body = self.request(
            "POST", f"/v2/{self.repository}/blobs/uploads/", data=b""
        )
        location = headers.get("Location") or headers.get("location")
        if status != 202 or not location:
            raise LayoutError(f"registry upload start returned HTTP {status}")
        parsed = urllib.parse.urlsplit(
            urllib.parse.urljoin(f"https://{self.registry}/", location)
        )
        if parsed.scheme != "https" or parsed.hostname != self.registry.split(":", 1)[0]:
            raise LayoutError("registry upload redirected to an unexpected host")
        return parsed

    def upload(self, handle: BinaryIO, size: int, value: str) -> None:
        if self._blob_exists(value):
            return
        location = self._upload_location()
        query = urllib.parse.parse_qsl(location.query, keep_blank_values=True)
        query.append(("digest", value))
        target = location.path + "?" + urllib.parse.urlencode(query)
        connection = http.client.HTTPSConnection(self.registry, timeout=600)
        try:
            connection.putrequest("PUT", target)
            if self.token is not None:
                connection.putheader("Authorization", f"Bearer {self.token}")
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(size))
            connection.endheaders()
            remaining = size
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise LayoutError("OCI blob changed during upload")
                connection.send(chunk)
                remaining -= len(chunk)
            response = connection.getresponse()
            response.read(1 << 20)
            if response.status != 201:
                raise LayoutError(f"registry blob upload returned HTTP {response.status}")
        finally:
            connection.close()


def publish(
    archive: pathlib.Path,
    *,
    repository: str,
    platform: str,
    tag: str,
    username: str,
    password: str,
) -> dict[str, Any]:
    match = REPOSITORY_RE.fullmatch(repository)
    if match is None or any(part in {"", ".", ".."} for part in match["repository"].split("/")):
        raise LayoutError("repository must be registry/contained/path")
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}", tag) is None:
        raise LayoutError("OCI tag is invalid")
    if not username or not password:
        raise LayoutError("registry username and token are required")
    with MaterializedLayout(archive, platform) as layout:
        registry = Registry(
            registry=match["registry"], repository=match["repository"],
            username=username, password=password,
        )
        registry.authenticate("pull,push")
        for descriptor in layout.reachable:
            path = layout.blob(str(descriptor["digest"]))
            with path.open("rb") as handle:
                registry.upload(handle, int(descriptor["size"]), str(descriptor["digest"]))
        status, headers, _body = registry.request(
            "PUT",
            f"/v2/{match['repository']}/manifests/{tag}",
            data=layout.manifest,
            content_type=str(layout.manifest_descriptor["mediaType"]),
        )
        if status != 201:
            raise LayoutError(f"registry manifest publish returned HTTP {status}")
        published = headers.get("Docker-Content-Digest") or headers.get("docker-content-digest")
        if published is not None and published != layout.manifest_digest:
            raise LayoutError("registry published a different manifest digest")
        return layout.document(repository)


def verify_reference(
    reference: str,
    *,
    expected_config: str | None = None,
    expected_platform: str | None = None,
) -> dict[str, Any]:
    match = REFERENCE_RE.fullmatch(reference)
    if match is None:
        raise LayoutError("Engine reference must be registry/repository@sha256:digest")
    registry = Registry(registry=match["registry"], repository=match["repository"])
    registry.authenticate("pull")
    accept = ", ".join(sorted(MANIFEST_MEDIA_TYPES))
    status, headers, body = registry.request(
        "GET", f"/v2/{match['repository']}/manifests/{match['digest']}", accept=accept
    )
    if status != 200 or len(body) > (4 << 20):
        raise LayoutError(f"registry manifest fetch returned HTTP {status}")
    if digest(body) != match["digest"]:
        raise LayoutError("registry returned a different manifest digest")
    manifest = _object(body, "registry manifest")
    config = _descriptor(manifest.get("config"), "registry image config")
    if expected_config is not None and config["digest"] != expected_config:
        raise LayoutError("published Engine configuration digest differs")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise LayoutError("published Engine manifest has no layers")
    is_runtime_artifact = config["mediaType"] == RUNTIME_CONFIG_MEDIA_TYPE
    if config["mediaType"] not in CONFIG_MEDIA_TYPES and not is_runtime_artifact:
        raise LayoutError("published artifact configuration media type is unsupported")
    state, _config_headers, config_body = registry.request(
        "GET", f"/v2/{match['repository']}/blobs/{config['digest']}", limit=4 << 20
    )
    if (
        state != 200
        or len(config_body) != config["size"]
        or digest(config_body) != config["digest"]
    ):
        raise LayoutError("published image configuration differs")
    image_config = _object(config_body, "published image configuration")
    if is_runtime_artifact:
        if (
            expected_platform is not None
            or image_config.get("schema_version") != 1
            or image_config.get("media_type") != RUNTIME_LAYER_MEDIA_TYPE
            or not isinstance(image_config.get("candidate"), str)
            or not isinstance(image_config.get("version"), str)
            or len(layers) != 1
        ):
            raise LayoutError("published runtime artifact configuration is invalid")
        layer = _descriptor(layers[0], "runtime artifact layer")
        if layer["mediaType"] != RUNTIME_LAYER_MEDIA_TYPE:
            raise LayoutError("published runtime artifact layer media type is invalid")
        state, _headers, _body = registry.request(
            "HEAD", f"/v2/{match['repository']}/blobs/{layer['digest']}"
        )
        if state != 200:
            raise LayoutError("published runtime artifact layer is unavailable")
        returned = headers.get("Docker-Content-Digest") or headers.get(
            "docker-content-digest"
        )
        if returned is not None and returned != match["digest"]:
            raise LayoutError("registry content-digest header differs")
        return {
            "schema_version": 1,
            "reference": reference,
            "manifest_digest": match["digest"],
            "config_digest": config["digest"],
            "layer_count": 1,
            "anonymous_pull_verified": True,
        }
    if expected_platform is not None:
        os_name, architecture = _platform(expected_platform)
        if (image_config.get("os"), image_config.get("architecture")) != (
            os_name,
            architecture,
        ):
            raise LayoutError("published image platform differs")
    rootfs = image_config.get("rootfs")
    diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, Mapping) else None
    if (
        not isinstance(rootfs, Mapping)
        or rootfs.get("type") != "layers"
        or not isinstance(diff_ids, list)
        or len(diff_ids) != len(layers)
        or any(DIGEST_RE.fullmatch(str(value)) is None for value in diff_ids)
    ):
        raise LayoutError("published image rootfs differs from manifest layers")
    for offset, raw in enumerate(layers):
        item = _descriptor(raw, f"registry layer {offset}")
        if item["mediaType"] not in LAYER_MEDIA_TYPES:
            raise LayoutError(f"published image layer {offset} media type is unsupported")
        if item["size"] > MAX_GHCR_LAYER_BYTES:
            raise LayoutError(f"published image layer {offset} exceeds GHCR's 10 GB limit")
        state, _headers, _body = registry.request(
            "HEAD", f"/v2/{match['repository']}/blobs/{item['digest']}"
        )
        if state != 200:
            raise LayoutError(f"published Engine blob is unavailable: {item['digest']}")
    returned = headers.get("Docker-Content-Digest") or headers.get("docker-content-digest")
    if returned is not None and returned != match["digest"]:
        raise LayoutError("registry content-digest header differs")
    return {
        "schema_version": 1,
        "reference": reference,
        "manifest_digest": match["digest"],
        "config_digest": config["digest"],
        "layer_count": len(layers),
        "anonymous_pull_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--archive", required=True, type=pathlib.Path)
    inspect.add_argument("--platform", required=True)
    inspect.add_argument("--repository")
    inspect.add_argument("--output", type=pathlib.Path)
    push = commands.add_parser("push")
    push.add_argument("--archive", required=True, type=pathlib.Path)
    push.add_argument("--platform", required=True)
    push.add_argument("--repository", required=True)
    push.add_argument("--tag", required=True)
    push.add_argument("--username", required=True)
    push.add_argument("--password", required=True)
    push.add_argument("--output", type=pathlib.Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--reference", required=True)
    verify.add_argument("--config-digest")
    verify.add_argument("--platform")
    verify.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()
    if arguments.command == "inspect":
        with MaterializedLayout(arguments.archive, arguments.platform) as layout:
            value = layout.document(arguments.repository)
    elif arguments.command == "push":
        value = publish(
            arguments.archive, repository=arguments.repository, platform=arguments.platform,
            tag=arguments.tag, username=arguments.username, password=arguments.password,
        )
    else:
        value = verify_reference(
            arguments.reference,
            expected_config=arguments.config_digest,
            expected_platform=arguments.platform,
        )
    data = canonical_bytes(value)
    if arguments.output:
        arguments.output.write_bytes(data)
    print(data.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LayoutError as error:
        raise SystemExit(f"FATAL: {error}")
