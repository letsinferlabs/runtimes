#!/usr/bin/env python3
"""Plan or publish one deterministic Let's Infer runtime-pack OCI artifact."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, BinaryIO


PACK_MEDIA_TYPE = "application/vnd.letsinfer.runtime.v5+tar"
CONFIG_MEDIA_TYPE = "application/vnd.letsinfer.runtime.config.v1+json"
MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
REPOSITORY_RE = re.compile(
    r"(?P<registry>[A-Za-z0-9.-]+(?::[0-9]+)?)/"
    r"(?P<repository>[A-Za-z0-9._/-]+)"
)
AUTH_PARAMETER_RE = re.compile(r'([a-z]+)="([^"]+)"')


class OciError(RuntimeError):
    pass


def compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_digest(path: pathlib.Path) -> tuple[str, int]:
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            value.update(chunk)
    return "sha256:" + value.hexdigest(), size


@dataclass(frozen=True)
class ArtifactPlan:
    registry: str
    repository: str
    tag: str
    candidate: str
    version: str
    layer: pathlib.Path
    layer_digest: str
    layer_size: int
    config: bytes
    config_digest: str
    manifest: bytes
    manifest_digest: str

    @property
    def source(self) -> str:
        return f"{self.registry}/{self.repository}@{self.manifest_digest}"

    def document(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "version": self.version,
            "tag": f"{self.registry}/{self.repository}:{self.tag}",
            "source": self.source,
            "manifest_digest": self.manifest_digest,
            "manifest_bytes": len(self.manifest),
            "config_digest": self.config_digest,
            "layer_digest": self.layer_digest,
            "layer_bytes": self.layer_size,
        }


def plan(
    layer: pathlib.Path,
    *,
    repository: str,
    candidate: str,
    version: str,
) -> ArtifactPlan:
    match = REPOSITORY_RE.fullmatch(repository)
    if match is None or any(
        part in {"", ".", ".."} for part in match["repository"].split("/")
    ):
        raise OciError("repository must be registry/contained/path")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", candidate):
        raise OciError("candidate must be a lowercase safe name")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", version):
        raise OciError("version must use semantic version syntax")
    layer = layer.resolve(strict=True)
    if layer.is_symlink() or not layer.is_file():
        raise OciError("runtime pack must be a regular file")
    layer_sha, layer_size = file_digest(layer)
    if layer_size <= 0:
        raise OciError("runtime pack cannot be empty")
    config = compact_bytes(
        {
            "candidate": candidate,
            "media_type": PACK_MEDIA_TYPE,
            "schema_version": 1,
            "version": version,
        }
    )
    config_sha = digest(config)
    manifest = compact_bytes(
        {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "config": {
                "mediaType": CONFIG_MEDIA_TYPE,
                "digest": config_sha,
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": PACK_MEDIA_TYPE,
                    "digest": layer_sha,
                    "size": layer_size,
                    "annotations": {
                        "org.opencontainers.image.title": "runtime.letsinfer"
                    },
                }
            ],
            "annotations": {
                "ai.letsinfer.candidate": candidate,
                "ai.letsinfer.version": version,
                "org.opencontainers.image.source": (
                    "https://github.com/letsinferlabs/runtimes"
                ),
            },
        }
    )
    return ArtifactPlan(
        registry=match["registry"],
        repository=match["repository"],
        tag=f"{candidate}-{version}",
        candidate=candidate,
        version=version,
        layer=layer,
        layer_digest=layer_sha,
        layer_size=layer_size,
        config=config,
        config_digest=config_sha,
        manifest=manifest,
        manifest_digest=digest(manifest),
    )


class Registry:
    def __init__(self, plan_value: ArtifactPlan, username: str, password: str) -> None:
        if not username or not password:
            raise OciError("registry username and token are required")
        self.plan = plan_value
        self.username = username
        self.password = password
        self.token = self._token()

    def _token(self) -> str:
        probe = urllib.request.Request(f"https://{self.plan.registry}/v2/")
        try:
            urllib.request.urlopen(probe, timeout=20).close()
            raise OciError("registry did not advertise bearer authentication")
        except urllib.error.HTTPError as error:
            if error.code != 401:
                raise OciError(f"registry authentication probe returned HTTP {error.code}") from error
            challenge = error.headers.get("WWW-Authenticate", "")
        if not challenge.startswith("Bearer "):
            raise OciError("registry did not advertise bearer authentication")
        parameters = dict(AUTH_PARAMETER_RE.findall(challenge))
        realm = parameters.get("realm")
        service = parameters.get("service")
        if not realm or not realm.startswith("https://") or not service:
            raise OciError("registry bearer challenge is invalid")
        query = urllib.parse.urlencode(
            {
                "service": service,
                "scope": f"repository:{self.plan.repository}:pull,push",
            }
        )
        basic = base64.b64encode(
            f"{self.username}:{self.password}".encode("utf-8")
        ).decode("ascii")
        request = urllib.request.Request(
            f"{realm}?{query}", headers={"Authorization": f"Basic {basic}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                value = json.loads(response.read(256 * 1024))
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise OciError(f"cannot obtain registry bearer token: {error}") from error
        token = value.get("token") if isinstance(value, dict) else None
        if not isinstance(token, str) or not token:
            raise OciError("registry bearer token response is invalid")
        return token

    @property
    def authorization(self) -> str:
        return f"Bearer {self.token}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        accept: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {"Authorization": self.authorization}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if accept is not None:
            headers["Accept"] = accept
        request = urllib.request.Request(
            f"https://{self.plan.registry}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, dict(response.headers), response.read(1024 * 1024)
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read(1024 * 1024)

    def _blob_exists(self, value: str) -> bool:
        status, _, _ = self._request(
            "HEAD", f"/v2/{self.plan.repository}/blobs/{value}"
        )
        if status not in {200, 404}:
            raise OciError(f"registry blob probe returned HTTP {status}")
        return status == 200

    def _upload_location(self) -> urllib.parse.SplitResult:
        status, headers, _ = self._request(
            "POST", f"/v2/{self.plan.repository}/blobs/uploads/", data=b""
        )
        location = headers.get("Location") or headers.get("location")
        if status != 202 or not location:
            raise OciError(f"registry upload start returned HTTP {status}")
        return urllib.parse.urlsplit(
            urllib.parse.urljoin(f"https://{self.plan.registry}/", location)
        )

    def _stream_upload(
        self,
        handle: BinaryIO,
        size: int,
        value: str,
    ) -> None:
        location = self._upload_location()
        if location.scheme != "https" or location.hostname != self.plan.registry.split(":", 1)[0]:
            raise OciError("registry upload redirected to an unexpected host")
        query = urllib.parse.parse_qsl(location.query, keep_blank_values=True)
        query.append(("digest", value))
        target = location.path + "?" + urllib.parse.urlencode(query)
        connection = http.client.HTTPSConnection(
            self.plan.registry, timeout=600
        )
        try:
            connection.putrequest("PUT", target)
            connection.putheader("Authorization", self.authorization)
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(size))
            connection.endheaders()
            remaining = size
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OciError("runtime pack changed while uploading")
                connection.send(chunk)
                remaining -= len(chunk)
            response = connection.getresponse()
            response.read(1024 * 1024)
            if response.status != 201:
                raise OciError(f"registry blob upload returned HTTP {response.status}")
            published = response.getheader("Docker-Content-Digest")
            if published is not None and published != value:
                raise OciError("registry stored a different blob digest")
        finally:
            connection.close()

    def upload_bytes(self, data: bytes, value: str) -> None:
        if self._blob_exists(value):
            return
        import io

        self._stream_upload(io.BytesIO(data), len(data), value)

    def upload_file(self, path: pathlib.Path, size: int, value: str) -> None:
        if self._blob_exists(value):
            return
        with path.open("rb") as handle:
            self._stream_upload(handle, size, value)

    def publish(self) -> str:
        self.upload_bytes(self.plan.config, self.plan.config_digest)
        self.upload_file(
            self.plan.layer, self.plan.layer_size, self.plan.layer_digest
        )
        tag_path = f"/v2/{self.plan.repository}/manifests/{self.plan.tag}"
        status, headers, _ = self._request(
            "PUT",
            tag_path,
            data=self.plan.manifest,
            content_type=MANIFEST_MEDIA_TYPE,
        )
        if status != 201:
            raise OciError(f"registry manifest publication returned HTTP {status}")
        published = headers.get("Docker-Content-Digest") or headers.get(
            "docker-content-digest"
        )
        if published is not None and published != self.plan.manifest_digest:
            raise OciError("registry stored a different manifest digest")
        return self.plan.source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "push"))
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--username", default=os.environ.get("OCI_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("OCI_PASSWORD"))
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()
    result = plan(
        arguments.artifact,
        repository=arguments.repository,
        candidate=arguments.candidate,
        version=arguments.version,
    )
    if arguments.command == "push":
        Registry(result, arguments.username or "", arguments.password or "").publish()
    document = result.document()
    data = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(data, encoding="utf-8")
    print(data, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OciError as error:
        raise SystemExit(f"FATAL: {error}")
