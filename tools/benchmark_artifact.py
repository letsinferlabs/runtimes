#!/usr/bin/env python3
"""Plan or publish immutable benchmark evidence as an OCI artifact."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from typing import Any

from tools.oci_artifact import (
    ArtifactPlan,
    MANIFEST_MEDIA_TYPE,
    OciError,
    REPOSITORY_RE,
    Registry,
    compact_bytes,
    digest,
    file_digest,
)


BENCHMARK_MEDIA_TYPE = "application/vnd.letsinfer.benchmark.v1+json"
CONFIG_MEDIA_TYPE = "application/vnd.letsinfer.benchmark.config.v1+json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OCI_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


def _object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OciError(f"benchmark evidence is invalid JSON: {error}") from error
    if not isinstance(value, dict) or not SHA256_RE.fullmatch(str(value.get("id"))):
        raise OciError("benchmark evidence must contain a valid benchmark id")
    return value


def plan(
    evidence: pathlib.Path,
    *,
    repository: str,
    candidate: str,
    version: str,
    runtime_plan: pathlib.Path,
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
    evidence = evidence.resolve(strict=True)
    if evidence.is_symlink() or not evidence.is_file():
        raise OciError("benchmark evidence must be a regular file")
    document = _object(evidence)
    try:
        runtime = json.loads(runtime_plan.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OciError(f"runtime OCI plan is invalid: {error}") from error
    if (
        not isinstance(runtime, dict)
        or runtime.get("candidate") != candidate
        or runtime.get("version") != version
        or not OCI_RE.fullmatch(str(runtime.get("source")))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(runtime.get("manifest_digest")))
        or not isinstance(runtime.get("manifest_bytes"), int)
        or runtime["manifest_bytes"] <= 0
    ):
        raise OciError("benchmark runtime plan does not match the candidate release")
    layer_sha, layer_size = file_digest(evidence)
    if layer_size <= 0:
        raise OciError("benchmark evidence cannot be empty")
    config = compact_bytes(
        {
            "benchmark_id": document["id"],
            "candidate": candidate,
            "media_type": BENCHMARK_MEDIA_TYPE,
            "runtime_source": runtime["source"],
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
            "subject": {
                "mediaType": MANIFEST_MEDIA_TYPE,
                "digest": runtime["manifest_digest"],
                "size": runtime["manifest_bytes"],
            },
            "layers": [
                {
                    "mediaType": BENCHMARK_MEDIA_TYPE,
                    "digest": layer_sha,
                    "size": layer_size,
                    "annotations": {
                        "org.opencontainers.image.title": "benchmark.json"
                    },
                }
            ],
            "annotations": {
                "ai.letsinfer.benchmark.id": document["id"],
                "ai.letsinfer.candidate": candidate,
                "ai.letsinfer.runtime.source": runtime["source"],
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
        tag=f"{candidate}-{version}-{document['id'][:12]}".replace("+", "-"),
        candidate=candidate,
        version=version,
        layer=evidence,
        layer_digest=layer_sha,
        layer_size=layer_size,
        config=config,
        config_digest=config_sha,
        manifest=manifest,
        manifest_digest=digest(manifest),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "push"))
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--runtime-plan", type=pathlib.Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--username", default=os.environ.get("OCI_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("OCI_PASSWORD"))
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()
    result = plan(
        arguments.evidence,
        repository=arguments.repository,
        candidate=arguments.candidate,
        version=arguments.version,
        runtime_plan=arguments.runtime_plan,
    )
    if arguments.command == "push":
        Registry(result, arguments.username or "", arguments.password or "").publish()
    evidence = _object(arguments.evidence.resolve(strict=True))
    runtime = json.loads(arguments.runtime_plan.resolve(strict=True).read_text())
    document = result.document() | {
        "benchmark_id": evidence["id"],
        "runtime_source": runtime["source"],
    }
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
