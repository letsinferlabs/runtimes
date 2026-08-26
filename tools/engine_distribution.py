#!/usr/bin/env python3
"""Closed Engine distribution validation for repository tooling."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from typing import Any


SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
OCI_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
PATH_RE = re.compile(
    r"(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*"
)


class DistributionError(ValueError):
    pass


def _path(value: Any, where: str) -> None:
    if not isinstance(value, str) or PATH_RE.fullmatch(value) is None:
        raise DistributionError(f"{where} is invalid")


def _archive(value: Any, where: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "url", "sha256", "bytes", "format", "strip_prefix"
    }:
        raise DistributionError(f"{where} fields are invalid")
    url = urllib.parse.urlsplit(str(value.get("url", "")))
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise DistributionError(f"{where} URL must use credential-free HTTPS")
    if re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None:
        raise DistributionError(f"{where} SHA-256 is invalid")
    if (
        not isinstance(value.get("bytes"), int)
        or isinstance(value.get("bytes"), bool)
        or not 0 < value["bytes"] <= 1 << 30
        or value.get("format") not in {"tar.gz", "zip"}
    ):
        raise DistributionError(f"{where} size or format is invalid")
    _path(value.get("strip_prefix"), f"{where} strip_prefix")


def validate(value: Any, *, platform: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DistributionError("Engine distribution must be an object")
    result = dict(value)
    kind = result.get("kind")
    if kind == "oci-container":
        if set(result) not in (
            {"kind", "reference", "immutable_id"},
            {"kind", "reference", "immutable_id", "base"},
            {"kind", "reference", "immutable_id", "payload_id"},
            {"kind", "reference", "immutable_id", "base", "payload_id"},
        ):
            raise DistributionError("OCI Engine distribution fields are invalid")
        if OCI_RE.fullmatch(str(result.get("reference", ""))) is None:
            raise DistributionError("OCI Engine reference is not digest-pinned")
        if SHA256_RE.fullmatch(str(result.get("immutable_id", ""))) is None:
            raise DistributionError("OCI Engine immutable_id is invalid")
        if "base" in result and OCI_RE.fullmatch(str(result["base"])) is None:
            raise DistributionError("OCI Engine base is not digest-pinned")
        if "payload_id" in result and SHA256_RE.fullmatch(
            str(result["payload_id"])
        ) is None:
            raise DistributionError("OCI Engine payload_id is invalid")
        return result
    common = {
        "kind", "platform", "payload_id", "source_revision", "entrypoint",
        "port_count",
    }
    if result.get("platform") != platform:
        raise DistributionError("native Engine platform differs from target")
    if SHA256_RE.fullmatch(str(result.get("payload_id", ""))) is None:
        raise DistributionError("native Engine payload_id is invalid")
    if REVISION_RE.fullmatch(str(result.get("source_revision", ""))) is None:
        raise DistributionError("native Engine source_revision is invalid")
    _path(result.get("entrypoint"), "native Engine entrypoint")
    if (
        not isinstance(result.get("port_count"), int)
        or isinstance(result.get("port_count"), bool)
        or result["port_count"] not in range(1, 5)
    ):
        raise DistributionError("native Engine port_count is invalid")
    if kind == "native-archive":
        if set(result) != common | {"archive", "upstream_executable"}:
            raise DistributionError("native archive Engine fields are invalid")
        _archive(result["archive"], "native Engine archive")
        _path(result["upstream_executable"], "native Engine executable")
        if result["port_count"] < 2:
            raise DistributionError("native archive Engine requires two ports")
    elif kind == "python-standalone":
        if set(result) != common | {"python", "requirements_lock"}:
            raise DistributionError("Python Engine fields are invalid")
        python = result.get("python")
        if not isinstance(python, Mapping) or set(python) != {
            "implementation", "version", "archive"
        } or python.get("implementation") != "cpython" or re.fullmatch(
            r"3\.(?:1[0-9]|[89])\.[0-9]+", str(python.get("version", ""))
        ) is None:
            raise DistributionError("Python Engine identity is invalid")
        _archive(python["archive"], "native Engine Python archive")
        _path(result["requirements_lock"], "native Engine requirements lock")
        if result["port_count"] < 2:
            raise DistributionError("Python Engine requires two ports")
    elif kind == "embedded-application":
        if set(result) != common | {
            "bundle_id", "signing_policy", "minimum_version", "embedded_engine"
        }:
            raise DistributionError("embedded Engine fields are invalid")
        if result.get("signing_policy") != "deployment-managed":
            raise DistributionError("embedded Engine signing policy is invalid")
        if re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)+",
            str(result.get("bundle_id", "")),
        ) is None:
            raise DistributionError("embedded Engine bundle_id is invalid")
        if re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?",
            str(result.get("minimum_version", "")),
        ) is None:
            raise DistributionError("embedded Engine minimum_version is invalid")
        if re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*", str(result.get("embedded_engine", ""))
        ) is None:
            raise DistributionError("embedded Engine name is invalid")
    else:
        raise DistributionError("Engine distribution kind is unsupported")
    return result


def projection(value: Any, *, platform: str) -> dict[str, Any]:
    distribution = validate(value, platform=platform)
    if distribution["kind"] == "oci-container":
        result = {
            "kind": "oci-container",
            "reference": distribution["reference"],
        }
        if distribution.get("payload_id") is not None:
            result["payload_id"] = distribution["payload_id"]
        return result
    return {
        "kind": distribution["kind"],
        "platform": distribution["platform"],
        "payload_id": distribution["payload_id"],
        "source_revision": distribution["source_revision"],
    }
