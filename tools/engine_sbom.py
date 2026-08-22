#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import re
import subprocess
import sys
import urllib.parse
from typing import Any


SCHEMA_VERSION = 1
DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})")
OCI_RE = re.compile(r"ghcr\.io/letsinferlabs/engines/[^@]+@sha256:[0-9a-f]{64}")


class SbomError(ValueError):
    pass


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "dpkg-query",
            "-W",
            "-f=${binary:Package}\\t${Version}\\t${Architecture}\\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    debian = []
    for line in result.stdout.splitlines():
        name, version, architecture = line.split("\t")
        debian.append(
            {"architecture": architecture, "name": name, "version": version}
        )
    python = {
        (distribution.metadata.get("Name") or distribution.name, distribution.version)
        for distribution in importlib.metadata.distributions()
    }
    return {
        "debian": sorted(
            debian,
            key=lambda item: (
                item["name"].casefold(),
                item["version"],
                item["architecture"],
            ),
        ),
        "python": [
            {"name": name, "version": version}
            for name, version in sorted(
                python, key=lambda item: (item[0].casefold(), item[1])
            )
        ],
        "schema_version": SCHEMA_VERSION,
    }


def read_inventory(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SbomError(f"cannot read Engine package inventory: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise SbomError("unsupported Engine package inventory")
    for collection, keys in (
        ("debian", ("name", "version", "architecture")),
        ("python", ("name", "version")),
    ):
        records = value.get(collection)
        if not isinstance(records, list):
            raise SbomError(f"Engine package inventory {collection} must be an array")
        previous: tuple[str, ...] | None = None
        for record in records:
            if not isinstance(record, dict) or set(record) != set(keys):
                raise SbomError(f"invalid Engine package inventory {collection} record")
            current = tuple(str(record[key]) for key in keys)
            if any(not item for item in current) or previous == current:
                raise SbomError(f"invalid Engine package inventory {collection} value")
            previous = current
    return value


def package_id(kind: str, *values: str) -> str:
    digest = hashlib.sha256("\0".join((kind, *values)).encode()).hexdigest()[:20]
    return f"SPDXRef-Package-{kind}-{digest}"


def package(
    kind: str,
    name: str,
    version: str,
    purl: str,
    *identity: str,
) -> dict[str, Any]:
    return {
        "SPDXID": package_id(kind, name, version, *identity),
        "copyrightText": "NOASSERTION",
        "downloadLocation": "NOASSERTION",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceLocator": purl,
                "referenceType": "purl",
            }
        ],
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "name": name,
        "versionInfo": version,
    }


def spdx(
    inventory: dict[str, Any],
    candidate: str,
    image: str,
    config_digest: str,
) -> dict[str, Any]:
    image_match = DIGEST_RE.search(image)
    config_match = DIGEST_RE.fullmatch(config_digest)
    if OCI_RE.fullmatch(image) is None or image_match is None:
        raise SbomError("Engine image must be an exact Let's Infer OCI reference")
    if config_match is None:
        raise SbomError("Engine configuration identity must be a SHA-256 digest")
    image_sha = image_match.group(1)
    root_id = "SPDXRef-Package-Engine"
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": root_id,
            "checksums": [
                {"algorithm": "SHA256", "checksumValue": image_sha}
            ],
            "copyrightText": "NOASSERTION",
            "downloadLocation": image,
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "name": candidate,
            "supplier": "Organization: letsinferlabs",
            "versionInfo": config_digest,
        }
    ]
    for record in inventory["debian"]:
        encoded_name = urllib.parse.quote(record["name"], safe="._-+")
        encoded_version = urllib.parse.quote(record["version"], safe="._-+~:")
        architecture = urllib.parse.quote(record["architecture"], safe="._-+")
        packages.append(
            package(
                "deb",
                record["name"],
                record["version"],
                f"pkg:deb/{encoded_name}@{encoded_version}?arch={architecture}",
                record["architecture"],
            )
        )
    for record in inventory["python"]:
        normalized = re.sub(r"[-_.]+", "-", record["name"]).lower()
        packages.append(
            package(
                "pypi",
                record["name"],
                record["version"],
                "pkg:pypi/"
                + urllib.parse.quote(normalized, safe="._-+")
                + "@"
                + urllib.parse.quote(record["version"], safe="._-+"),
            )
        )
    relationships = [
        {
            "relatedSpdxElement": root_id,
            "relationshipType": "DESCRIBES",
            "spdxElementId": "SPDXRef-DOCUMENT",
        }
    ]
    relationships.extend(
        {
            "relatedSpdxElement": record["SPDXID"],
            "relationshipType": "CONTAINS",
            "spdxElementId": root_id,
        }
        for record in packages[1:]
    )
    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: letsinfer-engine-sbom/1"],
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": f"https://letsinfer.ai/spdx/engine/{image_sha}",
        "name": f"{candidate}-engine-{image_sha[:12]}",
        "packages": packages,
        "relationships": relationships,
        "spdxVersion": "SPDX-2.3",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    snapshot_command = commands.add_parser("snapshot")
    snapshot_command.add_argument("--output", type=pathlib.Path, required=True)
    spdx_command = commands.add_parser("spdx")
    spdx_command.add_argument("--inventory", type=pathlib.Path, required=True)
    spdx_command.add_argument("--candidate", required=True)
    spdx_command.add_argument("--image", required=True)
    spdx_command.add_argument("--config-digest", required=True)
    spdx_command.add_argument("--output", type=pathlib.Path, required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "snapshot":
            write_json(arguments.output, snapshot())
        else:
            inventory = read_inventory(arguments.inventory.resolve(strict=True))
            write_json(
                arguments.output,
                spdx(
                    inventory,
                    arguments.candidate,
                    arguments.image,
                    arguments.config_digest,
                ),
            )
    except (OSError, subprocess.CalledProcessError, SbomError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
