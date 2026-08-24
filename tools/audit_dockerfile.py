#!/usr/bin/env python3
"""Reject mutable external Dockerfile bases while allowing local stages."""

from __future__ import annotations

import argparse
import pathlib
import re


FROM_RE = re.compile(
    r"^FROM(?:\s+--platform=\S+)?\s+(?P<image>\S+)"
    r"(?:\s+[Aa][Ss]\s+(?P<alias>[A-Za-z0-9._-]+))?\s*$"
)
DIGEST_RE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")


class DockerfileAuditError(ValueError):
    """A Dockerfile contains an unsafe or malformed base reference."""


def audit(path: pathlib.Path) -> None:
    stages: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.startswith("FROM "):
            continue
        match = FROM_RE.fullmatch(raw)
        if match is None:
            raise DockerfileAuditError(f"malformed FROM in {path}:{number}: {raw}")
        image = match.group("image")
        if image != "scratch" and image not in stages and DIGEST_RE.fullmatch(image) is None:
            raise DockerfileAuditError(f"unpinned FROM in {path}:{number}: {raw}")
        alias = match.group("alias")
        if alias is not None:
            if alias in stages:
                raise DockerfileAuditError(
                    f"duplicate stage alias in {path}:{number}: {alias}"
                )
            stages.add(alias)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dockerfiles", nargs="+", type=pathlib.Path)
    arguments = parser.parse_args()
    for path in arguments.dockerfiles:
        audit(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
