#!/usr/bin/env python3
"""Replace one candidate's immutable runtime-pack OCI source in manifest.json."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import tempfile
from typing import Any


OCI_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


class SourceError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def update(document: dict[str, Any], candidate: str, source: str) -> None:
    if OCI_RE.fullmatch(source) is None:
        raise SourceError("runtime-pack source must be digest-pinned OCI")
    matches: list[dict[str, Any]] = []
    models = document.get("models")
    if not isinstance(models, dict):
        raise SourceError("manifest model map is invalid")
    for model in models.values():
        targets = model.get("targets") if isinstance(model, dict) else None
        if not isinstance(targets, dict):
            raise SourceError("manifest target map is invalid")
        for target in targets.values():
            candidates = target.get("candidates") if isinstance(target, dict) else None
            if not isinstance(candidates, dict):
                raise SourceError("manifest candidate map is invalid")
            record = candidates.get(candidate)
            if isinstance(record, dict):
                releases = record.get("releases")
                latest = record.get("latest")
                if isinstance(releases, dict) and isinstance(releases.get(latest), dict):
                    matches.append(releases[latest])
                else:
                    matches.append(record)
    if len(matches) != 1:
        raise SourceError("candidate must occur exactly once in manifest")
    matches[0]["source"] = source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    arguments = parser.parse_args()
    path = arguments.manifest.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise SourceError("manifest must be a regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceError(f"cannot read manifest: {error}") from error
    if not isinstance(document, dict):
        raise SourceError("manifest must contain one JSON object")
    update(document, arguments.candidate, arguments.source)
    data = canonical_bytes(document)
    descriptor, temporary_value = tempfile.mkstemp(prefix=".manifest.", dir=path.parent)
    temporary = pathlib.Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"SOURCE candidate={arguments.candidate} source={arguments.source}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourceError as error:
        raise SystemExit(f"FATAL: {error}")
