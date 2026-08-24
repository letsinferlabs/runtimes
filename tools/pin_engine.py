#!/usr/bin/env python3
"""Pin a freshly published Engine OCI and invalidate stale qualification evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import re
import tempfile
from typing import Any


OCI_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")


class PinError(RuntimeError):
    pass


def readable_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def update(runtime: dict[str, Any], reference: str, immutable_id: str) -> bool:
    if OCI_RE.fullmatch(reference) is None:
        raise PinError("Engine OCI reference must be digest-pinned")
    if IMAGE_ID_RE.fullmatch(immutable_id) is None:
        raise PinError("Engine OCI immutable ID must be a SHA-256 image ID")
    engine = runtime.get("engine")
    benchmark = runtime.get("benchmark")
    if not all(isinstance(item, dict) for item in (engine, benchmark)):
        raise PinError("runtime engine or benchmark contract is invalid")
    contract = benchmark.get("contract")
    tokenizer = contract.get("tokenizer") if isinstance(contract, dict) else None
    if not isinstance(tokenizer, dict):
        raise PinError("runtime benchmark tokenizer identity is invalid")
    oci = engine.get("oci")
    if not isinstance(oci, dict):
        raise PinError("runtime Engine OCI contract is missing")
    engine_image_sha256 = immutable_id.removeprefix("sha256:")
    changed = (
        oci.get("reference") != reference
        or oci.get("immutable_id") != immutable_id
        or tokenizer.get("engine_image_sha256") != engine_image_sha256
    )
    oci["reference"] = reference
    oci["immutable_id"] = immutable_id
    tokenizer["engine_image_sha256"] = engine_image_sha256
    return changed


def update_bytes(
    content: bytes, reference: str, immutable_id: str
) -> tuple[dict[str, Any], bool, bytes]:
    """Update only the three owned JSON string values, preserving other bytes."""
    try:
        before = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PinError(f"cannot read runtime: {error}") from error
    if not isinstance(before, dict):
        raise PinError("runtime must contain one JSON object")
    after = copy.deepcopy(before)
    changed = update(after, reference, immutable_id)
    if not changed:
        return after, False, content

    paths = (
        ("engine", "oci", "reference"),
        ("engine", "oci", "immutable_id"),
        ("benchmark", "contract", "tokenizer", "engine_image_sha256"),
    )
    pinned = content
    for path in paths:
        old: Any = before
        new: Any = after
        for key in path:
            old = old[key]
            new = new[key]
        if old == new:
            continue
        if not isinstance(old, str) or not isinstance(new, str):
            raise PinError("Engine pin fields must contain JSON strings")
        old_token = json.dumps(old, ensure_ascii=False).encode("utf-8")
        new_token = json.dumps(new, ensure_ascii=False).encode("utf-8")
        if pinned.count(old_token) != 1:
            raise PinError(
                f"Engine pin field {'.'.join(path)} is not byte-unique"
            )
        pinned = pinned.replace(old_token, new_token, 1)

    try:
        reconstructed = json.loads(pinned)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PinError("byte-preserving Engine pin produced invalid JSON") from error
    if reconstructed != after:
        raise PinError("byte-preserving Engine pin changed unrelated JSON content")
    return after, True, pinned


def write_atomic(path: pathlib.Path, data: bytes) -> None:
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pin_runtime(
    path: pathlib.Path, reference: str, immutable_id: str
) -> tuple[dict[str, Any], bool]:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or path.name != "runtime.json":
        raise PinError("runtime must be a regular runtime.json")
    try:
        runtime, changed, pinned = update_bytes(
            path.read_bytes(), reference, immutable_id
        )
    except OSError as error:
        raise PinError(f"cannot read runtime: {error}") from error
    if changed:
        write_atomic(path, pinned)
    if changed:
        for name in ("benchmark.json", "benchmark.consensus.json"):
            evidence = path.parent / name
            if evidence.is_symlink():
                raise PinError(f"qualification evidence cannot be a symlink: {evidence}")
            evidence.unlink(missing_ok=True)
        release_path = path.parent / "release.json"
        if release_path.is_file() and not release_path.is_symlink():
            release = json.loads(release_path.read_text(encoding="utf-8"))
            if isinstance(release, dict) and "provenance" in release:
                release["provenance"] = None
                write_atomic(release_path, readable_bytes(release))
    return runtime, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--immutable-id", required=True)
    arguments = parser.parse_args()
    runtime, changed = pin_runtime(
        arguments.runtime, arguments.reference, arguments.immutable_id
    )
    print(f"PINNED changed={str(changed).lower()} candidate={runtime.get('id')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PinError as error:
        raise SystemExit(f"FATAL: {error}")
