#!/usr/bin/env python3
"""Pin a freshly published Engine OCI and invalidate stale qualification evidence."""

from __future__ import annotations

import argparse
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


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def update(runtime: dict[str, Any], reference: str, immutable_id: str) -> bool:
    if OCI_RE.fullmatch(reference) is None:
        raise PinError("Engine OCI reference must be digest-pinned")
    if IMAGE_ID_RE.fullmatch(immutable_id) is None:
        raise PinError("Engine OCI immutable ID must be a SHA-256 image ID")
    engine = runtime.get("engine")
    serving = runtime.get("serving")
    benchmark = runtime.get("benchmark")
    if not all(isinstance(item, dict) for item in (engine, serving, benchmark)):
        raise PinError("runtime engine, serving, or benchmark contract is invalid")
    oci = engine.get("oci")
    if not isinstance(oci, dict):
        raise PinError("runtime Engine OCI contract is missing")
    changed = (
        oci.get("reference") != reference
        or oci.get("immutable_id") != immutable_id
    )
    oci["reference"] = reference
    oci["immutable_id"] = immutable_id
    if changed:
        runtime["status"] = "candidate"
        serving["qualified"] = False
        serving["blocked_by"] = "engine-oci-requalification"
        benchmark["record"] = None
    return changed


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


def bound_benchmark_path(
    runtime_path: pathlib.Path, runtime: dict[str, Any]
) -> pathlib.Path | None:
    record = runtime.get("benchmark", {}).get("record")
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "id"}:
        raise PinError("runtime benchmark record is invalid")
    relative = pathlib.PurePosixPath(str(record.get("path", "")))
    if len(relative.parts) != 1 or relative.name != "benchmark.json":
        raise PinError("runtime benchmark record must be candidate-local benchmark.json")
    path = runtime_path.parent / relative.name
    if path.is_symlink() or not path.is_file():
        raise PinError("bound runtime benchmark record is unavailable")
    return path


def pin_runtime(
    path: pathlib.Path, reference: str, immutable_id: str
) -> tuple[dict[str, Any], bool]:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or path.name != "runtime.json":
        raise PinError("runtime must be a regular runtime.json")
    try:
        runtime = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PinError(f"cannot read runtime: {error}") from error
    if not isinstance(runtime, dict):
        raise PinError("runtime must contain one JSON object")
    benchmark_path = bound_benchmark_path(path, runtime)
    changed = update(runtime, reference, immutable_id)
    write_atomic(path, canonical_bytes(runtime))
    if changed and benchmark_path is not None:
        benchmark_path.unlink()
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
