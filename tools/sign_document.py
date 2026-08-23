#!/usr/bin/env python3
"""Sign one closed-schema Let's Infer JSON document with Ed25519."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile
from typing import Any


KIND_RE = re.compile(r"letsinfer\.[a-z0-9.-]+")


class SigningError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _run(command: list[str]) -> bytes:
    try:
        return subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", "replace").strip()
        raise SigningError(detail or f"command failed: {command[0]}") from error


def sign(
    document: pathlib.Path,
    private_key: pathlib.Path,
    public_key: pathlib.Path,
    *,
    kind: str,
) -> dict[str, Any]:
    if KIND_RE.fullmatch(kind) is None:
        raise SigningError("document kind is invalid")
    for path, label in (
        (document, "document"),
        (private_key, "private key"),
        (public_key, "public key"),
    ):
        if path.is_symlink() or not path.is_file():
            raise SigningError(f"{label} must be a regular file")
    data = document.read_bytes()
    public_der = _run(
        ["openssl", "pkey", "-pubin", "-in", str(public_key), "-outform", "DER"]
    )
    with tempfile.TemporaryDirectory(prefix="letsinfer-sign-") as temporary:
        raw = pathlib.Path(temporary) / "signature.bin"
        _run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(private_key),
                "-rawin",
                "-in",
                str(document),
                "-out",
                str(raw),
            ]
        )
        signature = raw.read_bytes()
    if len(signature) != 64:
        raise SigningError("Ed25519 signature must contain exactly 64 bytes")
    return {
        "schema_version": 1,
        "algorithm": "ed25519",
        "key_id_sha256": hashlib.sha256(public_der).hexdigest(),
        "document_kind": kind,
        "document_sha256": hashlib.sha256(data).hexdigest(),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True, type=pathlib.Path)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--private-key", required=True, type=pathlib.Path)
    parser.add_argument("--public-key", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    result = sign(
        arguments.document.resolve(strict=True),
        arguments.private_key.resolve(strict=True),
        arguments.public_key.resolve(strict=True),
        kind=arguments.kind,
    )
    arguments.output.write_bytes(canonical_bytes(result))
    print(
        f"SIGNED kind={result['document_kind']} "
        f"document_sha256={result['document_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SigningError as error:
        raise SystemExit(f"FATAL: {error}")
