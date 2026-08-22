#!/usr/bin/env python3
"""Create the closed Ed25519 signature document consumed by Let's Infer core."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import subprocess
import tempfile
from typing import Any


class SigningError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def run(command: list[str], *, input_data: bytes | None = None) -> bytes:
    try:
        return subprocess.run(
            command,
            input=input_data,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", "replace").strip()
        raise SigningError(detail or f"command failed: {command[0]}") from error


def sign(
    catalog: pathlib.Path,
    private_key: pathlib.Path,
    public_key: pathlib.Path,
) -> dict[str, Any]:
    for path, label in (
        (catalog, "catalog"),
        (private_key, "private key"),
        (public_key, "public key"),
    ):
        if path.is_symlink() or not path.is_file():
            raise SigningError(f"{label} must be a regular file")
    data = catalog.read_bytes()
    public_der = run(
        ["openssl", "pkey", "-pubin", "-in", str(public_key), "-outform", "DER"]
    )
    with tempfile.TemporaryDirectory(prefix="letsinfer-sign-") as temporary:
        signature_path = pathlib.Path(temporary) / "signature.bin"
        run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(private_key),
                "-rawin",
                "-in",
                str(catalog),
                "-out",
                str(signature_path),
            ]
        )
        signature = signature_path.read_bytes()
    if len(signature) != 64:
        raise SigningError("Ed25519 signature must contain exactly 64 bytes")
    return {
        "schema_version": 1,
        "algorithm": "ed25519",
        "key_id_sha256": hashlib.sha256(public_der).hexdigest(),
        "catalog_sha256": hashlib.sha256(data).hexdigest(),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--private-key", type=pathlib.Path, required=True)
    parser.add_argument("--public-key", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    document = sign(
        arguments.catalog.resolve(strict=True),
        arguments.private_key.resolve(strict=True),
        arguments.public_key.resolve(strict=True),
    )
    arguments.output.write_bytes(canonical_bytes(document))
    print(
        "SIGNED catalog_sha256=" + document["catalog_sha256"]
        + " key_id_sha256=" + document["key_id_sha256"]
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SigningError as error:
        raise SystemExit(f"FATAL: {error}")
