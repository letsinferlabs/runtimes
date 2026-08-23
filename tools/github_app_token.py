#!/usr/bin/env python3
"""Mint one short-lived repository-scoped GitHub App installation token."""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any


class TokenError(RuntimeError):
    pass


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def jwt(app_id: int, private_key: pathlib.Path, *, now: int | None = None) -> str:
    timestamp = int(time.time()) if now is None else now
    header = _b64(b'{"alg":"RS256","typ":"JWT"}')
    payload = _b64(
        json.dumps(
            {"iat": timestamp - 60, "exp": timestamp + 540, "iss": str(app_id)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    unsigned = f"{header}.{payload}".encode("ascii")
    try:
        signature = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", str(private_key)],
            input=unsigned,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise TokenError("cannot sign GitHub App JWT") from error
    return f"{unsigned.decode('ascii')}.{_b64(signature)}"


def request(
    method: str,
    endpoint: str,
    token: str,
    *,
    value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    if method == "POST":
        data = json.dumps(
            {} if value is None else value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    request_value = urllib.request.Request(
        "https://api.github.com" + endpoint,
        method=method,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "letsinfer-verification-bot/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request_value, timeout=30) as response:
            raw = response.read(1 << 20)
    except (OSError, urllib.error.URLError) as error:
        raise TokenError(f"GitHub App API request failed: {endpoint}") from error
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenError("GitHub App API returned invalid JSON") from error
    if not isinstance(document, dict):
        raise TokenError("GitHub App API returned an invalid object")
    return document


def installation_token(app_id: int, private_key: pathlib.Path) -> str:
    signed = jwt(app_id, private_key)
    installation = request(
        "GET", "/repos/letsinferlabs/runtimes/installation", signed
    )
    installation_id = installation.get("id")
    if not isinstance(installation_id, int) or installation_id <= 0:
        raise TokenError("GitHub App is not installed on letsinferlabs/runtimes")
    created = request(
        "POST",
        f"/app/installations/{installation_id}/access_tokens",
        signed,
        value={
            "repositories": ["runtimes"],
            "permissions": {
                "checks": "write",
                "contents": "write",
                "issues": "write",
                "metadata": "read",
                "pull_requests": "write",
            },
        },
    )
    token = created.get("token")
    if not isinstance(token, str) or not token.startswith("ghs_"):
        raise TokenError("GitHub did not return an installation token")
    return token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", required=True, type=int)
    parser.add_argument("--private-key", required=True, type=pathlib.Path)
    parser.add_argument("--github-output", type=pathlib.Path)
    arguments = parser.parse_args()
    key = arguments.private_key.resolve(strict=True)
    if key.is_symlink() or not key.is_file():
        raise TokenError("GitHub App private key must be a regular file")
    token = installation_token(arguments.app_id, key)
    output = arguments.github_output or pathlib.Path(os.environ["GITHUB_OUTPUT"])
    print(f"::add-mask::{token}")
    with output.open("a", encoding="utf-8") as handle:
        handle.write(f"token={token}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TokenError as error:
        raise SystemExit(f"FATAL: {error}")
