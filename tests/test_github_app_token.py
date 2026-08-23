#!/usr/bin/env python3
"""GitHub App authentication helper regressions."""

from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import github_app_token


class GitHubAppTokenTests(unittest.TestCase):
    def test_jwt_is_bounded_and_identifies_the_app(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            key = pathlib.Path(temporary) / "key.pem"
            subprocess.run(
                [
                    "openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt",
                    "rsa_keygen_bits:2048", "-out", str(key),
                ],
                check=True,
                capture_output=True,
            )
            value = github_app_token.jwt(12345, key, now=1_000_000)
        header, payload, signature = value.split(".")
        decoded = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        self.assertEqual(
            decoded, {"exp": 1_000_540, "iat": 999_940, "iss": "12345"}
        )
        self.assertTrue(header)
        self.assertTrue(signature)

    def test_installation_token_is_repository_and_permission_scoped(self) -> None:
        with mock.patch.object(github_app_token, "jwt", return_value="app-jwt"), \
                mock.patch.object(github_app_token, "request") as request:
            request.side_effect = [{"id": 123}, {"token": "ghs_example"}]
            token = github_app_token.installation_token(
                7, pathlib.Path("/does/not/need/to/exist")
            )
        self.assertEqual(token, "ghs_example")
        self.assertEqual(
            request.call_args_list,
            [
                mock.call(
                    "GET",
                    "/repos/letsinferlabs/runtimes/installation",
                    "app-jwt",
                ),
                mock.call(
                    "POST",
                    "/app/installations/123/access_tokens",
                    "app-jwt",
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
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
