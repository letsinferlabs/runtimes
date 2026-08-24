from __future__ import annotations

import base64
import copy
import json
import pathlib
import tempfile
import unittest

from tools import engine_pin_updater, pin_engine


ROOT = pathlib.Path(__file__).resolve().parents[1]
CANDIDATE = "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"


def fixture() -> tuple[dict, bytes, bytes]:
    before = json.loads((ROOT / CANDIDATE / "runtime.json").read_text(encoding="utf-8"))
    after = copy.deepcopy(before)
    reference = "ghcr.io/letsinferlabs/engine-images@sha256:" + "1" * 64
    immutable = "sha256:" + "2" * 64
    pin_engine.update(after, reference, immutable)
    before_bytes = pin_engine.readable_bytes(before)
    after_bytes = pin_engine.readable_bytes(after)
    value = {
        "schema_version": 1,
        "repository": "letsinferlabs/runtimes",
        "pull_request": 123,
        "proposal_base_sha": "b" * 40,
        "proposal_head_sha": "a" * 40,
        "proposal_tree_sha256": "3" * 64,
        "candidate": CANDIDATE,
        "mode": "build-engine",
        "head_repository": "letsinferlabs/runtimes",
        "head_ref": "feature/example-runtime",
        "build_run_id": 100,
        "build_workflow_sha": "c" * 40,
        "finalizer_run_id": 200,
        "finalizer_workflow_sha": "d" * 40,
        "raw_artifact_digest": "sha256:" + "4" * 64,
        "engine_repository": "ghcr.io/letsinferlabs/engine-images",
        "platform": "linux/arm64",
        "engine_reference": reference,
        "engine_manifest_digest": "sha256:" + "1" * 64,
        "engine_config_digest": immutable,
        "runtime_blob_sha_before": engine_pin_updater.git_blob_sha(before_bytes),
        "runtime_sha256_before": engine_pin_updater.sha256_digest(before_bytes),
        "runtime_blob_sha_after": engine_pin_updater.git_blob_sha(after_bytes),
        "runtime_sha256_after": engine_pin_updater.sha256_digest(after_bytes),
        "patch_sha256": engine_pin_updater.sha256_digest(b"trusted patch\n"),
    }
    value["request_key"] = engine_pin_updater._request_key(value)
    return value, before_bytes, after_bytes


class FakeGitHub:
    def __init__(self, request: dict, before: bytes, after: bytes) -> None:
        self.request = request
        self.before = before
        self.after = after
        self.live_head = request["proposal_head_sha"]
        self.protected = False
        self.head_repository = request["repository"]
        self.graphql_error: Exception | None = None
        self.variables: dict | None = None

    def get(self, path: str) -> dict:
        if path.endswith(f"/pulls/{self.request['pull_request']}"):
            return {
                "state": "open",
                "draft": False,
                "base": {"ref": "main", "sha": self.request["proposal_base_sha"]},
                "head": {
                    "sha": self.live_head,
                    "ref": self.request["head_ref"],
                    "repo": {"full_name": self.head_repository},
                },
            }
        if "/branches/" in path:
            return {"protected": self.protected}
        if "/commits/" in path:
            return {
                "parents": [{"sha": self.request["proposal_head_sha"]}],
                "files": [{"filename": f"{CANDIDATE}/runtime.json"}],
                "commit": {"message": f"Trusted-Engine-Pin: {self.request['request_key']}"},
            }
        if "/contents/" in path:
            content = self.after if self.live_head != self.request["proposal_head_sha"] else self.before
            return {
                "type": "file",
                "encoding": "base64",
                "sha": engine_pin_updater.git_blob_sha(content),
                "content": base64.b64encode(content).decode("ascii"),
            }
        raise AssertionError(path)

    def graphql(self, query: str, variables: dict) -> dict:
        if self.graphql_error is not None:
            raise self.graphql_error
        self.variables = variables
        return {"createCommitOnBranch": {"commit": {"oid": "e" * 40, "url": "https://example"}}}


class EnginePinUpdaterTests(unittest.TestCase):
    def test_same_repository_update_is_atomic_and_exactly_scoped(self) -> None:
        request, before, after = fixture()
        github = FakeGitHub(request, before, after)
        result = engine_pin_updater.apply_request(request, github)
        self.assertEqual(result, {"result": "applied", "new_head": "e" * 40})
        self.assertIsNotNone(github.variables)
        value = github.variables["input"]
        self.assertEqual(value["expectedHeadOid"], request["proposal_head_sha"])
        additions = value["fileChanges"]["additions"]
        self.assertEqual([item["path"] for item in additions], [f"{CANDIDATE}/runtime.json"])
        self.assertEqual(base64.b64decode(additions[0]["contents"]), after)
        self.assertIn(request["request_key"], value["message"]["body"])

    def test_duplicate_delivery_is_a_noop(self) -> None:
        request, before, after = fixture()
        github = FakeGitHub(request, before, after)
        github.live_head = "e" * 40
        self.assertEqual(
            engine_pin_updater.apply_request(request, github),
            {"result": "already-applied", "new_head": "e" * 40},
        )
        self.assertIsNone(github.variables)

    def test_advanced_head_fails_stale(self) -> None:
        request, before, after = fixture()
        github = FakeGitHub(request, before, after)
        github.live_head = "f" * 40
        github.get = lambda path: (
            {
                "state": "open", "draft": False,
                "base": {"ref": "main", "sha": request["proposal_base_sha"]},
                "head": {"sha": "f" * 40, "ref": request["head_ref"], "repo": {"full_name": request["repository"]}},
            }
            if path.endswith("/pulls/123")
            else {"parents": [], "files": [], "commit": {"message": "unrelated"}}
        )
        with self.assertRaises(engine_pin_updater.StaleUpdate):
            engine_pin_updater.apply_request(request, github)

    def test_protected_branch_fails_closed(self) -> None:
        request, before, after = fixture()
        github = FakeGitHub(request, before, after)
        github.protected = True
        with self.assertRaises(engine_pin_updater.StaleUpdate):
            engine_pin_updater.apply_request(request, github)

    def test_fork_never_reaches_write_path(self) -> None:
        request, before, after = fixture()
        request["head_repository"] = "contributor/runtimes"
        request["request_key"] = engine_pin_updater._request_key(request)
        github = FakeGitHub(request, before, after)
        github.head_repository = "contributor/runtimes"
        with self.assertRaises(engine_pin_updater.ForkUpdate):
            engine_pin_updater.apply_request(request, github)
        self.assertIsNone(github.variables)

    def test_overbroad_transition_is_rejected(self) -> None:
        request, before_bytes, after_bytes = fixture()
        before = json.loads(before_bytes)
        after = json.loads(after_bytes)
        after["description"] = "untrusted rewrite"
        with self.assertRaises(engine_pin_updater.UpdateError):
            engine_pin_updater.validate_transition(
                before,
                after,
                reference=request["engine_reference"],
                immutable_id=request["engine_config_digest"],
            )

    def test_runtime_digest_mismatch_is_rejected(self) -> None:
        request, before, after = fixture()
        github = FakeGitHub(request, before + b" ", after)
        with self.assertRaises(engine_pin_updater.UpdateError):
            engine_pin_updater.apply_request(request, github)

    def test_permission_failure_leaves_head_unmodified(self) -> None:
        request, before, after = fixture()
        github = FakeGitHub(request, before, after)
        github.graphql_error = engine_pin_updater.UpdateError("permission denied")
        with self.assertRaises(engine_pin_updater.UpdateError):
            engine_pin_updater.apply_request(request, github)
        self.assertIsNone(github.variables)

    def test_patch_and_request_digests_are_bound(self) -> None:
        request, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            request_path = root / "engine-pin-request.json"
            patch_path = root / "engine-pin.patch"
            request_path.write_bytes(engine_pin_updater.canonical_bytes(request))
            patch_path.write_bytes(b"trusted patch\n")
            self.assertEqual(
                engine_pin_updater.validate_files(
                    request_path,
                    patch_path,
                    repository="letsinferlabs/runtimes",
                    finalizer_run_id=200,
                    finalizer_workflow_sha="d" * 40,
                ),
                request,
            )
            patch_path.write_bytes(b"changed patch\n")
            with self.assertRaises(engine_pin_updater.UpdateError):
                engine_pin_updater.validate_files(request_path, patch_path)

    def test_malformed_request_and_unsafe_branch_are_rejected(self) -> None:
        request, before, after = fixture()
        malformed = dict(request)
        malformed["unexpected"] = True
        with self.assertRaises(engine_pin_updater.UpdateError):
            engine_pin_updater.validate_request(malformed)
        request["head_ref"] = "main"
        request["request_key"] = engine_pin_updater._request_key(request)
        github = FakeGitHub(request, before, after)
        with self.assertRaises(engine_pin_updater.StaleUpdate):
            engine_pin_updater.apply_request(request, github)


if __name__ == "__main__":
    unittest.main()
