#!/usr/bin/env python3
"""Validate and atomically apply trusted Engine pin requests."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

if __package__:
    from tools import generate_manifest, pin_engine
else:
    import generate_manifest
    import pin_engine


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
HEX256_RE = re.compile(r"[0-9a-f]{64}")
CANDIDATE_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]*(?:--[a-z0-9][a-z0-9._-]*){3}"
)
BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
PLATFORM_RE = re.compile(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*")
REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "repository",
        "pull_request",
        "proposal_base_sha",
        "proposal_head_sha",
        "proposal_tree_sha256",
        "candidate",
        "mode",
        "head_repository",
        "head_ref",
        "build_run_id",
        "build_workflow_sha",
        "finalizer_run_id",
        "finalizer_workflow_sha",
        "raw_artifact_digest",
        "engine_repository",
        "platform",
        "engine_reference",
        "engine_manifest_digest",
        "engine_config_digest",
        "runtime_blob_sha_before",
        "runtime_sha256_before",
        "runtime_blob_sha_after",
        "runtime_sha256_after",
        "patch_sha256",
        "request_key",
    }
)
ALLOWED_TRANSITION_PATHS = frozenset(
    {
        ("engine", "oci", "reference"),
        ("engine", "oci", "immutable_id"),
        ("benchmark", "contract", "tokenizer", "engine_image_sha256"),
    }
)
MAX_RUNTIME_BYTES = 4 << 20
MAX_REQUEST_BYTES = 1 << 20


class UpdateError(RuntimeError):
    pass


class StaleUpdate(UpdateError):
    pass


class ForkUpdate(UpdateError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def git_blob_sha(value: bytes) -> str:
    framed = f"blob {len(value)}\0".encode("ascii") + value
    return hashlib.sha1(framed).hexdigest()  # noqa: S324 - Git object identity is SHA-1.


def _object(path: pathlib.Path, *, limit: int = MAX_REQUEST_BYTES) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise UpdateError(f"invalid input file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpdateError(f"invalid JSON input: {path.name}") from error
    if not isinstance(value, dict):
        raise UpdateError(f"input must contain one JSON object: {path.name}")
    return value


def _differences(before: Any, after: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(before, dict) and isinstance(after, dict):
        result: set[tuple[str, ...]] = set()
        for key in set(before) | set(after):
            if key not in before or key not in after:
                result.add(path + (str(key),))
            else:
                result.update(_differences(before[key], after[key], path + (str(key),)))
        return result
    return set() if before == after else {path}


def validate_transition(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    reference: str,
    immutable_id: str,
) -> None:
    expected = copy.deepcopy(before)
    if not pin_engine.update(expected, reference, immutable_id):
        raise UpdateError("Engine pin request is already present")
    if after != expected:
        changed = sorted(".".join(path) for path in _differences(before, after))
        raise UpdateError(f"Engine pin changes fields outside the trusted contract: {changed}")
    changed_paths = _differences(before, after)
    if not changed_paths or not changed_paths.issubset(ALLOWED_TRANSITION_PATHS):
        raise UpdateError("Engine pin changes fields outside the three owned fields")
    if after.get("schema_version") != 5:
        raise UpdateError("Engine pin result is not runtime schema 5")
    generate_manifest.validate_runtime_execution_contract(after)


def _request_key(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("request_key", None)
    return sha256_digest(canonical_bytes(unsigned))


def create_request(
    *,
    before_path: pathlib.Path,
    after_path: pathlib.Path,
    patch_path: pathlib.Path,
    pull_path: pathlib.Path,
    build_path: pathlib.Path,
    audit_path: pathlib.Path,
    repositories_path: pathlib.Path,
    engine_plan_path: pathlib.Path,
    runtime_blob_sha_before: str,
    finalizer_run_id: int,
    finalizer_workflow_sha: str,
    raw_artifact_digest: str,
) -> dict[str, Any]:
    before_bytes = before_path.resolve(strict=True).read_bytes()
    after_bytes = after_path.resolve(strict=True).read_bytes()
    patch = patch_path.resolve(strict=True).read_bytes()
    if max(len(before_bytes), len(after_bytes)) > MAX_RUNTIME_BYTES or not patch:
        raise UpdateError("Engine pin input exceeds its bounded size")
    before = json.loads(before_bytes)
    after = json.loads(after_bytes)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise UpdateError("runtime.json must contain one object")
    pull = _object(pull_path)
    build = _object(build_path)
    audit = _object(audit_path)
    repositories = _object(repositories_path)
    plan = _object(engine_plan_path)
    reference = str(plan.get("reference"))
    config = str(plan.get("config_digest"))
    validate_transition(before, after, reference=reference, immutable_id=config)
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    value: dict[str, Any] = {
        "schema_version": 1,
        "repository": build.get("repository"),
        "pull_request": build.get("pull_request"),
        "proposal_base_sha": build.get("proposal_base_sha"),
        "proposal_head_sha": build.get("proposal_head_sha"),
        "proposal_tree_sha256": audit.get("candidate_tree_sha256"),
        "candidate": build.get("candidate"),
        "mode": build.get("mode"),
        "head_repository": head_repo.get("full_name"),
        "head_ref": head.get("ref"),
        "build_run_id": build.get("build_workflow_run_id"),
        "build_workflow_sha": build.get("build_workflow_sha"),
        "finalizer_run_id": finalizer_run_id,
        "finalizer_workflow_sha": finalizer_workflow_sha,
        "raw_artifact_digest": raw_artifact_digest,
        "engine_repository": repositories.get("engine_repository"),
        "platform": audit.get("target_platform"),
        "engine_reference": reference,
        "engine_manifest_digest": plan.get("manifest_digest"),
        "engine_config_digest": config,
        "runtime_blob_sha_before": runtime_blob_sha_before,
        "runtime_sha256_before": sha256_digest(before_bytes),
        "runtime_blob_sha_after": git_blob_sha(after_bytes),
        "runtime_sha256_after": sha256_digest(after_bytes),
        "patch_sha256": sha256_digest(patch),
    }
    value["request_key"] = _request_key(value)
    validate_request(value)
    return value


def validate_request(
    value: dict[str, Any],
    *,
    repository: str | None = None,
    finalizer_run_id: int | None = None,
    finalizer_workflow_sha: str | None = None,
) -> dict[str, Any]:
    if set(value) != REQUEST_FIELDS or value.get("schema_version") != 1:
        raise UpdateError("Engine pin request schema is invalid")
    if repository is not None and value.get("repository") != repository:
        raise UpdateError("Engine pin request repository differs")
    if finalizer_run_id is not None and value.get("finalizer_run_id") != finalizer_run_id:
        raise UpdateError("Engine pin request finalizer run differs")
    if finalizer_workflow_sha is not None and value.get("finalizer_workflow_sha") != finalizer_workflow_sha:
        raise UpdateError("Engine pin request finalizer workflow differs")
    if not isinstance(value.get("pull_request"), int) or value["pull_request"] <= 0:
        raise UpdateError("Engine pin request pull request is invalid")
    for field in ("build_run_id", "finalizer_run_id"):
        if not isinstance(value.get(field), int) or value[field] <= 0:
            raise UpdateError(f"Engine pin request {field} is invalid")
    for field in (
        "proposal_base_sha",
        "proposal_head_sha",
        "build_workflow_sha",
        "finalizer_workflow_sha",
        "runtime_blob_sha_before",
        "runtime_blob_sha_after",
    ):
        if COMMIT_RE.fullmatch(str(value.get(field))) is None:
            raise UpdateError(f"Engine pin request {field} is invalid")
    for field in (
        "raw_artifact_digest",
        "engine_manifest_digest",
        "engine_config_digest",
        "runtime_sha256_before",
        "runtime_sha256_after",
        "patch_sha256",
        "request_key",
    ):
        if DIGEST_RE.fullmatch(str(value.get(field))) is None:
            raise UpdateError(f"Engine pin request {field} is invalid")
    if HEX256_RE.fullmatch(str(value.get("proposal_tree_sha256"))) is None:
        raise UpdateError("Engine pin request candidate tree is invalid")
    if CANDIDATE_RE.fullmatch(str(value.get("candidate"))) is None:
        raise UpdateError("Engine pin request candidate is invalid")
    if value.get("mode") != "build-engine":
        raise UpdateError("Engine pin request is not a build-engine proposal")
    if value.get("engine_reference") != (
        f"{value.get('engine_repository')}@{value.get('engine_manifest_digest')}"
    ) or pin_engine.OCI_RE.fullmatch(str(value.get("engine_reference"))) is None:
        raise UpdateError("Engine pin request reference is invalid")
    if PLATFORM_RE.fullmatch(str(value.get("platform"))) is None:
        raise UpdateError("Engine pin request platform is invalid")
    if not all(
        isinstance(value.get(field), str) and value[field]
        for field in ("repository", "head_repository", "head_ref", "engine_repository")
    ):
        raise UpdateError("Engine pin request identity is invalid")
    if value.get("request_key") != _request_key(value):
        raise UpdateError("Engine pin request key differs")
    return value


def validate_files(
    request_path: pathlib.Path,
    patch_path: pathlib.Path,
    **identity: Any,
) -> dict[str, Any]:
    value = validate_request(_object(request_path), **identity)
    patch_path = patch_path.resolve(strict=True)
    if patch_path.is_symlink() or not patch_path.is_file():
        raise UpdateError("Engine pin patch is unavailable")
    if sha256_digest(patch_path.read_bytes()) != value["patch_sha256"]:
        raise UpdateError("Engine pin patch digest differs")
    return value


class GitHubClient(Protocol):
    def get(self, path: str) -> dict[str, Any]: ...
    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]: ...


class HttpGitHubClient:
    def __init__(self, token: str) -> None:
        if not token:
            raise UpdateError("GitHub token is unavailable")
        self.token = token

    def _request(self, url: str, *, value: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if value is None else canonical_bytes(value)
        request = urllib.request.Request(
            url,
            method="GET" if data is None else "POST",
            data=data,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "letsinfer-engine-pin-updater/1",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read(8 << 20)
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise UpdateError("GitHub returned an invalid object")
                return result
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise UpdateError(f"GitHub API request failed with HTTP {error.code}") from error
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                if attempt == 2:
                    raise UpdateError("GitHub API request failed") from error
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def get(self, path: str) -> dict[str, Any]:
        return self._request("https://api.github.com/" + path.lstrip("/"))

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        result = self._request(
            "https://api.github.com/graphql",
            value={"query": query, "variables": variables},
        )
        if result.get("errors"):
            raise UpdateError("GitHub rejected the atomic Engine pin commit")
        data = result.get("data")
        if not isinstance(data, dict):
            raise UpdateError("GitHub GraphQL response is invalid")
        return data


def _safe_branch(value: str) -> bool:
    return (
        BRANCH_RE.fullmatch(value) is not None
        and value not in {"main", "release"}
        and not value.startswith("refs/")
        and not value.endswith(("/", "."))
        and ".." not in value
        and "//" not in value
    )


def _runtime_content(client: GitHubClient, request: dict[str, Any], ref: str) -> tuple[bytes, str]:
    path = f"{request['candidate']}/runtime.json"
    encoded_path = urllib.parse.quote(path, safe="/")
    encoded_ref = urllib.parse.quote(ref, safe="")
    value = client.get(
        f"repos/{request['repository']}/contents/{encoded_path}?ref={encoded_ref}"
    )
    if value.get("type") != "file" or value.get("encoding") != "base64":
        raise UpdateError("runtime.json is not a regular GitHub blob")
    try:
        content = base64.b64decode(str(value.get("content", "")), validate=False)
    except (ValueError, TypeError) as error:
        raise UpdateError("runtime.json content is invalid") from error
    if not content or len(content) > MAX_RUNTIME_BYTES:
        raise UpdateError("runtime.json content exceeds its bound")
    blob = str(value.get("sha"))
    if COMMIT_RE.fullmatch(blob) is None or git_blob_sha(content) != blob:
        raise UpdateError("runtime.json Git blob identity differs")
    return content, blob


def _already_applied(
    client: GitHubClient, request: dict[str, Any], live_head: str
) -> bool:
    commit = client.get(f"repos/{request['repository']}/commits/{live_head}")
    parents = commit.get("parents")
    files = commit.get("files")
    message = commit.get("commit", {}).get("message") if isinstance(commit.get("commit"), dict) else None
    expected_path = f"{request['candidate']}/runtime.json"
    if (
        not isinstance(parents, list)
        or len(parents) != 1
        or parents[0].get("sha") != request["proposal_head_sha"]
        or not isinstance(files, list)
        or [item.get("filename") for item in files] != [expected_path]
        or not isinstance(message, str)
        or request["request_key"] not in message
    ):
        return False
    content, blob = _runtime_content(client, request, live_head)
    return (
        blob == request["runtime_blob_sha_after"]
        and sha256_digest(content) == request["runtime_sha256_after"]
    )


CREATE_COMMIT = """
mutation CreateEnginePin($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) {
    commit { oid url }
  }
}
"""


def apply_request(value: dict[str, Any], client: GitHubClient) -> dict[str, str]:
    request = validate_request(value)
    repository = request["repository"]
    pull = client.get(f"repos/{repository}/pulls/{request['pull_request']}")
    base = pull.get("base") if isinstance(pull.get("base"), dict) else {}
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    if pull.get("state") != "open" or pull.get("draft") is True or base.get("ref") != "main":
        raise StaleUpdate("pull request is closed, draft, or no longer targets main")
    live_head = str(head.get("sha"))
    if live_head != request["proposal_head_sha"]:
        if COMMIT_RE.fullmatch(live_head) and _already_applied(client, request, live_head):
            return {"result": "already-applied", "new_head": live_head}
        raise StaleUpdate("pull request head advanced before Engine pin application")
    if base.get("sha") != request["proposal_base_sha"]:
        raise StaleUpdate("pull request base advanced before Engine pin application")
    if head_repo.get("full_name") != repository or request["head_repository"] != repository:
        raise ForkUpdate("fork pull requests require the attested manual pin patch")
    branch = str(head.get("ref"))
    if branch != request["head_ref"] or not _safe_branch(branch):
        raise StaleUpdate("pull request branch is protected or unsafe for automatic pinning")
    branch_value = client.get(
        f"repos/{repository}/branches/{urllib.parse.quote(branch, safe='')}"
    )
    if branch_value.get("protected") is not False:
        raise StaleUpdate("pull request branch is protected")
    content, blob = _runtime_content(client, request, request["proposal_head_sha"])
    if (
        blob != request["runtime_blob_sha_before"]
        or sha256_digest(content) != request["runtime_sha256_before"]
    ):
        raise StaleUpdate("runtime.json differs from the attested Engine pin input")
    try:
        before = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpdateError("runtime.json is invalid JSON") from error
    if not isinstance(before, dict):
        raise UpdateError("runtime.json must contain one object")
    after = copy.deepcopy(before)
    pin_engine.update(after, request["engine_reference"], request["engine_config_digest"])
    validate_transition(
        before,
        after,
        reference=request["engine_reference"],
        immutable_id=request["engine_config_digest"],
    )
    pinned = pin_engine.readable_bytes(after)
    if (
        git_blob_sha(pinned) != request["runtime_blob_sha_after"]
        or sha256_digest(pinned) != request["runtime_sha256_after"]
    ):
        raise UpdateError("reconstructed Engine pin differs from the attested output")
    message = (
        "Pin verified Engine image\n\n"
        f"Trusted-Engine-Pin: {request['request_key']}\n"
        f"Finalizer-Run: {request['finalizer_run_id']}"
    )
    data = client.graphql(
        CREATE_COMMIT,
        {
            "input": {
                "branch": {
                    "repositoryNameWithOwner": repository,
                    "branchName": branch,
                },
                "expectedHeadOid": request["proposal_head_sha"],
                "message": {"headline": "Pin verified Engine image", "body": message.split("\n\n", 1)[1]},
                "fileChanges": {
                    "additions": [
                        {
                            "path": f"{request['candidate']}/runtime.json",
                            "contents": base64.b64encode(pinned).decode("ascii"),
                        }
                    ]
                },
            }
        },
    )
    created = data.get("createCommitOnBranch")
    commit = created.get("commit") if isinstance(created, dict) else None
    new_head = commit.get("oid") if isinstance(commit, dict) else None
    if COMMIT_RE.fullmatch(str(new_head)) is None:
        raise UpdateError("GitHub did not return the Engine pin commit")
    return {"result": "applied", "new_head": str(new_head)}


def _write_outputs(path: pathlib.Path, values: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    request = subparsers.add_parser("request")
    for name in (
        "before",
        "after",
        "patch",
        "pull",
        "build",
        "audit",
        "repositories",
        "engine-plan",
        "output",
    ):
        request.add_argument(f"--{name}", type=pathlib.Path, required=True)
    request.add_argument("--runtime-blob-sha-before", required=True)
    request.add_argument("--finalizer-run-id", type=int, required=True)
    request.add_argument("--finalizer-workflow-sha", required=True)
    request.add_argument("--raw-artifact-digest", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--request", type=pathlib.Path, required=True)
    validate.add_argument("--patch", type=pathlib.Path, required=True)
    validate.add_argument("--repository", required=True)
    validate.add_argument("--finalizer-run-id", type=int, required=True)
    validate.add_argument("--finalizer-workflow-sha", required=True)
    validate.add_argument("--github-output", type=pathlib.Path)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--request", type=pathlib.Path, required=True)
    apply.add_argument("--github-output", type=pathlib.Path)
    arguments = parser.parse_args()
    if arguments.command == "request":
        value = create_request(
            before_path=arguments.before,
            after_path=arguments.after,
            patch_path=arguments.patch,
            pull_path=arguments.pull,
            build_path=arguments.build,
            audit_path=arguments.audit,
            repositories_path=arguments.repositories,
            engine_plan_path=arguments.engine_plan,
            runtime_blob_sha_before=arguments.runtime_blob_sha_before,
            finalizer_run_id=arguments.finalizer_run_id,
            finalizer_workflow_sha=arguments.finalizer_workflow_sha,
            raw_artifact_digest=arguments.raw_artifact_digest,
        )
        arguments.output.write_bytes(canonical_bytes(value))
        print(f"PIN_REQUEST key={value['request_key']} pr={value['pull_request']}")
    elif arguments.command == "validate":
        value = validate_files(
            arguments.request,
            arguments.patch,
            repository=arguments.repository,
            finalizer_run_id=arguments.finalizer_run_id,
            finalizer_workflow_sha=arguments.finalizer_workflow_sha,
        )
        outputs = {
            "pull_request": value["pull_request"],
            "proposal_head_sha": value["proposal_head_sha"],
            "same_repository": value["head_repository"] == value["repository"],
            "request_key": value["request_key"],
        }
        if arguments.github_output:
            _write_outputs(arguments.github_output, outputs)
        print("PIN_REQUEST_VALID " + " ".join(f"{key}={value}" for key, value in outputs.items()))
    else:
        value = validate_request(_object(arguments.request))
        result = apply_request(value, HttpGitHubClient(os.environ.get("GH_TOKEN", "")))
        if arguments.github_output:
            _write_outputs(arguments.github_output, result)
        print(f"PIN_UPDATE result={result['result']} new_head={result['new_head']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ForkUpdate as error:
        raise SystemExit(f"FORK: {error}")
    except StaleUpdate as error:
        raise SystemExit(f"STALE: {error}")
    except (UpdateError, generate_manifest.ManifestError, pin_engine.PinError) as error:
        raise SystemExit(f"FATAL: {error}")
