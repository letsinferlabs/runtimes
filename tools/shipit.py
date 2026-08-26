#!/usr/bin/env python3
"""Trusted maintainer-only publisher for one reviewed runtime proposal."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

if __package__:
    from tools import community_verification, generate_manifest, oci_artifact, oci_layout
    from tools import verification_bot, verifier_bundle
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from tools import community_verification, generate_manifest, oci_artifact, oci_layout
    from tools import verification_bot, verifier_bundle


REPOSITORY = "letsinferlabs/runtimes"
CHECK_NAME = "runtime/shipit"
RECEIPT_MARKER = "<!-- letsinfer-shipit:v1 -->"
PLAIN_COMMAND = "/shipit"
BYPASS_COMMAND = "/shipit --bypass-verifiers"
ALLOWED_BOT_FILES = {"manifest.json"}
MIN_GH_VERSION = (2, 97, 0)
FINALIZER_CERT_IDENTITY = (
    "https://github.com/letsinferlabs/runtimes/"
    ".github/workflows/finalize-verifier.yml@refs/heads/main"
)


class ShipitError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def parse_command(body: str) -> tuple[bool, str | None]:
    normalized = body.replace("\r\n", "\n").strip()
    if normalized == PLAIN_COMMAND:
        return False, None
    lines = normalized.splitlines()
    if len(lines) == 2 and lines[0].strip() == BYPASS_COMMAND:
        prefix = "Reason:"
        if not lines[1].startswith(prefix):
            raise ShipitError("bypass requires `Reason: ...` on the second line")
        reason = lines[1][len(prefix) :].strip()
        if not reason or len(reason.encode("utf-8")) > 1000:
            raise ShipitError("bypass reason must contain 1 to 1000 UTF-8 bytes")
        return True, reason
    raise ShipitError(
        "command must be exactly `/shipit` or `/shipit --bypass-verifiers` plus `Reason: ...`"
    )


def require_configured_bypass_actor(actor_id: int, configured: str) -> None:
    values = configured.split(",") if configured else []
    if (
        not values
        or any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in values)
        or len(values) != len({int(value) for value in values})
        or actor_id not in {int(value) for value in values}
    ):
        raise ShipitError(
            "verifier bypass is restricted to configured maintainer account IDs"
        )


def _identity(value: Any, where: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("login"), str)
        or not value["login"]
        or not isinstance(value.get("id"), int)
        or isinstance(value.get("id"), bool)
        or value["id"] <= 0
        or value.get("type") != "User"
    ):
        raise ShipitError(f"{where} identity is invalid")
    return {
        "github_login": value["login"],
        "github_id": value["id"],
        "github_type": value["type"],
    }


def _permission(identity: Mapping[str, Any]) -> str:
    login = str(identity["github_login"])
    value = verification_bot.api(
        f"repos/{REPOSITORY}/collaborators/{login}/permission"
    )
    permission = value.get("permission") if isinstance(value, dict) else None
    user = value.get("user") if isinstance(value, dict) else None
    if (
        permission not in {"admin", "maintain", "write", "triage", "read", "none"}
        or not isinstance(user, Mapping)
        or user.get("id") != identity["github_id"]
        or user.get("type") != "User"
    ):
        raise ShipitError("GitHub collaborator permission is invalid")
    return str(permission)


def _pull(number: int, *, require_open: bool = True) -> dict[str, Any]:
    value = verification_bot.api(f"repos/{REPOSITORY}/pulls/{number}")
    if (
        not isinstance(value, dict)
        or value.get("number") != number
        or value.get("base", {}).get("ref") != "main"
        or not isinstance(value.get("head", {}).get("sha"), str)
        or not isinstance(value.get("html_url"), str)
    ):
        raise ShipitError("issue is not a runtimes pull request against main")
    if require_open and (value.get("state") != "open" or value.get("draft") is True):
        raise ShipitError("pull request must be open and ready for review")
    return value


def _verify_bundle_attestations(root: pathlib.Path) -> None:
    token = os.environ.get("LETSINFER_ATTESTATION_TOKEN", "")
    if not token:
        raise ShipitError("trusted attestation token is unavailable")
    version = subprocess.run(
        ["gh", "--version"], capture_output=True, check=False
    )
    match = re.search(rb"gh version ([0-9]+)\.([0-9]+)\.([0-9]+)", version.stdout)
    actual = tuple(map(int, match.groups())) if match is not None else None
    if version.returncode or actual is None or actual < MIN_GH_VERSION:
        raise ShipitError("GitHub CLI 2.97.0 or newer is required for safe attestation verification")
    environment = dict(os.environ)
    environment["GH_TOKEN"] = token
    paths = sorted(root.iterdir(), key=lambda value: value.name)
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ShipitError("verifier bundle contains a non-regular entry")
    for path in paths:
        result = subprocess.run(
            [
                "gh",
                "attestation",
                "verify",
                str(path),
                "--repo",
                REPOSITORY,
                "--cert-identity",
                FINALIZER_CERT_IDENTITY,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
        )
        if len(result.stdout) > (4 << 20) or len(result.stderr) > (4 << 20):
            raise ShipitError("attestation verifier output exceeded its bounded limit")
        if result.returncode:
            raise ShipitError(f"verifier bundle attestation is invalid: {path.name}")


def _changed_candidate(number: int) -> str:
    candidates = verification_bot.changed_runtime_candidates(number)
    if len(candidates) != 1:
        raise ShipitError("/shipit requires exactly one changed runtime candidate")
    return candidates[0]


def _approved_review(
    number: int,
    pull_author_id: int,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    values = verification_bot._flatten_pages(
        verification_bot.api(
            f"repos/{REPOSITORY}/pulls/{number}/reviews?per_page=100", paginate=True
        )
    )
    latest: dict[int, dict[str, Any]] = {}
    for review in values:
        user = review.get("user")
        user_id = user.get("id") if isinstance(user, dict) else None
        if isinstance(user_id, int):
            prior = latest.get(user_id)
            if prior is None or int(review.get("id", 0)) > int(prior.get("id", 0)):
                latest[user_id] = review
    if any(review.get("state") == "CHANGES_REQUESTED" for review in latest.values()):
        raise ShipitError("a current review requests changes")
    if not required:
        return None
    for review in latest.values():
        user = review.get("user")
        if (
            review.get("state") == "APPROVED"
            and isinstance(user, dict)
            and user.get("id") != pull_author_id
            and _permission(_identity(user, "reviewer")) in {"admin", "maintain"}
        ):
            return review
    raise ShipitError("one current non-author maintainer approval is required")


def _check_runs(head: str) -> list[dict[str, Any]]:
    value = verification_bot.api(
        f"repos/{REPOSITORY}/commits/{head}/check-runs?per_page=100"
    )
    records = value.get("check_runs") if isinstance(value, dict) else None
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ShipitError("GitHub check-run response is invalid")
    latest: dict[str, dict[str, Any]] = {}
    for item in records:
        name = item.get("name")
        if isinstance(name, str):
            prior = latest.get(name)
            if prior is None or int(item.get("id", 0)) > int(prior.get("id", 0)):
                latest[name] = item
    return list(latest.values())


def _bypassed_community_wrapper(
    item: Mapping[str, Any], *, head: str, checks: Mapping[str, Mapping[str, Any]]
) -> bool:
    authoritative = checks.get(verification_bot.CHECK_NAME)
    return (
        isinstance(authoritative, Mapping)
        and authoritative.get("status") == "completed"
        and authoritative.get("conclusion") == "success"
        and _community_wrapper_for_head(item, head=head)
    )


def _community_wrapper_for_head(item: Mapping[str, Any], *, head: str) -> bool:
    app = item.get("app")
    match = re.fullmatch(
        r"https://github\.com/letsinferlabs/runtimes/actions/runs/"
        r"([1-9][0-9]*)/job/[1-9][0-9]*",
        str(item.get("details_url")),
    )
    if (
        item.get("name") != "process"
        or not isinstance(app, Mapping)
        or app.get("slug") != "github-actions"
        or match is None
    ):
        return False
    run = verification_bot.api(
        f"repos/{REPOSITORY}/actions/runs/{int(match[1])}"
    )
    return (
        isinstance(run, Mapping)
        and run.get("path") == ".github/workflows/community-verification.yml"
        and run.get("head_sha") == head
        and run.get("event") in {"pull_request_target", "issue_comment"}
    )


def finalize_bypassed_community_check(
    head: str, *, wait_seconds: int = 1800
) -> None:
    """Complete the authoritative check after its workflow wrapper has settled.

    Materializing waived consensus creates a new pull-request head. The community
    workflow can publish a newer pending check after shipit first records the
    override. Wait for that trusted wrapper, then update its authoritative check
    in place so branch protection cannot observe the stale pending run.
    """
    deadline = time.monotonic() + wait_seconds
    while True:
        runs = _check_runs(head)
        authoritative = next(
            (
                item
                for item in runs
                if item.get("name") == verification_bot.CHECK_NAME
            ),
            None,
        )
        wrappers = [
            item
            for item in runs
            if item.get("name") == "process"
            and _community_wrapper_for_head(item, head=head)
        ]
        if (
            isinstance(authoritative, Mapping)
            and wrappers
            and all(item.get("status") == "completed" for item in wrappers)
        ):
            check_id = authoritative.get("id")
            if not isinstance(check_id, int) or check_id <= 0:
                raise ShipitError("community verification check identity is invalid")
            verification_bot.api(
                f"repos/{REPOSITORY}/check-runs/{check_id}",
                method="PATCH",
                value={
                    "status": "completed",
                    "conclusion": "success",
                    "output": {
                        "title": "Maintainer verification override applied",
                        "summary": (
                            "An allowlisted maintainer applied the audited "
                            "verifier override."
                        ),
                    },
                },
            )
            return
        if time.monotonic() >= deadline:
            raise ShipitError(
                "community verification wrapper did not settle before publication"
            )
        time.sleep(5)


def require_checks(head: str, *, bypass: bool, wait_seconds: int = 0) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        runs = _check_runs(head)
        by_name = {str(item.get("name")): item for item in runs}
        required = {"validate"}
        if not bypass:
            required.add(verification_bot.CHECK_NAME)
        pending = [
            name
            for name in required
            if name not in by_name or by_name[name].get("status") != "completed"
        ]
        failures = [
            str(item.get("name"))
            for item in runs
            if item.get("status") == "completed"
            and item.get("conclusion") not in {"success", "neutral", "skipped"}
            and item.get("name") not in {CHECK_NAME}
            and not (bypass and item.get("name") == verification_bot.CHECK_NAME)
            and not (
                bypass
                and _bypassed_community_wrapper(item, head=head, checks=by_name)
            )
        ]
        bad_required = [
            name
            for name in required
            if name in by_name
            and by_name[name].get("status") == "completed"
            and by_name[name].get("conclusion") not in {"success", "neutral", "skipped"}
        ]
        if failures or bad_required:
            raise ShipitError("blocking checks failed: " + ", ".join(sorted(set(failures + bad_required))))
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise ShipitError("required checks are incomplete: " + ", ".join(sorted(pending)))
        time.sleep(20)


def _bot_only_head(
    *, proposal: str, current: str, candidate: str, bot_login: str
) -> None:
    if proposal == current:
        return
    comparison = verification_bot.api(
        f"repos/{REPOSITORY}/compare/{proposal}...{current}"
    )
    if not isinstance(comparison, dict) or comparison.get("status") != "ahead":
        raise ShipitError("current PR head is not a descendant of the verified proposal")
    allowed = ALLOWED_BOT_FILES | {
        f"{candidate}/benchmark.consensus.json",
        f"{candidate}/release.json",
    }
    files = comparison.get("files")
    commits = comparison.get("commits")
    if (
        not isinstance(files, list)
        or any(item.get("filename") not in allowed for item in files if isinstance(item, dict))
        or not isinstance(commits, list)
        or not commits
    ):
        raise ShipitError("post-verification head changes are not bot-owned qualification files")
    for commit in commits:
        author = commit.get("author") if isinstance(commit, dict) else None
        login = author.get("login") if isinstance(author, dict) else None
        if not isinstance(login, str) or login.casefold() != bot_login.casefold():
            raise ShipitError("post-verification commit is not owned by the configured bot")


def _download_artifact(
    *,
    name: str,
    expected_pr: int,
    expected_head: str,
    expected_base: str,
    destination: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    response = verification_bot.api(
        f"repos/{REPOSITORY}/actions/artifacts?name={name}&per_page=100"
    )
    artifacts = response.get("artifacts") if isinstance(response, dict) else None
    exact = [
        item
        for item in artifacts or []
        if isinstance(item, dict)
        and item.get("name") == name
        and item.get("expired") is False
        and isinstance(item.get("id"), int)
    ]
    if len(exact) != 1:
        raise ShipitError("exact verifier bundle artifact is unavailable or ambiguous")
    artifact = exact[0]
    workflow = artifact.get("workflow_run")
    run_id = workflow.get("id") if isinstance(workflow, dict) else None
    if not isinstance(run_id, int) or run_id <= 0:
        raise ShipitError("verifier bundle finalizer run identity is invalid")
    finalizer = verification_bot.api(
        f"repos/{REPOSITORY}/actions/runs/{run_id}"
    )
    if (
        not isinstance(finalizer, dict)
        or finalizer.get("event") != "workflow_run"
        or finalizer.get("path") != ".github/workflows/finalize-verifier.yml"
        or finalizer.get("conclusion") != "success"
        or finalizer.get("head_branch") != "main"
    ):
        raise ShipitError("verifier bundle finalizer identity is invalid")
    archive = destination / "bundle.zip"
    with archive.open("xb") as output:
        result = subprocess.run(
            ["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode:
        raise ShipitError("cannot download verifier bundle")
    root = destination / "bundle"
    root.mkdir(mode=0o700)
    try:
        source = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as error:
        raise ShipitError("verifier bundle artifact is not a zip") from error
    with source:
        members = source.infolist()
        if not members or len(members) > 32:
            raise ShipitError("verifier bundle artifact file count is invalid")
        total = 0
        seen: set[str] = set()
        for member in members:
            path = pathlib.PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                path.is_absolute()
                or len(path.parts) != 1
                or any(part in {"", ".", ".."} for part in path.parts)
                or member.filename in seen
                or (mode and not stat.S_ISREG(mode))
            ):
                raise ShipitError("verifier bundle artifact contains an unsafe path")
            seen.add(member.filename)
            total += member.file_size
            if total > (20 << 30) or member.compress_size > (20 << 30) or member.is_dir():
                raise ShipitError("verifier bundle artifact exceeds its bounds")
            target = root / member.filename
            with source.open(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    archive.unlink(missing_ok=True)
    bundle = verifier_bundle.validate(
        root, expected_pr=expected_pr, expected_head=expected_head
    )
    _verify_bundle_attestations(root)
    identity = bundle.get("finalizer_workflow")
    if not isinstance(identity, dict) or identity.get("run_id") != run_id or identity.get("workflow_sha") != finalizer.get("head_sha"):
        raise ShipitError("verifier bundle does not bind its trusted finalizer")
    build_id = bundle.get("build_workflow", {}).get("run_id")
    build = verification_bot.api(f"repos/{REPOSITORY}/actions/runs/{build_id}")
    if (
        not isinstance(build, dict)
        or build.get("event") != "workflow_run"
        or build.get("path") != ".github/workflows/build-verifier.yml"
        or build.get("conclusion") != "success"
        or build.get("head_branch") != "main"
        or build.get("head_sha") != expected_base
        or bundle.get("build_workflow", {}).get("workflow_sha") != expected_base
    ):
        raise ShipitError("verifier bundle does not bind its untrusted build run")
    return root, bundle


def _existing_model_has_release(
    root: pathlib.Path, runtime: Mapping[str, Any]
) -> bool:
    manifest = generate_manifest.read_object(root / "manifest.json")
    model = manifest.get("models", {}).get(runtime["logical_model"])
    target = (
        model.get("targets", {}).get(runtime["target"]["id"])
        if isinstance(model, Mapping)
        else None
    )
    existing = (
        target.get("candidates", {}) if isinstance(target, Mapping) else {}
    )
    return any(
        isinstance(record, Mapping) and bool(record.get("releases"))
        for record in existing.values()
    )


def _cheap_verifier_ids(number: int) -> set[int]:
    comments = verification_bot._flatten_pages(
        verification_bot.api(
            f"repos/{REPOSITORY}/issues/{number}/comments?per_page=100"
        )
    )
    return {
        int(user["id"])
        for item in comments
        if isinstance(item, Mapping)
        and verification_bot.SUBMISSION_MARKER
        in str(item.get("body", ""))
        and isinstance((user := item.get("user")), Mapping)
        and isinstance(user.get("id"), int)
        and not isinstance(user["id"], bool)
        and user["id"] > 0
    }


def _preflight_publication(
    *, number: int, candidate: str, root: pathlib.Path, bypass: bool
) -> None:
    """Reject missing evidence before retrieving any verifier artifact."""

    runtime = generate_manifest.read_object(root / candidate / "runtime.json")
    verifier_ids = _cheap_verifier_ids(number)
    if not bypass and len(verifier_ids) < community_verification.POLICY[
        "required_verifiers"
    ]:
        raise ShipitError(
            "independent verification is incomplete; verifier artifact was not downloaded"
        )
    if (
        bypass
        and not verifier_ids
        and not (root / candidate / "benchmark.json").is_file()
        and _existing_model_has_release(root, runtime)
    ):
        raise ShipitError(
            "maintainer bypass requires benchmark.json for an existing model; "
            "verifier artifact was not downloaded"
        )


def _bypass_consensus(
    *,
    pr: Mapping[str, Any],
    candidate: str,
    subject: Mapping[str, Any],
    root: pathlib.Path,
    actor: Mapping[str, Any],
    reason: str,
    comment: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = generate_manifest.read_object(root / candidate / "runtime.json")
    release = generate_manifest.read_object(root / candidate / "release.json")
    submissions = verification_bot.accepted_submissions(int(pr["number"]), subject=subject)
    author = {
        "github_login": pr["user"]["login"],
        "github_id": pr["user"]["id"],
        "github_type": pr["user"]["type"],
    }
    if submissions:
        consensus = community_verification.build_consensus(
            candidate_id=candidate,
            runtime_version=runtime["version"],
            pull_request=int(pr["number"]),
            pull_request_url=str(pr["html_url"]),
            proposal_head_sha=str(subject["proposal_head_sha"]),
            author=author,
            runtime_authors=release["authors"],
            accepted_comments=submissions,
        )
    else:
        benchmark_path = root / candidate / "benchmark.json"
        author_result: dict[str, Any] | None = None
        if benchmark_path.is_file():
            benchmark = generate_manifest.read_object(benchmark_path)
            generate_manifest.validate_benchmark_binding(runtime, benchmark)
            author_result = {
                "source": "author-benchmark-v1",
                "benchmark_id": benchmark["id"],
                "benchmark_record_sha256": generate_manifest.sha256_file(
                    benchmark_path
                ),
                "results_sha256": benchmark["results_sha256"],
                "results": benchmark["results"],
            }
            if "ttft_cache" in benchmark:
                author_result["ttft_cache"] = benchmark["ttft_cache"]
        elif _existing_model_has_release(root, runtime):
            raise ShipitError(
                "maintainer bypass requires benchmark.json for an existing model"
            )
        consensus = {
            "schema_version": community_verification.SCHEMA_VERSION,
            "candidate_id": candidate,
            "runtime_version": runtime["version"],
            "pull_request": int(pr["number"]),
            "pull_request_url": str(pr["html_url"]),
            "proposal_head_sha": str(subject["proposal_head_sha"]),
            "author": author,
            "runtime_authors": release["authors"],
            "subject": dict(subject),
            "policy": dict(community_verification.POLICY),
            "qualification": {
                "passed": False,
                "independent_verifiers": 0,
                "required_verifiers": community_verification.POLICY[
                    "required_verifiers"
                ],
                "safety_passed": True,
                "blocking_failures": [],
            },
            "verifiers": [],
            "results": [] if author_result is None else [author_result],
            "score": {
                "policy": (
                    "letsinfer-throughput-geomean-of-verifier-means-v1"
                    if author_result is None
                    else "letsinfer-throughput-geomean-of-author-run-v1"
                ),
                "aggregate_tps": (
                    None
                    if author_result is None
                    else generate_manifest.benchmark_score(benchmark)
                ),
            },
            "verifications": [],
        }
    qualification = consensus["qualification"]
    if (
        qualification["safety_passed"] is not True
        or qualification["blocking_failures"]
    ):
        raise ShipitError("maintainer bypass cannot override a blocking failure")
    consensus["qualification"]["passed"] = True
    consensus["waiver"] = {
        "schema_version": 1,
        "policy": "allowlisted-maintainer-bypass-v1",
        "actor": dict(actor),
        "reason": reason,
        "comment_id": int(comment["id"]),
        "comment_url": str(comment["html_url"]),
        "issued_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    consensus.pop("consensus_id", None)
    consensus["consensus_id"] = hashlib.sha256(canonical_bytes(consensus)).hexdigest()
    return consensus


def _ordinary_consensus(
    *, root: pathlib.Path, candidate: str, subject: Mapping[str, Any]
) -> dict[str, Any]:
    consensus = generate_manifest.read_object(root / candidate / "benchmark.consensus.json")
    if (
        consensus.get("subject") != subject
        or consensus.get("qualification", {}).get("passed") is not True
        or consensus.get("waiver") is not None
    ):
        raise ShipitError("two-independent-verifier consensus is unavailable or stale")
    return consensus


def _publish(
    *, root: pathlib.Path, bundle: Mapping[str, Any], candidate: str
) -> dict[str, Any]:
    username = os.environ.get("OCI_USERNAME", "")
    password = os.environ.get("OCI_PASSWORD", "")
    if not username or not password:
        raise ShipitError("production registry credentials are unavailable")
    engine = bundle["engine"]
    if bundle["mode"] == "build-engine":
        engine_repository = str(engine["reference"]).rsplit("@", 1)[0]
        published_engine = oci_layout.publish(
            root / "engine.oci.tar",
            repository=engine_repository,
            platform=engine["platform"],
            tag=f"verified-{bundle['proposal_head_sha'][:12]}",
            username=username,
            password=password,
        )
        if published_engine["reference"] != engine["reference"] or published_engine["config_digest"] != engine["config_digest"]:
            raise ShipitError("published Engine identity differs from verifier bundle")
    if bundle["mode"] == "build-native-engine":
        engine_receipt = {
            key: engine[key]
            for key in ("kind", "payload_digest", "platform", "source_revision")
        }
    else:
        engine_receipt = oci_layout.verify_reference(
            engine["reference"],
            expected_config=engine["config_digest"],
            expected_platform=engine.get("platform"),
        )
    runtime = bundle["runtime"]
    runtime_repository = str(runtime["source"]).rsplit("@", 1)[0]
    runtime_plan = oci_artifact.plan(
        root / "runtime.letsinfer",
        repository=runtime_repository,
        candidate=candidate,
        version=runtime["version"],
    )
    if runtime_plan.document() != runtime:
        raise ShipitError("runtime publication plan differs from verifier bundle")
    registry = oci_artifact.Registry(runtime_plan, username, password)
    source = registry.publish()
    if source != runtime["source"]:
        raise ShipitError("published runtime identity differs from verifier bundle")
    runtime_receipt = oci_layout.verify_reference(
        source, expected_config=runtime["config_digest"]
    )
    return {"engine": engine_receipt, "runtime": runtime_receipt}


def _check(name: str, head: str, conclusion: str, summary: str) -> None:
    verification_bot.api(
        f"repos/{REPOSITORY}/check-runs",
        method="POST",
        value={
            "name": name,
            "head_sha": head,
            "status": "completed",
            "conclusion": conclusion,
            "output": {"title": "Runtime publication " + conclusion, "summary": summary[:65000]},
        },
    )


def _existing_publication_receipt(
    *,
    number: int,
    candidate: str,
    current: str,
    runtime: Mapping[str, Any],
    release: Mapping[str, Any],
    consensus: Mapping[str, Any],
    bot_login: str,
) -> dict[str, Any] | None:
    comments = verification_bot._flatten_pages(
        verification_bot.api(
            f"repos/{REPOSITORY}/issues/{number}/comments?per_page=100",
            paginate=True,
        )
    )
    provenance = release.get("provenance")
    subject = consensus.get("subject")
    if not isinstance(provenance, Mapping) or not isinstance(subject, Mapping):
        return None
    expected_engine = runtime.get("engine")
    expected_oci = expected_engine.get("oci") if isinstance(expected_engine, Mapping) else None
    expected_engine_reference = (
        expected_oci.get("reference") if isinstance(expected_oci, Mapping) else None
    )
    expected_runtime_digest = subject.get("runtime_oci_manifest_digest")
    expected_execution = provenance.get("execution_sha256")
    for comment in reversed(comments):
        if not isinstance(comment, Mapping):
            continue
        user = comment.get("user")
        body = comment.get("body")
        if (
            not isinstance(user, Mapping)
            or str(user.get("login", "")).casefold() != bot_login.casefold()
            or user.get("type") != "Bot"
            or not isinstance(body, str)
            or RECEIPT_MARKER not in body
        ):
            continue
        match = re.search(r"```json\n(\{.*?\})\n```", body, flags=re.DOTALL)
        if match is None:
            continue
        try:
            receipt = json.loads(match[1])
        except json.JSONDecodeError:
            continue
        if not isinstance(receipt, dict):
            continue
        published = receipt.get("published")
        engine = published.get("engine") if isinstance(published, Mapping) else None
        runtime_artifact = (
            published.get("runtime") if isinstance(published, Mapping) else None
        )
        if (
            receipt.get("schema_version") == 1
            and receipt.get("repository") == REPOSITORY
            and receipt.get("pull_request") == number
            and receipt.get("candidate") == candidate
            and receipt.get("runtime_version") == runtime.get("version")
            and receipt.get("merge_head_sha") == current
            and receipt.get("proposal_head_sha") == provenance.get("proposal_head_sha")
            and receipt.get("execution_sha256") == expected_execution
            and receipt.get("waiver") == consensus.get("waiver")
            and isinstance(engine, Mapping)
            and engine.get("anonymous_pull_verified") is True
            and engine.get("reference") == expected_engine_reference
            and isinstance(runtime_artifact, Mapping)
            and runtime_artifact.get("anonymous_pull_verified") is True
            and isinstance(expected_runtime_digest, str)
            and isinstance(runtime_artifact.get("reference"), str)
            and runtime_artifact["reference"].endswith("@" + expected_runtime_digest)
        ):
            return receipt
    return None


def _merge_exact(number: int, current: str, candidate: str, version: str) -> None:
    merged = verification_bot.api(
        f"repos/{REPOSITORY}/pulls/{number}/merge",
        method="PUT",
        value={
            "sha": current,
            "merge_method": "squash",
            "commit_title": f"Publish {candidate} {version}",
        },
    )
    if not isinstance(merged, dict) or merged.get("merged") is not True:
        raise ShipitError("GitHub refused the exact-head merge after publication")


def _update_behind_receipt_head(
    *,
    number: int,
    current: str,
    candidate: str,
    root: pathlib.Path,
    wait_seconds: int = 300,
) -> str:
    pull = _pull(number)
    if pull.get("mergeable_state") != "behind":
        return current
    verification_bot.api(
        f"repos/{REPOSITORY}/pulls/{number}/update-branch",
        method="PUT",
        value={"expected_head_sha": current},
    )
    deadline = time.monotonic() + wait_seconds
    updated = current
    while time.monotonic() < deadline:
        latest = _pull(number)
        head = latest.get("head")
        value = head.get("sha") if isinstance(head, Mapping) else None
        if isinstance(value, str) and value != current:
            updated = value
            break
        time.sleep(5)
    if updated == current:
        raise ShipitError("GitHub did not update the published proposal head")
    fetch = subprocess.run(
        ["git", "fetch", "--no-tags", "origin", f"pull/{number}/head"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if fetch.returncode != 0:
        raise ShipitError("cannot fetch the updated published proposal head")
    fetched = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD"], cwd=root, text=True
    ).strip()
    if fetched != updated:
        raise ShipitError("fetched proposal head differs from GitHub")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", current, updated],
        cwd=root,
        check=False,
    )
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", current, updated, "--", candidate],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0 or unchanged.returncode != 0:
        raise ShipitError(
            "updating the proposal changed its published runtime candidate"
        )
    return updated


def process(event: Mapping[str, Any], root: pathlib.Path) -> dict[str, Any]:
    issue, comment = event.get("issue"), event.get("comment")
    if not isinstance(issue, Mapping) or not isinstance(comment, Mapping) or not isinstance(issue.get("pull_request"), Mapping):
        return {"processed": False, "reason": "not a pull-request comment"}
    body = comment.get("body")
    if not isinstance(body, str) or not body.strip().startswith("/shipit"):
        return {"processed": False, "reason": "not a shipit command"}
    bypass, reason = parse_command(body)
    actor = _identity(comment.get("user"), "shipit actor")
    if _permission(actor) not in {"admin", "maintain"}:
        raise ShipitError("/shipit requires repository maintain or admin permission")
    if bypass:
        require_configured_bypass_actor(
            int(actor["github_id"]),
            os.environ.get("LETSINFER_VERIFIER_BYPASS_GITHUB_IDS", ""),
        )
    number = int(issue["number"])
    pr = _pull(number)
    candidate = _changed_candidate(number)
    root = root.resolve(strict=True)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if current != pr["head"]["sha"]:
        raise ShipitError("checked-out proposal head differs from GitHub")
    runtime = generate_manifest.read_object(root / candidate / "runtime.json")
    release = generate_manifest.read_object(root / candidate / "release.json")
    provenance = release.get("provenance")
    proposal = (
        provenance.get("proposal_head_sha")
        if isinstance(provenance, dict)
        else current
    )
    if not isinstance(proposal, str) or re.fullmatch(r"[0-9a-f]{40}", proposal) is None:
        raise ShipitError("verified proposal head is unavailable")
    bot_login = os.environ.get("LETSINFER_VERIFICATION_BOT_LOGIN", "")
    if not bot_login:
        raise ShipitError("verification bot login is not configured")
    _approved_review(number, int(pr["user"]["id"]), required=not bypass)
    if bypass and (root / candidate / "benchmark.consensus.json").is_file():
        consensus = generate_manifest.read_object(
            root / candidate / "benchmark.consensus.json"
        )
        generate_manifest.validate_consensus_binding(runtime, consensus)
        receipt = _existing_publication_receipt(
            number=number,
            candidate=candidate,
            current=current,
            runtime=runtime,
            release=release,
            consensus=consensus,
            bot_login=bot_login,
        )
        if receipt is not None:
            current = _update_behind_receipt_head(
                number=number,
                current=current,
                candidate=candidate,
                root=root,
            )
            require_checks(current, bypass=True, wait_seconds=1800)
            finalize_bypassed_community_check(current)
            _check(
                CHECK_NAME,
                current,
                "success",
                "Resumed an exact publication that was already anonymously verified.",
            )
            _merge_exact(number, current, candidate, str(runtime["version"]))
            return {
                "processed": True,
                "merged": True,
                "resumed": True,
                "candidate": candidate,
                "head": current,
            }
    _bot_only_head(
        proposal=proposal,
        current=current,
        candidate=candidate,
        bot_login=bot_login,
    )
    _preflight_publication(
        number=number, candidate=candidate, root=root, bypass=bypass
    )
    with tempfile.TemporaryDirectory(prefix="letsinfer-shipit-") as temporary:
        artifact_root, bundle = _download_artifact(
            name=f"verification-bundle-pr-{number}-{proposal}",
            expected_pr=number,
            expected_head=proposal,
            expected_base=str(pr["base"]["sha"]),
            destination=pathlib.Path(temporary),
        )
        if bundle["candidate"] != candidate or bundle["subject"]["runtime_version"] != runtime["version"]:
            raise ShipitError("verifier bundle candidate or version differs")
        if bypass:
            assert reason is not None
            consensus = _bypass_consensus(
                pr=pr, candidate=candidate, subject=bundle["subject"], root=root,
                actor=actor, reason=reason, comment=comment,
            )
            generate_manifest.validate_consensus_binding(runtime, consensus)
            _url, current = verification_bot.materialize(root, candidate, pr, consensus)
            _check(
                verification_bot.CHECK_NAME,
                current,
                "success",
                "An allowlisted maintainer applied the audited verifier override.",
            )
            require_checks(current, bypass=True, wait_seconds=1800)
        else:
            consensus = _ordinary_consensus(root=root, candidate=candidate, subject=bundle["subject"])
            generate_manifest.validate_consensus_binding(runtime, consensus)
            require_checks(current, bypass=False)
        receipts = _publish(root=artifact_root, bundle=bundle, candidate=candidate)
        latest = _pull(number)
        if latest["head"]["sha"] != current:
            raise ShipitError("pull-request head changed during publication")
        _approved_review(
            number,
            int(latest["user"]["id"]),
            required=not bypass,
        )
        require_checks(current, bypass=bypass)
        receipt = {
            "schema_version": 1,
            "repository": REPOSITORY,
            "pull_request": number,
            "proposal_head_sha": proposal,
            "merge_head_sha": current,
            "candidate": candidate,
            "runtime_version": runtime["version"],
            "execution_sha256": bundle["subject"]["execution_sha256"],
            "actor": actor,
            "command_comment_id": comment["id"],
            "waiver": consensus.get("waiver"),
            "published": receipts,
            "workflow_run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
            "published_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        verification_bot._post_comment(
            number,
            "## Runtime publication receipt\n\n```json\n"
            + json.dumps(receipt, indent=2, sort_keys=True)
            + "\n```\n\n"
            + RECEIPT_MARKER,
        )
        if bypass:
            finalize_bypassed_community_check(current)
        _check(CHECK_NAME, current, "success", f"Published exact Engine/runtime artifacts for `{bundle['subject']['execution_sha256']}`.")
        _merge_exact(number, current, candidate, str(runtime["version"]))
    return {"processed": True, "merged": True, "candidate": candidate, "head": current}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    try:
        event = json.loads(arguments.event.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShipitError(f"cannot read GitHub event: {error}") from error
    if not isinstance(event, dict):
        raise ShipitError("GitHub event must contain an object")
    print(json.dumps(process(event, arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ShipitError,
        community_verification.ConsensusError,
        generate_manifest.ManifestError,
        oci_artifact.OciError,
        oci_layout.LayoutError,
        verifier_bundle.BundleError,
        verification_bot.BotError,
    ) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
