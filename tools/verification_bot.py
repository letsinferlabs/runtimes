#!/usr/bin/env python3
"""Trusted GitHub App processor for signed runtime-verification comments.

This program runs only from the repository's trusted default branch.  It may
read PR-controlled files as data, but it never imports or executes them.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from tools import community_verification, generate_manifest


REPOSITORY = "letsinferlabs/runtimes"
CHECK_NAME = "runtime/community-verification"
SUBMISSION_MARKER = "<!-- letsinfer-verification:v1\n"
TALLY_MARKER = "<!-- letsinfer-verification-tally:v1 -->"
REJECTION_MARKER = "letsinfer-verification-rejected"
BYPASS_LOGINS_ENV = "LETSINFER_VERIFICATION_BYPASS_LOGINS"


class BotError(RuntimeError):
    pass


def _run(command: Sequence[str], *, input_data: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            list(command), input=input_data, check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", "replace").strip()
        raise BotError(detail or f"command failed: {command[0]}") from error
    return result.stdout


def api(
    endpoint: str,
    *,
    method: str = "GET",
    value: Mapping[str, Any] | None = None,
    paginate: bool = False,
) -> Any:
    command = ["gh", "api"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    if method != "GET":
        command.extend(["--method", method])
    command.append(endpoint)
    if value is not None:
        command.extend(["--input", "-"])
    raw = _run(command, input_data=None if value is None else canonical_bytes(value))
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BotError(f"GitHub API returned invalid JSON for {endpoint}") from error


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def maintainer_bypass(pull: Mapping[str, Any]) -> bool:
    configured = {
        login.strip().casefold()
        for login in os.environ.get(BYPASS_LOGINS_ENV, "").split(",")
        if login.strip()
    }
    author = pull.get("user")
    login = author.get("login") if isinstance(author, Mapping) else None
    return isinstance(login, str) and login.casefold() in configured


def publish_maintainer_bypass(pull: Mapping[str, Any]) -> None:
    head = pull.get("head")
    author = pull.get("user")
    head_sha = head.get("sha") if isinstance(head, Mapping) else None
    login = author.get("login") if isinstance(author, Mapping) else None
    if not isinstance(head_sha, str) or not isinstance(login, str):
        raise BotError("maintainer pull request identity is invalid")
    api(
        f"repos/{REPOSITORY}/check-runs",
        method="POST",
        value={
            "name": CHECK_NAME,
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": "success",
            "output": {
                "title": "Maintainer verification bypass",
                "summary": (
                    f"Repository-maintenance bypass granted to @{login}. "
                    "This does not create benchmark consensus or qualify runtime bytes."
                ),
            },
        },
    )


def _core() -> tuple[Any, Any]:
    core = pathlib.Path(os.environ.get("LETSINFER_CORE_ROOT", ".core")).resolve()
    if not (core / "core" / "benchmark_verification.py").is_file():
        raise BotError("trusted released core contract is unavailable")
    sys.path.insert(0, str(core))
    try:
        from core import benchmark_verification
        from core.runtime_packs import build_archive
    except ImportError as error:
        raise BotError("cannot import trusted released core contract") from error
    return benchmark_verification, build_archive


def _flatten_pages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BotError("GitHub pagination response is invalid")
    pages = value if all(isinstance(item, list) for item in value) else [value]
    flattened = [item for page in pages for item in page]
    if any(not isinstance(item, dict) for item in flattened):
        raise BotError("GitHub pagination response contains invalid records")
    return flattened


def pull_request(number: int) -> dict[str, Any]:
    value = api(f"repos/{REPOSITORY}/pulls/{number}")
    if (
        not isinstance(value, dict)
        or value.get("number") != number
        or value.get("state") != "open"
        or value.get("base", {}).get("ref") != "main"
        or not isinstance(value.get("head", {}).get("sha"), str)
        or not isinstance(value.get("head", {}).get("ref"), str)
        or not isinstance(value.get("head", {}).get("repo", {}).get("full_name"), str)
    ):
        raise BotError("pull request is not an open runtimes proposal against main")
    labels = {
        item.get("name")
        for item in value.get("labels", [])
        if isinstance(item, dict)
    }
    if "benchmark-ready" not in labels:
        raise BotError("pull request has not passed the benchmark-ready gate")
    return value


def _pr_contract(pr: Mapping[str, Any], files: list[str]) -> Any:
    benchmark_verification, _build_archive = _core()
    author = pr.get("user", {})
    author_identity = benchmark_verification.GitHubIdentity(
        str(author.get("login")), int(author.get("id", -1)), str(author.get("type"))
    )
    return benchmark_verification.PullRequest(
        int(pr["number"]),
        str(pr["html_url"]),
        "OPEN",
        "main",
        str(pr["head"]["sha"]),
        author_identity,
        tuple(files),
        tuple(sorted(
            str(item["name"])
            for item in pr.get("labels", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        )),
    )


def execution_source(
    pr: Mapping[str, Any], destination: pathlib.Path
) -> tuple[pathlib.Path, str, dict[str, Any]]:
    benchmark_verification, build_archive = _core()
    files = _flatten_pages(
        api(
            f"repos/{REPOSITORY}/pulls/{pr['number']}/files?per_page=100",
            paginate=True,
        )
    )
    names = [str(item.get("filename")) for item in files]
    contract = _pr_contract(pr, names)
    candidate = benchmark_verification.select_candidate(contract, None)
    checkout = benchmark_verification.fetch_pull_request(
        contract, destination / "checkout", gh="gh"
    )
    candidate_root = checkout / candidate
    runtime_path = candidate_root / "runtime.json"
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BotError("candidate runtime.json is invalid") from error
    if not isinstance(runtime, dict) or runtime.get("id") != candidate:
        raise BotError("candidate runtime identity differs from its directory")
    pack = destination / f"{candidate}.letsinfer"
    try:
        build_archive(candidate_root, pack)
        subject = benchmark_verification.execution_subject(
            runtime,
            pack_sha256=hashlib.sha256(pack.read_bytes()).hexdigest(),
            pack_bytes=pack.stat().st_size,
        )
    except Exception as error:
        raise BotError(f"candidate execution subject is invalid: {error}") from error
    return checkout, candidate, subject


def accepted_submissions(
    number: int, *, subject: Mapping[str, Any]
) -> list[dict[str, Any]]:
    comments = _flatten_pages(
        api(
            f"repos/{REPOSITORY}/issues/{number}/comments?per_page=100",
            paginate=True,
        )
    )
    accepted: list[dict[str, Any]] = []
    seen: dict[str, bytes] = {}
    bodies = [str(item.get("body", "")) for item in comments]
    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str) or SUBMISSION_MARKER not in body:
            continue
        try:
            item = community_verification.accepted_comment_value(comment)
        except Exception as error:
            publish_rejection(number, comment, str(error), bodies)
            continue
        if item["record"].get("subject") != subject:
            publish_rejection(
                number,
                comment,
                "submission targets a stale execution subject",
                bodies,
            )
            continue
        verification_id = item["record"].get("verification_id")
        encoded = canonical_bytes(item["record"])
        previous = seen.get(str(verification_id))
        if previous is not None:
            if previous != encoded:
                publish_rejection(
                    number,
                    comment,
                    "verification ID conflicts with an earlier submission",
                    bodies,
                )
            continue
        seen[str(verification_id)] = encoded
        accepted.append(item)
    return accepted


def publish_rejection(
    number: int,
    comment: Mapping[str, Any],
    reason: str,
    existing_bodies: list[str],
) -> None:
    comment_id = comment.get("id")
    if not isinstance(comment_id, int):
        return
    marker = f"<!-- {REJECTION_MARKER}:{comment_id} -->"
    if any(marker in body for body in existing_bodies):
        return
    original = str(comment.get("html_url", ""))
    bounded = " ".join(reason.split())[:300] or "submission is invalid"
    body = (
        "## Verification rejected\n\n"
        f"This submission was not counted: {bounded}.\n\n"
        f"**Original:** {original}\n\n{marker}\n"
    )
    created = _post_comment(number, body)
    existing_bodies.append(str(created.get("body", body)))


def _post_comment(number: int, body: str) -> dict[str, Any]:
    value = api(
        f"repos/{REPOSITORY}/issues/{number}/comments",
        method="POST",
        value={"body": body},
    )
    if not isinstance(value, dict):
        raise BotError("GitHub did not return the created comment")
    return value


def publish_comments(number: int, consensus: Mapping[str, Any]) -> str:
    comments = _flatten_pages(
        api(
            f"repos/{REPOSITORY}/issues/{number}/comments?per_page=100",
            paginate=True,
        )
    )
    bodies = [str(item.get("body", "")) for item in comments]
    active_ids = {
        item["verification_id"] for item in consensus.get("verifications", [])
    }
    for verification in consensus.get("_accepted_items", []):
        verification_id = verification["record"]["verification_id"]
        marker = f"<!-- letsinfer-accepted:{verification_id} -->"
        if verification_id in active_ids and not any(marker in body for body in bodies):
            _post_comment(
                number, community_verification.canonical_accepted_comment(verification)
            )
    tally = community_verification.tally_comment(consensus)
    existing = next(
        (item for item in comments if TALLY_MARKER in str(item.get("body", ""))),
        None,
    )
    if existing is None:
        return str(_post_comment(number, tally).get("html_url", ""))
    if existing.get("body") != tally:
        updated = api(
            f"repos/{REPOSITORY}/issues/comments/{existing['id']}",
            method="PATCH",
            value={"body": tally},
        )
        return str(updated.get("html_url", ""))
    return str(existing.get("html_url", ""))


def update_check(
    pr: Mapping[str, Any],
    consensus: Mapping[str, Any],
    tally_url: str,
    *,
    head_sha: str | None = None,
) -> None:
    qualification = consensus["qualification"]
    if qualification["passed"]:
        status = "completed"
        conclusion = "success"
        title = "Community verification passed"
    elif not qualification["safety_passed"]:
        status = "completed"
        conclusion = "failure"
        title = "Safety or correctness verification failed"
    elif not qualification["agreement_passed"]:
        status = "completed"
        conclusion = "neutral"
        title = "More independent measurements are needed"
    else:
        status = "in_progress"
        conclusion = None
        title = "Independent verification is incomplete"
    check: dict[str, Any] = {
        "name": CHECK_NAME,
        "head_sha": head_sha or pr["head"]["sha"],
        "status": status,
        "details_url": tally_url,
        "output": {
            "title": title,
            "summary": (
                f"{qualification['independent_verifiers']} / "
                f"{qualification['required_verifiers']} independent verifications; "
                f"consensus `{consensus['consensus_id']}`"
            ),
        },
    }
    if conclusion is not None:
        check["conclusion"] = conclusion
    api(
        f"repos/{REPOSITORY}/check-runs",
        method="POST",
        value=check,
    )


def cancel_check(pr: Mapping[str, Any]) -> None:
    api(
        f"repos/{REPOSITORY}/check-runs",
        method="POST",
        value={
            "name": CHECK_NAME,
            "head_sha": pr["head"]["sha"],
            "status": "completed",
            "conclusion": "cancelled",
            "output": {
                "title": "Runtime proposal closed",
                "summary": "Community verification stopped because the proposal closed.",
            },
        },
    )


def _content(path: str, *, ref: str) -> dict[str, Any] | None:
    try:
        value = api(f"repos/{REPOSITORY}/contents/{path}?ref={ref}")
    except BotError as error:
        if "HTTP 404" in str(error):
            return None
        raise
    return value if isinstance(value, dict) else None


def _content_bytes(value: Mapping[str, Any] | None) -> bytes | None:
    if value is None or value.get("encoding") != "base64":
        return None
    encoded = value.get("content")
    if not isinstance(encoded, str):
        return None
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return None


def _branch_head(branch: str) -> str:
    value = api(f"repos/{REPOSITORY}/git/ref/heads/{branch}")
    sha = value.get("object", {}).get("sha") if isinstance(value, dict) else None
    if not isinstance(sha, str) or len(sha) != 40:
        raise BotError(f"GitHub did not return the head of {branch}")
    return sha


def put_content(
    path: str, data: bytes, *, branch: str, message: str
) -> str:
    existing = _content(path, ref=branch)
    if _content_bytes(existing) == data:
        return _branch_head(branch)
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(data).decode("ascii"),
        "branch": branch,
    }
    if existing is not None and isinstance(existing.get("sha"), str):
        payload["sha"] = existing["sha"]
    result = api(
        f"repos/{REPOSITORY}/contents/{path}", method="PUT", value=payload
    )
    commit = result.get("commit") if isinstance(result, dict) else None
    if not isinstance(commit, dict) or not isinstance(commit.get("sha"), str):
        raise BotError(f"GitHub did not return the commit for {path}")
    return commit["sha"]


def _branch(pr: Mapping[str, Any], execution: str) -> tuple[str, bool]:
    same_repo = pr["head"]["repo"]["full_name"] == REPOSITORY
    if same_repo:
        return str(pr["head"]["ref"]), False
    branch = f"letsinfer-verification/pr-{pr['number']}-{execution[:12]}"
    try:
        api(f"repos/{REPOSITORY}/git/ref/heads/{branch}")
    except BotError as error:
        if "HTTP 404" not in str(error):
            raise
        api(
            f"repos/{REPOSITORY}/git/refs",
            method="POST",
            value={"ref": f"refs/heads/{branch}", "sha": pr["head"]["sha"]},
        )
    return branch, True


def proposal_head(
    pr: Mapping[str, Any], candidate: str, subject: Mapping[str, Any]
) -> str:
    """Keep the executable proposal identity stable across bot-only commits."""

    release = _content(f"{candidate}/release.json", ref=str(pr["head"]["ref"]))
    raw = _content_bytes(release)
    if raw is not None:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        provenance = value.get("provenance") if isinstance(value, dict) else None
        if (
            isinstance(provenance, dict)
            and provenance.get("repository") == REPOSITORY
            and provenance.get("pull_request") == pr["number"]
            and provenance.get("execution_sha256") == subject.get("execution_sha256")
            and isinstance(provenance.get("proposal_head_sha"), str)
            and len(provenance["proposal_head_sha"]) == 40
        ):
            return provenance["proposal_head_sha"]
    return str(pr["head"]["sha"])


def materialize(
    checkout: pathlib.Path,
    candidate: str,
    pr: Mapping[str, Any],
    consensus: dict[str, Any],
) -> tuple[str, str]:
    branch, companion = _branch(pr, consensus["subject"]["execution_sha256"])
    consensus_path = f"{candidate}/benchmark.consensus.json"
    consensus_data = canonical_bytes(consensus)
    existing_consensus = _content(consensus_path, ref=branch)
    existing_release = _content(f"{candidate}/release.json", ref=branch)
    existing_release_data = _content_bytes(existing_release)
    existing_provenance: Mapping[str, Any] | None = None
    if existing_release_data is not None:
        try:
            existing_release_value = json.loads(existing_release_data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            existing_release_value = None
        if isinstance(existing_release_value, dict) and isinstance(
            existing_release_value.get("provenance"), dict
        ):
            existing_provenance = existing_release_value["provenance"]
    if (
        _content_bytes(existing_consensus) == consensus_data
        and existing_provenance is not None
        and existing_provenance.get("consensus_sha256")
        == hashlib.sha256(consensus_data).hexdigest()
        and isinstance(existing_provenance.get("qualified_commit_sha"), str)
    ):
        consensus_commit = str(existing_provenance["qualified_commit_sha"])
    else:
        consensus_commit = put_content(
            consensus_path,
            consensus_data,
            branch=branch,
            message=f"Record community consensus for runtimes PR #{pr['number']}",
        )
    candidate_root = checkout / candidate
    release_path = candidate_root / "release.json"
    release = generate_manifest.read_object(release_path)
    release["provenance"] = {
        "repository": REPOSITORY,
        "pull_request": pr["number"],
        "pull_request_url": pr["html_url"],
        "proposal_head_sha": consensus["proposal_head_sha"],
        "execution_sha256": consensus["subject"]["execution_sha256"],
        "qualified_commit_sha": consensus_commit,
        "consensus_sha256": hashlib.sha256(consensus_data).hexdigest(),
    }
    release_data = canonical_bytes(release)
    put_content(
        f"{candidate}/release.json",
        release_data,
        branch=branch,
        message=f"Bind qualification provenance for runtimes PR #{pr['number']}",
    )
    (candidate_root / "benchmark.consensus.json").write_bytes(consensus_data)
    release_path.write_bytes(release_data)
    previous = generate_manifest.read_object(checkout / "manifest.json")
    sources = generate_manifest.sources_from_manifest(checkout / "manifest.json")
    version = consensus["runtime_version"]
    sources[(candidate, version)] = (
        f"ghcr.io/letsinferlabs/runtimes/{candidate}@"
        f"{consensus['subject']['runtime_oci_manifest_digest']}"
    )
    manifest = generate_manifest.generate(checkout, sources, previous)
    final_commit = put_content(
        "manifest.json",
        canonical_bytes(manifest),
        branch=branch,
        message=f"Regenerate qualified runtime catalog for PR #{pr['number']}",
    )
    if companion:
        title = f"Qualify {candidate} from #{pr['number']}"
        existing = api(
            f"repos/{REPOSITORY}/pulls?state=open&head=letsinferlabs:{branch}&base=main"
        )
        if not isinstance(existing, list) or not existing:
            created = api(
                f"repos/{REPOSITORY}/pulls",
                method="POST",
                value={
                    "title": title,
                    "head": branch,
                    "base": "main",
                    "body": (
                        f"Bot-owned qualification for #{pr['number']}. Executable bytes "
                        f"remain bound to `{consensus['subject']['execution_sha256']}`."
                    ),
                },
            )
            return str(created.get("html_url", "")), final_commit
        return str(existing[0].get("html_url", "")), final_commit
    return str(pr["html_url"]), final_commit


def empty_consensus(
    *, candidate: str, version: str, subject: Mapping[str, Any], proposal: str
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": community_verification.SCHEMA_VERSION,
        "candidate_id": candidate,
        "runtime_version": version,
        "proposal_head_sha": proposal,
        "subject": dict(subject),
        "qualification": {
            "passed": False,
            "independent_verifiers": 0,
            "required_verifiers": community_verification.POLICY["initial_verifiers"],
            "agreement_passed": True,
            "safety_passed": True,
        },
        "verifications": [],
        "score": {
            "policy": "letsinfer-throughput-geomean-of-consensus-means-v1",
            "aggregate_tps": None,
        },
    }
    document["consensus_id"] = hashlib.sha256(canonical_bytes(document)).hexdigest()
    return document


def close_stale_companions(number: int, execution: str) -> None:
    current = f"letsinfer-verification/pr-{number}-{execution[:12]}"
    pulls = _flatten_pages(
        api(
            f"repos/{REPOSITORY}/pulls?state=open&base=main&per_page=100",
            paginate=True,
        )
    )
    prefix = f"letsinfer-verification/pr-{number}-"
    for pull in pulls:
        head = pull.get("head", {}).get("ref") if isinstance(pull, dict) else None
        pull_number = pull.get("number") if isinstance(pull, dict) else None
        if (
            isinstance(head, str)
            and head.startswith(prefix)
            and head != current
            and isinstance(pull_number, int)
        ):
            api(
                f"repos/{REPOSITORY}/pulls/{pull_number}",
                method="PATCH",
                value={"state": "closed"},
            )


def process_pull_request(number: int) -> dict[str, Any]:
    pr = pull_request(number)
    with tempfile.TemporaryDirectory(prefix="letsinfer-verification-bot-") as temporary:
        checkout, candidate, subject = execution_source(pr, pathlib.Path(temporary))
        close_stale_companions(number, str(subject["execution_sha256"]))
        runtime = generate_manifest.read_object(checkout / candidate / "runtime.json")
        release = generate_manifest.read_object(checkout / candidate / "release.json")
        runtime_authors = release.get("authors")
        if not isinstance(runtime_authors, list) or not runtime_authors:
            raise BotError("candidate runtime authors are invalid")
        for index, value in enumerate(runtime_authors):
            generate_manifest.github_identity(
                value, f"{candidate}.authors[{index}]", allow_organization=True
            )
        proposal = proposal_head(pr, candidate, subject)
        submissions = accepted_submissions(number, subject=subject)
        if submissions:
            author = {
                "github_login": pr["user"]["login"],
                "github_id": pr["user"]["id"],
                "github_type": pr["user"]["type"],
            }
            consensus = community_verification.build_consensus(
                candidate_id=candidate,
                runtime_version=runtime["version"],
                pull_request=number,
                pull_request_url=pr["html_url"],
                proposal_head_sha=proposal,
                author=author,
                runtime_authors=runtime_authors,
                accepted_comments=submissions,
            )
        else:
            consensus = empty_consensus(
                candidate=candidate,
                version=runtime["version"],
                subject=subject,
                proposal=proposal,
            )
        comment_projection = dict(consensus)
        comment_projection["_accepted_items"] = submissions
        tally = publish_comments(number, comment_projection)
        update_check(pr, consensus, tally)
        qualification_url = None
        if consensus["qualification"]["passed"]:
            qualification_url, qualified_head = materialize(
                checkout, candidate, pr, consensus
            )
            update_check(pr, consensus, tally, head_sha=qualified_head)
    return {
        "processed": True,
        "candidate": candidate,
        "consensus_id": consensus["consensus_id"],
        "qualified": consensus["qualification"]["passed"],
        "qualification_url": qualification_url,
    }


def process(event: Mapping[str, Any]) -> dict[str, Any]:
    pull = event.get("pull_request")
    if isinstance(pull, Mapping):
        if event.get("action") == "closed":
            cancel_check(pull)
            return {"processed": True, "closed": True}
        if maintainer_bypass(pull):
            publish_maintainer_bypass(pull)
            return {"processed": True, "maintainer_bypass": True}
        labels = {
            item.get("name")
            for item in pull.get("labels", [])
            if isinstance(item, Mapping)
        }
        if "benchmark-ready" not in labels:
            return {"processed": False, "reason": "benchmark-ready gate is pending"}
        return process_pull_request(int(pull["number"]))
    issue = event.get("issue")
    comment = event.get("comment")
    if (
        not isinstance(issue, Mapping)
        or not isinstance(comment, Mapping)
        or not isinstance(issue.get("pull_request"), Mapping)
        or SUBMISSION_MARKER not in str(comment.get("body", ""))
    ):
        return {"processed": False, "reason": "not a verification submission"}
    number = int(issue["number"])
    return process_pull_request(number)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    try:
        event = json.loads(arguments.event.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BotError(f"cannot read GitHub event: {error}") from error
    if not isinstance(event, dict):
        raise BotError("GitHub event must be a JSON object")
    print(json.dumps(process(event), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BotError, community_verification.ConsensusError, generate_manifest.ManifestError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
