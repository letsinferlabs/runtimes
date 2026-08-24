#!/usr/bin/env python3
"""Validate signed community evidence and build canonical benchmark consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 2
POLICY = {
    "id": "letsinfer-two-independent-passes-v1",
    "required_verifiers": 2,
    "author_votes_count": False,
    "duplicate_github_accounts_count": False,
    "duplicate_devices_count": False,
    "blocking_failures_are_terminal": True,
}
SHA256 = "0123456789abcdef"


class ConsensusError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConsensusError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConsensusError(f"{path} must contain an object")
    return value


def identity(
    value: Any, where: str, *, allow_organization: bool = False
) -> dict[str, Any]:
    actor_types = {"User", "Organization"} if allow_organization else {"User"}
    if (
        not isinstance(value, Mapping)
        or set(value) != {"github_login", "github_id", "github_type"}
        or not isinstance(value.get("github_login"), str)
        or not value["github_login"]
        or not isinstance(value.get("github_id"), int)
        or isinstance(value.get("github_id"), bool)
        or value["github_id"] <= 0
        or value.get("github_type") not in actor_types
    ):
        raise ConsensusError(f"{where} must identify one GitHub user")
    return dict(value)


def _metric_stats(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ConsensusError("consensus metrics must be positive finite values")
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _result_key(value: Mapping[str, Any]) -> tuple[str, str, bool]:
    workload = value.get("workload")
    domain = value.get("prompt_domain")
    cached = value.get("is_prefix_cached")
    if (
        not isinstance(workload, str)
        or domain not in {"code", "prose"}
        or not isinstance(cached, bool)
    ):
        raise ConsensusError("benchmark result key is invalid")
    return workload, str(domain), cached


def _benchmark_rows(record: Mapping[str, Any]) -> dict[tuple[str, str, bool], Mapping[str, Any]]:
    rows = record.get("results")
    if not isinstance(rows, list) or not rows:
        raise ConsensusError("benchmark results are unavailable")
    output: dict[tuple[str, str, bool], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ConsensusError("benchmark result must be an object")
        key = _result_key(row)
        if key in output:
            raise ConsensusError("benchmark contains duplicate workload results")
        output[key] = row
    return output


def _active_votes(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one latest valid vote per GitHub account and per device."""

    ordered = sorted(
        records,
        key=lambda item: (
            int(item["record"].get("submitted_at_unix", 0)),
            str(item["record"].get("verification_id", "")),
        ),
        reverse=True,
    )
    users: set[int] = set()
    devices: set[str] = set()
    active: list[dict[str, Any]] = []
    for item in ordered:
        record = item["record"]
        verifier = identity(record.get("verifier"), "verification.verifier")
        device = record.get("device_id")
        if (
            not isinstance(device, str)
            or len(device) != 64
            or any(character not in SHA256 for character in device)
        ):
            raise ConsensusError("verification device identity is invalid")
        if verifier["github_id"] in users or device in devices:
            continue
        users.add(verifier["github_id"])
        devices.add(device)
        active.append(item)
    return sorted(active, key=lambda item: item["record"]["verification_id"])


def _validate_record(
    record: Mapping[str, Any], *, excluded_ids: set[int], subject: Mapping[str, Any]
) -> None:
    verifier = identity(record.get("verifier"), "verification.verifier")
    if record.get("subject") != subject:
        raise ConsensusError("verification execution subjects differ")
    if record.get("run_order") != ["baseline", "candidate"]:
        raise ConsensusError("verification run order is invalid")
    expected_counting = verifier["github_id"] not in excluded_ids
    if record.get("counts_toward_consensus") is not expected_counting:
        raise ConsensusError("verification author-vote classification differs")
    for name in ("correctness", "safety", "restoration"):
        value = record.get(name)
        if not isinstance(value, Mapping) or not isinstance(value.get("passed"), bool):
            raise ConsensusError(f"verification {name} result is invalid")
    failed = any(
        record[name]["passed"] is False
        for name in ("correctness", "safety", "restoration")
    )
    if failed:
        if not isinstance(record.get("failure"), Mapping) or record.get("run_score") is not None:
            raise ConsensusError("blocking verification failure evidence is invalid")
        if record.get("candidate") is not None and not isinstance(
            record.get("candidate"), Mapping
        ):
            raise ConsensusError("partial candidate benchmark is invalid")
    else:
        if record.get("failure") is not None:
            raise ConsensusError("successful verification carries failure evidence")
        if not isinstance(record.get("candidate"), Mapping):
            raise ConsensusError("verification candidate benchmark is unavailable")
        if not isinstance(record.get("baseline"), Mapping):
            raise ConsensusError("verification paired baseline benchmark is unavailable")
        if not isinstance(record.get("run_score"), Mapping):
            raise ConsensusError("verification score is unavailable")


def build_consensus(
    *,
    candidate_id: str,
    runtime_version: str,
    pull_request: int,
    pull_request_url: str,
    proposal_head_sha: str,
    author: Mapping[str, Any],
    runtime_authors: Sequence[Mapping[str, Any]],
    accepted_comments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate already signature-validated canonical GitHub comments."""

    author_document = identity(author, "pull_request.author")
    runtime_author_documents = [
        identity(value, f"runtime.authors[{index}]", allow_organization=True)
        for index, value in enumerate(runtime_authors)
    ]
    author_ids = {value["github_id"] for value in runtime_author_documents}
    if len(author_ids) != len(runtime_author_documents):
        raise ConsensusError("runtime authors contain duplicate accounts")
    excluded_ids = {author_document["github_id"], *author_ids}
    if not isinstance(proposal_head_sha, str) or len(proposal_head_sha) != 40 or any(
        character not in SHA256 for character in proposal_head_sha
    ):
        raise ConsensusError("proposal head commit identity is invalid")
    if not accepted_comments:
        raise ConsensusError("no accepted verification comments are available")
    first = accepted_comments[0].get("record")
    if not isinstance(first, Mapping) or not isinstance(first.get("subject"), Mapping):
        raise ConsensusError("verification subject is unavailable")
    subject = dict(first["subject"])
    if (
        subject.get("candidate_id") != candidate_id
        or subject.get("runtime_version") != runtime_version
    ):
        raise ConsensusError("verification subject does not identify this runtime")
    for item in accepted_comments:
        record = item.get("record")
        if not isinstance(record, Mapping):
            raise ConsensusError("accepted comment has no verification record")
        if (
            record.get("pull_request") != pull_request
            or record.get("pull_request_url") != pull_request_url
        ):
            raise ConsensusError("verification pull-request identity differs")
        _validate_record(record, excluded_ids=excluded_ids, subject=subject)

    active = _active_votes(list(accepted_comments))
    counted = [
        item
        for item in active
        if item["record"].get("counts_toward_consensus") is True
    ]
    successful = [
        item
        for item in counted
        if all(
            item["record"][name]["passed"]
            for name in ("correctness", "safety", "restoration")
        )
    ]
    key_set: set[tuple[str, str, bool]] | None = None
    rows_by_vote: list[dict[tuple[str, str, bool], Mapping[str, Any]]] = []
    for item in successful:
        rows = _benchmark_rows(item["record"]["candidate"])
        if key_set is None:
            key_set = set(rows)
        elif set(rows) != key_set:
            raise ConsensusError("verification workload sets differ")
        rows_by_vote.append(rows)
    key_set = set() if key_set is None else key_set

    metrics = ("aggregate_tps", "decode_tps", "ttft_seconds")
    aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(key_set):
        summary: dict[str, Any] = {
            "workload": key[0],
            "prompt_domain": key[1],
            "is_prefix_cached": key[2],
            "metrics": {},
        }
        for metric in metrics:
            values: list[float] = []
            for rows in rows_by_vote:
                value = rows[key].get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise ConsensusError(f"verification {metric} is invalid")
                values.append(float(value))
            stats = _metric_stats(values)
            summary["metrics"][metric] = stats
        aggregate_rows.append(summary)

    blocking = [
        item
        for item in accepted_comments
        if any(
            item["record"][name]["passed"] is False
            for name in ("correctness", "safety", "restoration")
        )
    ]
    safety_passed = not blocking
    required = POLICY["required_verifiers"]
    passed = len(successful) >= required and safety_passed

    official_values = [
        row["metrics"]["aggregate_tps"]["mean"]
        for row in aggregate_rows
        if row["is_prefix_cached"] is False
    ]
    official_score = (
        None
        if not official_values
        else math.exp(
            sum(math.log(value) for value in official_values) / len(official_values)
        )
    )
    active_ids = {
        str(item["record"]["verification_id"])
        for item in active
    }
    displayed = [
        *active,
        *[
            item
            for item in blocking
            if str(item["record"]["verification_id"]) not in active_ids
        ],
    ]
    displayed.sort(key=lambda item: str(item["record"]["verification_id"]))
    verifications: list[dict[str, Any]] = []
    for item in displayed:
        record = item["record"]
        verifications.append(
            {
                "user_id": identity(record["verifier"], "verification.verifier"),
                "device_id": record["device_id"],
                "verification_id": record["verification_id"],
                "counts_toward_consensus": record["counts_toward_consensus"],
                "result": record["candidate"],
                "baseline": record["baseline"],
                "score": record["run_score"],
                "evidence": {
                    "comment_id": item.get("comment_id"),
                    "comment_url": item.get("comment_url"),
                    "observed_head_sha": record["observed_head_sha"],
                    "submitted_at_unix": record["submitted_at_unix"],
                    "run_order": record["run_order"],
                    "device_public_key_pem": item.get("device_public_key_pem"),
                    "failure": record.get("failure"),
                },
            }
        )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "runtime_version": runtime_version,
        "pull_request": pull_request,
        "pull_request_url": pull_request_url,
        "proposal_head_sha": proposal_head_sha,
        "author": author_document,
        "runtime_authors": runtime_author_documents,
        "subject": subject,
        "policy": POLICY,
        "qualification": {
            "passed": passed,
            "independent_verifiers": len(successful),
            "required_verifiers": required,
            "safety_passed": safety_passed,
            "blocking_failures": [
                str(item["record"]["verification_id"]) for item in blocking
            ],
        },
        "verifiers": [
            identity(item["record"]["verifier"], "verification.verifier")
            for item in successful
        ],
        "results": aggregate_rows,
        "score": {
            "policy": "letsinfer-throughput-geomean-of-verifier-means-v1",
            "aggregate_tps": official_score,
        },
        "verifications": verifications,
    }
    without_id = canonical_bytes(document)
    document["consensus_id"] = sha256_bytes(without_id)
    return document


def _core_parser() -> Callable[[str], tuple[dict[str, Any], dict[str, Any]]]:
    root = pathlib.Path(os.environ.get("LETSINFER_CORE_ROOT", ".core")).resolve()
    if not (root / "core" / "benchmark_verification.py").is_file():
        raise ConsensusError("released core verification contract is unavailable")
    sys.path.insert(0, str(root))
    try:
        from core.benchmark_verification import parse_comment
    except ImportError as error:
        raise ConsensusError("cannot load released core verification contract") from error
    return parse_comment


def accepted_comment(path: pathlib.Path) -> dict[str, Any]:
    value = read_object(path)
    return accepted_comment_value(value, where=str(path))


def accepted_comment_value(
    value: Mapping[str, Any], *, where: str = "GitHub comment"
) -> dict[str, Any]:
    body = value.get("body")
    user = value.get("user")
    if not isinstance(body, str) or not isinstance(user, Mapping):
        raise ConsensusError(f"{where} is not a GitHub issue-comment record")
    envelope, record = _core_parser()(body)
    verifier = identity(record.get("verifier"), "verification.verifier")
    if (
        user.get("id") != verifier["github_id"]
        or user.get("login", "").lower() != verifier["github_login"].lower()
        or user.get("type") != verifier["github_type"]
    ):
        raise ConsensusError("GitHub comment author differs from signed verifier")
    return {
        "record": record,
        "device_public_key_pem": envelope["device_public_key_pem"],
        "comment_id": value.get("id"),
        "comment_url": value.get("html_url"),
        "original_body": body,
    }


def canonical_accepted_comment(item: Mapping[str, Any]) -> str:
    record = item.get("record")
    if not isinstance(record, Mapping):
        raise ConsensusError("accepted verification record is unavailable")
    verifier = identity(record.get("verifier"), "verification.verifier")
    run_score = record.get("run_score")
    score = run_score.get("overall", {}) if isinstance(run_score, Mapping) else {}
    throughput = score.get("aggregate_tps_geomean")
    change = score.get("change_percent")
    failure = record.get("failure")
    original = item.get("comment_url")
    body = item.get("original_body")
    if (
        not isinstance(original, str)
        or not isinstance(body, str)
        or (
            failure is None
            and (
                not isinstance(throughput, (int, float))
                or isinstance(throughput, bool)
            )
        )
        or (failure is not None and not isinstance(failure, Mapping))
    ):
        raise ConsensusError("accepted verification summary is incomplete")
    if failure is None:
        result = f"{float(throughput):.3f} aggregate tok/s" + (
            "" if change is None else f" · {float(change):+.2f}% vs baseline"
        )
        disposition = (
            "was validated and counted."
            if record.get("counts_toward_consensus") is True
            else "was validated as informational evidence."
        )
    else:
        result = f"blocking `{failure.get('category')}` failure"
        disposition = "was validated as blocking evidence."
    return (
        "## Verification accepted\n\n"
        f"Benchmark by [@{verifier['github_login']}](https://github.com/{verifier['github_login']}) "
        f"{disposition}\n\n"
        f"**Verification:** `{record['verification_id']}`  \n"
        f"**Subject:** `{record['subject']['execution_sha256']}`  \n"
        f"**Result:** {result}"
        + f"  \n**Original:** {original}\n\n"
        + f"<!-- letsinfer-accepted:{record['verification_id']} -->\n"
        + body[body.index("<!-- letsinfer-verification:v1\n") :]
    )


def tally_comment(consensus: Mapping[str, Any]) -> str:
    qualification = consensus.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ConsensusError("consensus qualification is unavailable")
    if qualification.get("safety_passed") is not True:
        state = "FAILED"
    else:
        state = "QUALIFIED" if qualification.get("passed") is True else "COLLECTING"
    lines = [
        "## Let’s Infer community verification",
        "",
        "<!-- letsinfer-verification-tally:v1 -->",
        f"**Status:** {state} · {qualification.get('independent_verifiers')} / "
        f"{qualification.get('required_verifiers')} independent verifications  ",
        f"**Runtime:** `{consensus.get('candidate_id')}@{consensus.get('runtime_version')}`  ",
        f"**Subject:** `{consensus.get('subject', {}).get('execution_sha256')}`",
        "",
        "| Verifier | Role | Aggregate tok/s | Change vs baseline | Verification |",
        "|---|---|---:|---:|---|",
    ]
    for verification in consensus.get("verifications", []):
        user = verification["user_id"]
        role = (
            "independent"
            if verification.get("counts_toward_consensus") is True
            else "informational"
        )
        if verification.get("score") is None:
            failure = verification.get("evidence", {}).get(
                "failure", "blocking failure"
            )
            lines.append(
                f"| [@{user['github_login']}](https://github.com/{user['github_login']}) "
                f"| {role} | — | **BLOCKED:** {failure} "
                f"| `{verification['verification_id'][:12]}` |"
            )
            continue
        overall = verification["score"]["overall"]
        change = overall.get("change_percent")
        lines.append(
            f"| [@{user['github_login']}](https://github.com/{user['github_login']}) "
            f"| {role} | {float(overall['aggregate_tps_geomean']):.3f} "
            f"| {'—' if change is None else f'{float(change):+.2f}%'} "
            f"| `{verification['verification_id'][:12]}` |"
        )
    lines.extend(
        [
            "",
            "**Consensus score:** "
            + (
                "—  "
                if consensus["score"]["aggregate_tps"] is None
                else f"{float(consensus['score']['aggregate_tps']):.3f}  "
            ),
            f"**Safety and correctness:** {'PASS' if qualification.get('safety_passed') else 'BLOCKED'}  ",
            f"**Consensus ID:** `{consensus['consensus_id']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--pull-request", required=True, type=int)
    parser.add_argument("--pull-request-url", required=True)
    parser.add_argument("--proposal-head-sha", required=True)
    parser.add_argument("--author", required=True, type=pathlib.Path)
    parser.add_argument("--runtime-authors", required=True, type=pathlib.Path)
    parser.add_argument("--comment", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--output", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    document = build_consensus(
        candidate_id=arguments.candidate,
        runtime_version=arguments.version,
        pull_request=arguments.pull_request,
        pull_request_url=arguments.pull_request_url,
        proposal_head_sha=arguments.proposal_head_sha,
        author=read_object(arguments.author),
        runtime_authors=read_object(arguments.runtime_authors)["authors"],
        accepted_comments=[accepted_comment(path) for path in arguments.comment],
    )
    arguments.output.write_bytes(canonical_bytes(document))
    print(
        "CONSENSUS "
        f"passed={str(document['qualification']['passed']).lower()} "
        f"verifiers={document['qualification']['independent_verifiers']} "
        f"id={document['consensus_id']}"
    )
    return 0 if document["qualification"]["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConsensusError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
