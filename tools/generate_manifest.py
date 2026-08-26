#!/usr/bin/env python3
"""Validate flat runtime candidates and generate the public selection manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from typing import Any


SCHEMA_VERSION = 6
RUNTIME_SCHEMA_VERSION = 5
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OCI_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
LICENSE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,126}")
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")
CANDIDATE_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]*--[a-z0-9][a-z0-9._-]*--"
    r"[a-z0-9][a-z0-9._-]*--[a-z0-9][a-z0-9._-]*"
)
RECOMMENDATION_POLICY = {
    "id": "letsinfer-throughput-geomean-v1",
    "benchmark_suite": "letsinfer-code-prose-v1",
    "metric": "aggregate_tps",
    "cache": "uncached",
    "tie_breakers": ["score", "version", "candidate"],
}
CONTRACT_MIGRATION_METHOD = "runtime-contract-migration-v1"
BENCHMARK_SCHEMA_VERSIONS = {4, 5, 6, 7}
SHARED_BENCHMARK_SCHEMA_VERSIONS = {5, 6, 7}
TTFT_CACHE_BENCHMARK_SCHEMA_VERSION = 6
EXECUTION_PAYLOAD_BENCHMARK_SCHEMA_VERSION = 7
REVOCATION_REASON_CODES = {
    "compromised-verifier-key",
    "fraudulent-evidence",
    "incorrect-target",
    "invalid-benchmark-contract",
    "output-correctness-failure",
    "safety-failure",
    "structurally-invalid-evidence",
}


class ManifestError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def github_identity(value: Any, where: str, *, allow_organization: bool) -> dict[str, Any]:
    account_types = {"User", "Organization"} if allow_organization else {"User"}
    if (
        not isinstance(value, dict)
        or set(value) != {"github_login", "github_id", "github_type"}
        or not isinstance(value.get("github_login"), str)
        or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", value["github_login"])
        is None
        or not isinstance(value.get("github_id"), int)
        or isinstance(value.get("github_id"), bool)
        or value["github_id"] <= 0
        or value.get("github_type") not in account_types
    ):
        raise ManifestError(f"{where} must identify a GitHub account")
    return value


def normalized_candidate(runtime: dict[str, Any]) -> str:
    engine = runtime.get("engine", {}).get("id")
    target = runtime.get("target", {}).get("id")
    uri = runtime.get("model", {}).get("uri")
    match = re.fullmatch(r"hf://([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)", str(uri))
    if not isinstance(engine, str) or not isinstance(target, str) or match is None:
        raise ManifestError("runtime exact engine, target, or model URI is invalid")
    return "--".join((engine, match[1].lower(), match[2].lower(), target))


def validate_runtime_execution_contract(runtime: dict[str, Any]) -> None:
    """Keep repository candidates aligned with core's generic group boundary."""
    target = runtime.get("target")
    placement = target.get("placement") if isinstance(target, dict) else None
    if not isinstance(placement, dict) or set(placement) != {
        "strategy", "node_count", "interconnect"
    }:
        raise ManifestError("runtime target placement must declare strategy, node_count, and interconnect")
    strategy = placement.get("strategy")
    node_count = placement.get("node_count")
    if (
        strategy not in {"single", "parallel"}
        or not isinstance(node_count, int)
        or isinstance(node_count, bool)
        or node_count not in range(1, 65)
        or (strategy == "single" and node_count != 1)
    ):
        raise ManifestError("runtime target placement strategy or node_count is invalid")
    interconnect = placement.get("interconnect")
    if not isinstance(interconnect, dict) or set(interconnect) != {
        "kind", "rdma_required", "minimum_speed_mbps", "minimum_mtu"
    }:
        raise ManifestError("runtime target interconnect contract is invalid")
    if (
        interconnect.get("kind") not in {"any", "connectx", "ethernet", "wifi", "other"}
        or not isinstance(interconnect.get("rdma_required"), bool)
        or any(
            not isinstance(interconnect.get(key), int)
            or isinstance(interconnect.get(key), bool)
            or interconnect[key] < 0
            for key in ("minimum_speed_mbps", "minimum_mtu")
        )
    ):
        raise ManifestError("runtime target interconnect values are invalid")
    contract = runtime.get("orchestration")
    if strategy == "single":
        if contract is not None:
            raise ManifestError("single-node runtime cannot declare orchestration")
        return
    required = {
        "schema_version", "failure_policy", "endpoint_owner", "startup_order", "tasks"
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != required
        or contract.get("schema_version") != 3
        or contract.get("failure_policy") != "whole-group"
    ):
        raise ManifestError("parallel runtime orchestration contract is invalid")
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != node_count:
        raise ManifestError("parallel runtime must declare one task per required node")
    task_ids: list[str] = []
    for index, task in enumerate(tasks):
        where = f"runtime.orchestration.tasks[{index}]"
        common = {"task_id", "launcher", "environment", "port_count", "readiness"}
        if not isinstance(task, dict) or not common.issubset(task):
            raise ManifestError(f"{where} is incomplete")
        launcher = task.get("launcher")
        fields = common if launcher == "manifest" else common | {"command"}
        if launcher not in {"manifest", "runtime-command"} or set(task) != fields:
            raise ManifestError(f"{where} has unsupported fields")
        task_id = f"task-{index}"
        if task.get("task_id") != task_id:
            raise ManifestError(f"{where}.task_id must be {task_id}")
        port_count = task.get("port_count")
        if (
            not isinstance(port_count, int)
            or isinstance(port_count, bool)
            or port_count not in range(1, 33)
            or not isinstance(task.get("environment"), dict)
            or any(str(key).startswith("LETSINFER_") for key in task["environment"])
        ):
            raise ManifestError(f"{where} has invalid bounded resources")
        if launcher == "runtime-command":
            command = task.get("command")
            if (
                not isinstance(command, list)
                or not command
                or not isinstance(command[0], str)
                or not command[0].startswith("/")
                or command[0] in {"/bin/sh", "/bin/bash", "/usr/bin/env"}
                or any(not isinstance(argument, str) or not argument for argument in command)
            ):
                raise ManifestError(f"{where}.command must be a shell-free argv")
        task_ids.append(task_id)
    if contract.get("endpoint_owner") not in task_ids:
        raise ManifestError("parallel runtime endpoint owner is not a task")
    phases = contract.get("startup_order")
    if not isinstance(phases, list) or any(
        not isinstance(phase, list) or not phase for phase in phases
    ):
        raise ManifestError("parallel runtime startup order is invalid")
    flattened = [task for phase in phases for task in phase]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != sorted(task_ids):
        raise ManifestError("parallel runtime startup order must contain every task once")


def validate_model_links(runtime: dict[str, Any], readme: str) -> None:
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("runtime artifacts must be a non-empty array")
    for artifact in artifacts:
        uri = artifact.get("uri") if isinstance(artifact, dict) else None
        match = re.fullmatch(
            r"hf://([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)", str(uri)
        )
        if match is None:
            raise ManifestError("runtime artifact Hugging Face URI is invalid")
        link = f"https://huggingface.co/{match[1]}/{match[2]}"
        if link not in readme:
            raise ManifestError(
                f"runtime README is missing Hugging Face artifact link: {link}"
            )


def runtime_execution_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    """Normalize the engine-visible contract across manifest/protocol revisions."""
    value = copy.deepcopy(runtime)
    for field in ("schema_version", "version", "status"):
        value.pop(field, None)
    engine = value.get("engine")
    if isinstance(engine, dict):
        engine.pop("protocol", None)
    target = value.get("target")
    placement = target.get("placement") if isinstance(target, dict) else None
    if isinstance(placement, dict):
        if "member_count" in placement and "node_count" not in placement:
            placement["node_count"] = placement.pop("member_count")
        placement.pop("engine_strategy", None)
    serving = value.get("serving")
    if isinstance(serving, dict):
        serving.pop("qualified", None)
    benchmark = value.get("benchmark")
    if isinstance(benchmark, dict):
        benchmark.pop("record", None)
    return value


def runtime_execution_contract_sha256(runtime: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(runtime_execution_contract(runtime))).hexdigest()


def benchmark_score(record: dict[str, Any]) -> float:
    results = record.get("results")
    if not isinstance(results, list) or not results:
        raise ManifestError("benchmark results must be a non-empty array")
    values = [
        float(row["aggregate_tps"])
        for row in results
        if isinstance(row, dict)
        and row.get("is_prefix_cached") is False
        and isinstance(row.get("aggregate_tps"), (int, float))
        and not isinstance(row.get("aggregate_tps"), bool)
        and math.isfinite(float(row["aggregate_tps"]))
        and float(row["aggregate_tps"]) > 0
    ]
    if not values:
        raise ManifestError("benchmark has no uncached positive aggregate throughput")
    return round(math.exp(sum(math.log(value) for value in values) / len(values)), 9)


def benchmark_subject(
    runtime: dict[str, Any], *, measured_engine_oci: str | None = None
) -> dict[str, Any]:
    primary_name = runtime.get("model", {}).get("artifact")
    primary = next(
        (
            artifact
            for artifact in runtime.get("artifacts", [])
            if isinstance(artifact, dict) and artifact.get("name") == primary_name
        ),
        None,
    )
    if not isinstance(primary, dict):
        raise ManifestError(f"runtime primary artifact is missing: {runtime.get('id')}")
    target = runtime.get("target")
    if not isinstance(target, dict):
        raise ManifestError(f"runtime target is invalid: {runtime.get('id')}")
    oci = runtime.get("engine", {}).get("oci", {})
    payload_id = oci.get("payload_id") if isinstance(oci, dict) else None
    engine_identity = (
        {
            "engine_payload_sha256": str(payload_id).removeprefix("sha256:"),
            "measured_engine_oci": measured_engine_oci or oci.get("reference"),
        }
        if isinstance(payload_id, str)
        else {"engine_oci": oci.get("reference")}
    )
    return {
        "candidate_id": runtime.get("id"),
        "runtime_version": runtime.get("version"),
        "model_uri": runtime.get("model", {}).get("uri"),
        "model_revision": primary.get("revision"),
        **engine_identity,
        "target": target.get("id"),
        "target_contract_sha256": hashlib.sha256(canonical_bytes(target)).hexdigest(),
    }


def validate_benchmark_binding(
    runtime: dict[str, Any], record: dict[str, Any]
) -> None:
    schema_version = record.get("schema_version")
    if schema_version not in BENCHMARK_SCHEMA_VERSIONS:
        raise ManifestError(f"benchmark schema is unsupported: {runtime['id']}")
    record_subject = record.get("subject")
    measured_engine_oci = (
        record_subject.get("measured_engine_oci")
        if schema_version == EXECUTION_PAYLOAD_BENCHMARK_SCHEMA_VERSION
        and isinstance(record_subject, dict)
        else None
    )
    subject = benchmark_subject(
        runtime, measured_engine_oci=measured_engine_oci
    )
    if record_subject != subject:
        raise ManifestError(f"benchmark subject differs from runtime: {runtime['id']}")
    contract = runtime["benchmark"]["contract"]
    contract_sha = hashlib.sha256(canonical_bytes(contract)).hexdigest()
    if record.get("benchmark_contract_sha256") != contract_sha:
        raise ManifestError(
            f"benchmark contract identity differs from runtime: {runtime['id']}"
        )
    if (
        schema_version in SHARED_BENCHMARK_SCHEMA_VERSIONS
        and record.get("benchmark_contract") != contract
    ):
        raise ManifestError(
            f"embedded benchmark contract differs from runtime: {runtime['id']}"
        )
    results = record.get("results")
    if not isinstance(results, list) or not results:
        raise ManifestError(f"benchmark results are unavailable: {runtime['id']}")
    if schema_version == TTFT_CACHE_BENCHMARK_SCHEMA_VERSION:
        ttft_cache = record.get("ttft_cache")
        if not isinstance(ttft_cache, dict):
            raise ManifestError(f"benchmark TTFT cache result is unavailable: {runtime['id']}")
        bound_results = {"results": results, "ttft_cache": ttft_cache}
    else:
        bound_results = results
    results_sha = hashlib.sha256(canonical_bytes(bound_results)).hexdigest()
    if record.get("results_sha256") != results_sha:
        raise ManifestError(f"benchmark results identity differs: {runtime['id']}")
    timestamp_ns = record.get("timestamp_unix_ns")
    installation_id = record.get("installation_id")
    if (
        not isinstance(timestamp_ns, int)
        or isinstance(timestamp_ns, bool)
        or timestamp_ns <= 0
        or not SHA256_RE.fullmatch(str(installation_id))
    ):
        raise ManifestError(f"benchmark installation identity is invalid: {runtime['id']}")
    identity = hashlib.sha256(
        canonical_bytes(
            {
                "benchmark_contract_sha256": contract_sha,
                "contract": "letsinfer-benchmark-identity-v2",
                "installation_id": installation_id,
                "results_sha256": results_sha,
                "subject": subject,
                "timestamp_unix_ns": timestamp_ns,
            }
        )
    ).hexdigest()
    if record.get("id") != identity:
        raise ManifestError(f"benchmark identity differs: {runtime['id']}")


def validate_benchmark_integrity(record: dict[str, Any], where: str) -> None:
    """Validate a sealed record without rebinding it to a newer runtime version."""
    subject = record.get("subject")
    subject_fields = {
        "candidate_id", "runtime_version", "model_uri", "model_revision",
        "engine_oci", "target", "target_contract_sha256",
    }
    if (
        record.get("schema_version") != 4
        or not isinstance(subject, dict)
        or set(subject) != subject_fields
        or not SHA256_RE.fullmatch(str(subject.get("target_contract_sha256")))
        or not SHA256_RE.fullmatch(str(record.get("benchmark_contract_sha256")))
    ):
        raise ManifestError(f"benchmark migration record is invalid: {where}")
    results = record.get("results")
    if not isinstance(results, list) or not results:
        raise ManifestError(f"benchmark migration results are unavailable: {where}")
    results_sha = hashlib.sha256(canonical_bytes(results)).hexdigest()
    if record.get("results_sha256") != results_sha:
        raise ManifestError(f"benchmark migration results identity differs: {where}")
    timestamp_ns = record.get("timestamp_unix_ns")
    installation_id = record.get("installation_id")
    if (
        not isinstance(timestamp_ns, int)
        or isinstance(timestamp_ns, bool)
        or timestamp_ns <= 0
        or not SHA256_RE.fullmatch(str(installation_id))
    ):
        raise ManifestError(f"benchmark migration installation identity is invalid: {where}")
    identity = hashlib.sha256(
        canonical_bytes(
            {
                "benchmark_contract_sha256": record["benchmark_contract_sha256"],
                "contract": "letsinfer-benchmark-identity-v2",
                "installation_id": installation_id,
                "results_sha256": results_sha,
                "subject": subject,
                "timestamp_unix_ns": timestamp_ns,
            }
        )
    ).hexdigest()
    if record.get("id") != identity:
        raise ManifestError(f"benchmark migration identity differs: {where}")


def contract_migration_entries(migration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = migration.get("contract_migrations")
    if not isinstance(entries, dict):
        raise ManifestError("runtime contract migration ledger is invalid")
    fields = {
        "from_version",
        "benchmark_record",
        "benchmark_record_sha256",
        "execution_contract_sha256",
    }
    for identity, entry in entries.items():
        candidate, separator, version = str(identity).rpartition("@")
        if (
            not separator
            or not CANDIDATE_RE.fullmatch(candidate)
            or not VERSION_RE.fullmatch(version)
            or not isinstance(entry, dict)
            or set(entry) != fields
            or not VERSION_RE.fullmatch(str(entry.get("from_version")))
            or entry["from_version"] == version
            or entry.get("benchmark_record")
            != f"{candidate}/benchmark.previous.json"
            or not SHA256_RE.fullmatch(str(entry.get("benchmark_record_sha256")))
            or not SHA256_RE.fullmatch(str(entry.get("execution_contract_sha256")))
        ):
            raise ManifestError(f"runtime contract migration is invalid: {identity}")
    return entries


def migrated_release(
    *,
    root: pathlib.Path,
    runtime: dict[str, Any],
    source: str | None,
    release_metadata: dict[str, Any],
    old_release: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    candidate = runtime["id"]
    version = runtime["version"]
    from_version = entry["from_version"]
    identity = f"{candidate}@{version}"
    if source is None:
        raise ManifestError(f"published OCI source is missing for {identity}")
    if _version_key(from_version) >= _version_key(version):
        raise ManifestError(f"runtime contract migration does not move forward: {identity}")
    execution_sha = runtime_execution_contract_sha256(runtime)
    if execution_sha != entry["execution_contract_sha256"]:
        raise ManifestError(f"runtime execution contract changed during migration: {identity}")
    benchmark_relative = pathlib.PurePosixPath(entry["benchmark_record"])
    benchmark_path = root.joinpath(*benchmark_relative.parts)
    if benchmark_path.is_symlink() or not benchmark_path.is_file():
        raise ManifestError(f"runtime migration benchmark is missing: {identity}")
    if sha256_file(benchmark_path) != entry["benchmark_record_sha256"]:
        raise ManifestError(f"runtime migration benchmark digest differs: {identity}")
    benchmark = read_object(benchmark_path)
    validate_benchmark_integrity(benchmark, identity)
    subject = benchmark["subject"]
    primary_name = runtime["model"]["artifact"]
    primary = next(
        (
            artifact
            for artifact in runtime["artifacts"]
            if artifact.get("name") == primary_name
        ),
        None,
    )
    if not isinstance(primary, dict):
        raise ManifestError(f"runtime migration primary artifact is missing: {identity}")
    expected_subject = {
        "candidate_id": candidate,
        "runtime_version": from_version,
        "model_uri": runtime["model"]["uri"],
        "model_revision": primary["revision"],
        "engine_oci": runtime["engine"]["oci"]["reference"],
        "target": runtime["target"]["id"],
        # A schema-only target rename changes this historical digest. Its
        # semantic equivalence is attested by execution_contract_sha256.
        "target_contract_sha256": subject.get("target_contract_sha256"),
    }
    if subject != expected_subject:
        raise ManifestError(f"runtime migration benchmark subject differs: {identity}")
    contract_sha = hashlib.sha256(
        canonical_bytes(runtime["benchmark"]["contract"])
    ).hexdigest()
    old_benchmark = old_release.get("benchmark")
    if (
        benchmark.get("benchmark_contract_sha256") != contract_sha
        or not isinstance(old_benchmark, dict)
        or old_benchmark.get("id") != benchmark.get("id")
        or old_benchmark.get("suite") != runtime["benchmark"]["contract"]["suite"]
        or old_benchmark.get("score") != benchmark_score(benchmark)
        or old_release.get("engine") != runtime["engine"]["id"]
        or old_release.get("engine_oci") != runtime["engine"]["oci"]["reference"]
        or old_release.get("model_uri") != runtime["model"]["uri"]
        or not OCI_RE.fullmatch(str(old_release.get("source")))
    ):
        raise ManifestError(f"runtime migration evidence differs from prior release: {identity}")
    old_provenance = old_release.get("provenance")
    old_verification = old_release.get("verification")
    if (
        not isinstance(old_provenance, dict)
        or not isinstance(old_verification, dict)
        or not isinstance(old_verification.get("verifiers"), list)
    ):
        raise ManifestError(f"runtime migration qualification is invalid: {identity}")
    common_provenance = {
        key: old_provenance.get(key)
        for key in (
            "repository",
            "pull_request",
            "pull_request_url",
            "proposal_head_sha",
            "qualified_commit_sha",
        )
    }
    return {
        "authors": release_metadata["authors"],
        "source": source,
        "engine": runtime["engine"]["id"],
        "engine_oci": runtime["engine"]["oci"]["reference"],
        "model_uri": runtime["model"]["uri"],
        "license": release_metadata["license"],
        "benchmark": {
            "id": benchmark["id"],
            "suite": runtime["benchmark"]["contract"]["suite"],
            "score": benchmark_score(benchmark),
        },
        "provenance": {
            "method": CONTRACT_MIGRATION_METHOD,
            **common_provenance,
            "from_version": from_version,
            "from_source": old_release["source"],
            "benchmark_record_sha256": entry["benchmark_record_sha256"],
            "execution_contract_sha256": execution_sha,
        },
        "verification": {
            "method": CONTRACT_MIGRATION_METHOD,
            "from_version": from_version,
            "from_source": old_release["source"],
            "benchmark_record_path": entry["benchmark_record"],
            "benchmark_record_sha256": entry["benchmark_record_sha256"],
            "execution_contract_sha256": execution_sha,
            "verifiers": old_verification["verifiers"],
        },
    }


def source_map(values: list[str]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for value in values:
        identity, separator, source = value.partition("=")
        candidate, version_separator, version = identity.rpartition("@")
        key = (candidate, version)
        if (
            not separator
            or not version_separator
            or not CANDIDATE_RE.fullmatch(candidate)
            or not re.fullmatch(VERSION_RE.pattern, version)
            or not OCI_RE.fullmatch(source)
        ):
            raise ManifestError(
                "--source must be CANDIDATE@VERSION=digest-pinned-OCI"
            )
        if key in result:
            raise ManifestError(f"duplicate runtime publication source: {identity}")
        result[key] = source
    return result


def sources_from_manifest(path: pathlib.Path) -> dict[tuple[str, str], str]:
    document = read_object(path)
    sources: dict[tuple[str, str], str] = {}
    models = document.get("models")
    if not isinstance(models, dict):
        raise ManifestError("existing manifest models are invalid")
    for model in models.values():
        if not isinstance(model, dict) or not isinstance(model.get("targets"), dict):
            raise ManifestError("existing manifest target map is invalid")
        for target in model["targets"].values():
            if not isinstance(target, dict) or not isinstance(target.get("candidates"), dict):
                raise ManifestError("existing manifest candidate map is invalid")
            for candidate, record in target["candidates"].items():
                releases = record.get("releases") if isinstance(record, dict) else None
                if not CANDIDATE_RE.fullmatch(str(candidate)) or not isinstance(releases, dict):
                    raise ManifestError("existing manifest publication source is invalid")
                for version, release in releases.items():
                    key = (candidate, version)
                    if (
                        not isinstance(release, dict)
                        or not OCI_RE.fullmatch(str(release.get("source")))
                        or key in sources
                    ):
                        raise ManifestError("existing manifest publication source is invalid")
                    sources[key] = release["source"]
    return sources


def validate_consensus_binding(
    runtime: dict[str, Any], consensus: dict[str, Any]
) -> None:
    candidate = runtime["id"]
    waiver = consensus.get("waiver")
    waived = waiver is not None
    waiver_policy = waiver.get("policy") if isinstance(waiver, dict) else None
    if waiver_policy == "maintainer-one-independent-pass-v1":
        expected_verifiers = 1
    elif waiver_policy == "allowlisted-maintainer-bypass-v1":
        expected_verifiers = consensus.get("qualification", {}).get(
            "independent_verifiers"
        )
        if (
            not isinstance(expected_verifiers, int)
            or isinstance(expected_verifiers, bool)
            or expected_verifiers < 0
            or expected_verifiers > 2
        ):
            raise ManifestError(f"runtime maintainer bypass is invalid: {candidate}")
    else:
        expected_verifiers = 2
    qualification = consensus.get("qualification")
    if (
        consensus.get("schema_version") != 2
        or consensus.get("candidate_id") != candidate
        or consensus.get("runtime_version") != runtime["version"]
        or not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or qualification.get("independent_verifiers") != expected_verifiers
        or qualification.get("required_verifiers") != 2
        or qualification.get("safety_passed") is not True
        or qualification.get("blocking_failures") != []
        or consensus.get("policy", {}).get("id")
        != "letsinfer-two-independent-passes-v1"
        or not isinstance(consensus.get("verifications"), list)
        or len(consensus["verifications"]) < expected_verifiers
        or not isinstance(consensus.get("verifiers"), list)
        or len(consensus["verifiers"]) != expected_verifiers
    ):
        raise ManifestError(f"runtime consensus is not qualified: {candidate}")
    if waiver_policy == "allowlisted-maintainer-bypass-v1":
        score = consensus.get("score")
        aggregate_tps = score.get("aggregate_tps") if isinstance(score, dict) else None
        results = consensus.get("results")
        numeric_score = (
            isinstance(aggregate_tps, (int, float))
            and not isinstance(aggregate_tps, bool)
            and math.isfinite(float(aggregate_tps))
            and aggregate_tps > 0
        )
        legacy_unscored = (
            expected_verifiers == 0
            and isinstance(score, dict)
            and score.get("policy")
            == "letsinfer-throughput-geomean-of-verifier-means-v1"
            and aggregate_tps is None
            and results == []
        )
        author_scored = False
        if (
            expected_verifiers == 0
            and isinstance(score, dict)
            and score.get("policy")
            == "letsinfer-throughput-geomean-of-author-run-v1"
            and numeric_score
            and isinstance(results, list)
            and len(results) == 1
            and isinstance(results[0], dict)
        ):
            author = results[0]
            required = {
                "source",
                "benchmark_id",
                "benchmark_record_sha256",
                "results_sha256",
                "results",
            }
            fields = set(author)
            bound_results = (
                {"results": author.get("results"), "ttft_cache": author.get("ttft_cache")}
                if "ttft_cache" in author
                else author.get("results")
            )
            author_scored = (
                (fields == required or fields == required | {"ttft_cache"})
                and author.get("source") == "author-benchmark-v1"
                and SHA256_RE.fullmatch(str(author.get("benchmark_id"))) is not None
                and SHA256_RE.fullmatch(str(author.get("benchmark_record_sha256")))
                is not None
                and SHA256_RE.fullmatch(str(author.get("results_sha256"))) is not None
                and isinstance(author.get("results"), list)
                and bool(author["results"])
                and author["results_sha256"]
                == hashlib.sha256(canonical_bytes(bound_results)).hexdigest()
                and aggregate_tps == benchmark_score({"results": author["results"]})
            )
        verifier_scored = (
            expected_verifiers > 0
            and isinstance(score, dict)
            and score.get("policy")
            == "letsinfer-throughput-geomean-of-verifier-means-v1"
            and numeric_score
            and isinstance(results, list)
            and bool(results)
        )
        if (
            not isinstance(score, dict)
            or set(score) != {"policy", "aggregate_tps"}
            or not (legacy_unscored or author_scored or verifier_scored)
        ):
            raise ManifestError(f"runtime maintainer bypass score is invalid: {candidate}")
    if waived:
        configured = os.environ.get("LETSINFER_VERIFIER_BYPASS_GITHUB_IDS", "")
        values = configured.split(",") if configured else []
        if (
            not values
            or any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in values)
            or len(values) != len({int(value) for value in values})
        ):
            raise ManifestError("maintainer verifier-waiver IDs are not configured")
        authorized_ids = {int(value) for value in values}
        if (
            not isinstance(waiver, dict)
            or set(waiver)
            != {
                "schema_version",
                "policy",
                "actor",
                "reason",
                "comment_id",
                "comment_url",
                "issued_at",
            }
            or waiver.get("schema_version") != 1
            or waiver.get("policy")
            not in {
                "maintainer-one-independent-pass-v1",
                "allowlisted-maintainer-bypass-v1",
            }
            or not isinstance(waiver.get("reason"), str)
            or not waiver["reason"].strip()
            or len(waiver["reason"].encode("utf-8")) > 1000
            or not isinstance(waiver.get("comment_id"), int)
            or waiver["comment_id"] <= 0
            or not isinstance(waiver.get("comment_url"), str)
            or waiver["comment_url"]
            != (
                "https://github.com/letsinferlabs/runtimes/pull/"
                f"{consensus.get('pull_request')}#issuecomment-{waiver.get('comment_id')}"
            )
            or not isinstance(waiver.get("issued_at"), str)
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
                r"(?:\.[0-9]{1,6})?Z",
                waiver["issued_at"],
            )
            is None
        ):
            raise ManifestError(f"runtime verifier waiver is invalid: {candidate}")
        github_identity(
            waiver.get("actor"),
            f"{candidate}.waiver.actor",
            allow_organization=False,
        )
        if waiver["actor"]["github_id"] not in authorized_ids:
            raise ManifestError(f"runtime verifier waiver actor is unauthorized: {candidate}")
    subject = consensus.get("subject")
    if not isinstance(subject, dict):
        raise ManifestError(f"runtime consensus subject is invalid: {candidate}")
    engine_oci = runtime["engine"]["oci"]
    payload_id = engine_oci.get("payload_id")
    engine_bound = (
        subject.get("engine_payload_sha256")
        == str(payload_id).removeprefix("sha256:")
        and "engine_oci_manifest_digest" not in subject
        if payload_id is not None
        else subject.get("engine_oci_manifest_digest")
        == engine_oci["reference"].rsplit("@", 1)[-1]
        and "engine_payload_sha256" not in subject
    )
    if (
        subject.get("candidate_id") != candidate
        or subject.get("runtime_version") != runtime["version"]
        or not engine_bound
        or subject.get("benchmark_contract_sha256")
        != hashlib.sha256(canonical_bytes(runtime["benchmark"]["contract"])).hexdigest()
        or subject.get("target_contract_sha256")
        != hashlib.sha256(canonical_bytes(runtime["target"])).hexdigest()
    ):
        raise ManifestError(f"runtime consensus execution binding differs: {candidate}")
    verifier_ids: list[int] = []
    for index, verifier in enumerate(consensus["verifiers"]):
        github_identity(
            verifier, f"{candidate}.verifiers[{index}]", allow_organization=False
        )
        verifier_ids.append(verifier["github_id"])
    if len(verifier_ids) != len(set(verifier_ids)):
        raise ManifestError(f"runtime consensus repeats a verifier: {candidate}")
    consensus_id = consensus.get("consensus_id")
    unsigned = dict(consensus)
    unsigned.pop("consensus_id", None)
    if (
        not SHA256_RE.fullmatch(str(consensus_id))
        or hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != consensus_id
    ):
        raise ManifestError(f"runtime consensus identity differs: {candidate}")


def validate_provenance(
    candidate: str, provenance: Any, consensus: dict[str, Any]
) -> None:
    fields = {
        "repository",
        "pull_request",
        "pull_request_url",
        "proposal_head_sha",
        "execution_sha256",
        "qualified_commit_sha",
        "consensus_sha256",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != fields
        or provenance.get("repository") != "letsinferlabs/runtimes"
        or provenance.get("pull_request") != consensus.get("pull_request")
        or provenance.get("pull_request_url") != consensus.get("pull_request_url")
        or provenance.get("proposal_head_sha")
        != consensus.get("proposal_head_sha")
        or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("proposal_head_sha")))
        or provenance.get("execution_sha256")
        != consensus.get("subject", {}).get("execution_sha256")
        or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("qualified_commit_sha")))
        or provenance.get("consensus_sha256")
        != hashlib.sha256(canonical_bytes(consensus)).hexdigest()
    ):
        raise ManifestError(f"runtime bot provenance is invalid: {candidate}")


def candidates(
    root: pathlib.Path,
    sources: dict[tuple[str, str], str],
    *,
    require_sources: bool = True,
) -> list[dict[str, Any]]:
    nested = sorted(root.glob("*/*/runtime.json"))
    if nested:
        raise ManifestError(f"nested runtime hierarchy is forbidden: {nested[0]}")
    found: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if directory.name.startswith(".") or directory.name in {"tools", "tests"}:
            continue
        runtime_path = directory / "runtime.json"
        if not runtime_path.is_file():
            raise ManifestError(f"top-level runtime directory lacks runtime.json: {directory.name}")
        runtime = read_object(runtime_path)
        readme_path = directory / "README.md"
        if readme_path.is_symlink() or not readme_path.is_file():
            raise ManifestError(f"runtime README is missing: {directory.name}")
        try:
            readme = readme_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ManifestError(f"cannot read {readme_path}: {error}") from error
        validate_model_links(runtime, readme)
        if runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise ManifestError(f"unsupported runtime schema in {directory.name}")
        validate_runtime_execution_contract(runtime)
        candidate = normalized_candidate(runtime)
        if runtime.get("id") != candidate or directory.name != candidate:
            raise ManifestError(f"candidate directory and runtime.id differ: {directory.name}")
        release_path = directory / "release.json"
        if release_path.is_symlink() or not release_path.is_file():
            raise ManifestError(f"runtime release metadata is missing: {candidate}")
        release_metadata = read_object(release_path)
        if (
            set(release_metadata) != {"schema_version", "authors", "license", "provenance"}
            or release_metadata.get("schema_version") != 2
            or not isinstance(release_metadata.get("authors"), list)
            or not release_metadata["authors"]
            or len(release_metadata["authors"]) > 32
            or not LICENSE_RE.fullmatch(str(release_metadata.get("license")))
        ):
            raise ManifestError(f"runtime release metadata is invalid: {candidate}")
        for index, author in enumerate(release_metadata["authors"]):
            github_identity(author, f"{candidate}.authors[{index}]", allow_organization=True)
        author_ids = [author["github_id"] for author in release_metadata["authors"]]
        if len(author_ids) != len(set(author_ids)):
            raise ManifestError(f"runtime authors contain duplicate accounts: {candidate}")
        adapter = directory / "adapter" / "engine-adapter"
        dockerfile = directory / "image" / "Dockerfile"
        engine_source = any(
            (directory / name).exists()
            for name in ("adapter", "engine", "image", "kernels", "patches", "scripts")
        )
        if engine_source:
            if not adapter.is_file() or adapter.is_symlink():
                raise ManifestError(f"Engine protocol adapter is missing: {candidate}")
            if not dockerfile.is_file() or dockerfile.is_symlink():
                raise ManifestError(f"Engine OCI Dockerfile is missing: {candidate}")
        benchmark_path = directory / "benchmark.json"
        benchmark = read_object(benchmark_path) if benchmark_path.is_file() else None
        if benchmark is not None:
            # A proposal may carry a fresh author run, but it must identify the
            # exact executable runtime if present.
            validate_benchmark_binding(runtime, benchmark)
        consensus_path = directory / "benchmark.consensus.json"
        consensus = read_object(consensus_path) if consensus_path.is_file() else None
        provenance = release_metadata["provenance"]
        if (consensus is None) != (provenance is None):
            raise ManifestError(
                f"consensus and bot-owned provenance must appear together: {candidate}"
            )
        qualified = consensus is not None
        if qualified:
            validate_consensus_binding(runtime, consensus)
            score = consensus.get("score")
            if (
                isinstance(score, dict)
                and score.get("policy")
                == "letsinfer-throughput-geomean-of-author-run-v1"
            ):
                author_results = consensus.get("results")
                author = (
                    author_results[0]
                    if isinstance(author_results, list)
                    and len(author_results) == 1
                    and isinstance(author_results[0], dict)
                    else None
                )
                if (
                    benchmark is None
                    or not isinstance(author, dict)
                    or author.get("benchmark_id") != benchmark.get("id")
                    or author.get("benchmark_record_sha256")
                    != sha256_file(benchmark_path)
                    or author.get("results_sha256")
                    != benchmark.get("results_sha256")
                ):
                    raise ManifestError(
                        f"runtime maintainer benchmark differs: {candidate}"
                    )
            validate_provenance(candidate, provenance, consensus)
        source_key = (candidate, runtime["version"])
        if qualified and require_sources and source_key not in sources:
            raise ManifestError(
                f"published OCI source is missing for {candidate}@{runtime['version']}"
            )
        engine_oci = runtime.get("engine", {}).get("oci", {}).get("reference")
        if not OCI_RE.fullmatch(str(engine_oci)):
            raise ManifestError(f"Engine OCI is not digest-pinned: {candidate}")
        contract = runtime.get("benchmark", {}).get("contract")
        if not isinstance(contract, dict) or not isinstance(contract.get("suite"), str):
            raise ManifestError(f"runtime benchmark contract is invalid: {candidate}")
        found.append(
            {
                "runtime": runtime,
                "source": sources.get(source_key),
                "benchmark": benchmark,
                "consensus": consensus,
                "qualified": qualified,
                "release_metadata": release_metadata,
            }
        )
    if not found:
        raise ManifestError("repository contains no runtime candidates")
    return found


def _previous_releases(previous: dict[str, Any] | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if previous is None:
        return {}
    if previous.get("schema_version") not in {5, SCHEMA_VERSION}:
        return {}
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model, model_record in previous.get("models", {}).items():
        for target_id, target_record in model_record.get("targets", {}).items():
            for candidate, candidate_record in target_record.get("candidates", {}).items():
                releases = candidate_record.get("releases")
                if isinstance(releases, dict):
                    result[(model, target_id, candidate)] = json.loads(
                        json.dumps(releases)
                    )
    return result


def _version_key(value: str) -> tuple[Any, ...]:
    match = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
        r"(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?",
        value,
    )
    if match is None:
        raise ManifestError(f"runtime version is not semantic: {value}")
    prerelease = match[4]
    pre_key: tuple[Any, ...]
    if prerelease is None:
        pre_key = (1,)
    else:
        parts: list[tuple[int, Any]] = []
        for item in prerelease.split("."):
            parts.append((0, int(item)) if item.isdecimal() else (1, item))
        pre_key = (0, *parts)
    return int(match[1]), int(match[2]), int(match[3]), pre_key


def revocation_identities(root: pathlib.Path) -> set[tuple[str, str]]:
    value = read_object(root / "revocations.json")
    if (
        set(value) != {"schema_version", "sequence", "generated_at_unix", "revocations"}
        or value.get("schema_version") != 1
        or not isinstance(value.get("sequence"), int)
        or isinstance(value.get("sequence"), bool)
        or value["sequence"] < 0
        or not isinstance(value.get("generated_at_unix"), int)
        or isinstance(value.get("generated_at_unix"), bool)
        or value["generated_at_unix"] < 0
        or not isinstance(value.get("revocations"), list)
    ):
        raise ManifestError("revocation ledger schema is invalid")
    result: set[tuple[str, str]] = set()
    previous: tuple[str, str] | None = None
    for index, entry in enumerate(value["revocations"]):
        where = f"revocations[{index}]"
        identity = (
            entry.get("runtime_oci_digest") if isinstance(entry, dict) else None,
            entry.get("consensus_sha256") if isinstance(entry, dict) else None,
        )
        actor = entry.get("actor") if isinstance(entry, dict) else None
        verification_ids = (
            entry.get("verification_ids") if isinstance(entry, dict) else None
        )
        replacement = entry.get("replacement") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "runtime_oci_digest",
                "consensus_sha256",
                "actor",
                "revoked_at_unix",
                "reason_code",
                "verification_ids",
                "replacement",
            }
            or not isinstance(identity[0], str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", identity[0]) is None
            or not isinstance(identity[1], str)
            or SHA256_RE.fullmatch(identity[1]) is None
            or not isinstance(actor, dict)
            or set(actor) != {"github_login", "github_id", "github_type"}
            or not isinstance(actor.get("github_login"), str)
            or re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", actor["github_login"]
            )
            is None
            or not isinstance(actor.get("github_id"), int)
            or isinstance(actor.get("github_id"), bool)
            or actor["github_id"] <= 0
            or actor.get("github_type") not in {"User", "Organization", "Bot"}
            or not isinstance(entry.get("revoked_at_unix"), int)
            or isinstance(entry.get("revoked_at_unix"), bool)
            or entry["revoked_at_unix"] <= 0
            or entry.get("reason_code") not in REVOCATION_REASON_CODES
            or not isinstance(verification_ids, list)
            or not verification_ids
            or any(
                not isinstance(item, str) or SHA256_RE.fullmatch(item) is None
                for item in verification_ids
            )
            or verification_ids != sorted(set(verification_ids))
            or (
                replacement is not None
                and (
                    not isinstance(replacement, dict)
                    or set(replacement) != {"candidate", "version", "source"}
                    or not isinstance(replacement.get("candidate"), str)
                    or CANDIDATE_RE.fullmatch(replacement["candidate"]) is None
                    or not isinstance(replacement.get("version"), str)
                    or VERSION_RE.fullmatch(replacement["version"]) is None
                    or not isinstance(replacement.get("source"), str)
                    or OCI_RE.fullmatch(replacement["source"]) is None
                )
            )
            or identity in result
            or (previous is not None and identity < previous)
        ):
            raise ManifestError(f"{where} schema or release identity is invalid")
        result.add(identity)
        previous = identity
    return result


def generate(
    root: pathlib.Path,
    sources: dict[tuple[str, str], str],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = candidates(root, sources)
    revoked = revocation_identities(root)
    history = _previous_releases(previous)
    migration = read_object(root / "qualification-migration.json")
    migration_releases = migration.get("releases")
    if (
        set(migration) != {"schema_version", "method", "releases", "contract_migrations"}
        or migration.get("schema_version") != 2
        or migration.get("method") != "maintainer-qualified-pre-community-v1"
        or not isinstance(migration_releases, dict)
    ):
        raise ManifestError("pre-community qualification migration is invalid")
    migration_entries = contract_migration_entries(migration)
    consumed_migrations: set[str] = set()
    retained_historical_migrations: set[str] = set()
    targets: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for item in items:
        runtime = item["runtime"]
        candidate = runtime["id"]
        target = runtime["target"]
        target_id = target["id"]
        releases = history.get((runtime["logical_model"], target_id, candidate), {})
        retained_historical_migrations.update(
            f"{candidate}@{version}" for version in releases
        )
        normalized_releases: dict[str, Any] = {}
        for version, old_release in releases.items():
            if "qualified" not in old_release:
                normalized_releases[version] = old_release
                continue
            migration_key = f"{candidate}@{version}"
            provenance = migration_releases.get(migration_key)
            if (
                old_release.get("qualified") is not True
                or old_release.get("revoked") is not False
                or not isinstance(provenance, dict)
            ):
                raise ManifestError(
                    f"legacy release lacks explicit qualification migration: {migration_key}"
                )
            benchmark = old_release.get("benchmark")
            normalized_releases[version] = {
                "authors": item["release_metadata"]["authors"],
                "source": old_release["source"],
                "engine": old_release["engine"],
                "engine_oci": old_release["engine_oci"],
                "model_uri": old_release["model_uri"],
                "license": old_release["license"],
                "benchmark": (
                    None
                    if benchmark is None
                    else {
                        "id": benchmark["id"],
                        "suite": benchmark["suite"],
                        "score": benchmark["score"],
                    }
                ),
                "provenance": {
                    "method": migration["method"],
                    **provenance,
                },
                "verification": {
                    "method": migration["method"],
                    "verifiers": [],
                },
            }
        releases = normalized_releases
        migration_key = f"{candidate}@{runtime['version']}"
        migration_entry = migration_entries.get(migration_key)
        if migration_entry is not None:
            if item["qualified"]:
                raise ManifestError(
                    f"runtime cannot use consensus and contract migration together: {migration_key}"
                )
            from_version = migration_entry["from_version"]
            old_release = releases.get(from_version)
            existing_release = releases.get(runtime["version"])
            if old_release is None and existing_release is not None:
                existing_verification = existing_release.get("verification")
                existing_provenance = existing_release.get("provenance")
                if not isinstance(existing_verification, dict) or not isinstance(
                    existing_provenance, dict
                ):
                    raise ManifestError(
                        f"runtime contract migration history is invalid: {migration_key}"
                    )
                old_release = {
                    "source": existing_verification.get("from_source"),
                    "engine": existing_release.get("engine"),
                    "engine_oci": existing_release.get("engine_oci"),
                    "model_uri": existing_release.get("model_uri"),
                    "benchmark": existing_release.get("benchmark"),
                    "provenance": {
                        key: existing_provenance.get(key)
                        for key in (
                            "repository",
                            "pull_request",
                            "pull_request_url",
                            "proposal_head_sha",
                            "qualified_commit_sha",
                        )
                    },
                    "verification": {
                        "verifiers": existing_verification.get("verifiers")
                    },
                }
            if old_release is None:
                raise ManifestError(
                    f"runtime contract migration source release is missing: {migration_key}"
                )
            release = migrated_release(
                root=root,
                runtime=runtime,
                source=item["source"],
                release_metadata=item["release_metadata"],
                old_release=old_release,
                entry=migration_entry,
            )
            if existing_release is not None and existing_release != release:
                raise ManifestError(
                    f"immutable migrated release changed for {migration_key}"
                )
            releases[runtime["version"]] = release
            releases.pop(from_version, None)
            consumed_migrations.add(migration_key)
        if item["qualified"]:
            consensus = item["consensus"]
            consensus_path = f"{candidate}/benchmark.consensus.json"
            consensus_score = consensus["score"]["aggregate_tps"]
            author_benchmark = bool(
                consensus.get("waiver", {}).get("policy")
                == "allowlisted-maintainer-bypass-v1"
                and consensus.get("verifiers") == []
                and consensus.get("score", {}).get("policy")
                == "letsinfer-throughput-geomean-of-author-run-v1"
                and consensus_score is not None
            )
            release = {
                "authors": item["release_metadata"]["authors"],
                "source": item["source"],
                "engine": runtime["engine"]["id"],
                "engine_oci": runtime["engine"]["oci"]["reference"],
                "model_uri": runtime["model"]["uri"],
                "license": item["release_metadata"]["license"],
                "benchmark": (
                    None
                    if consensus_score is None
                    else {
                        "id": consensus["consensus_id"],
                        "suite": runtime["benchmark"]["contract"]["suite"],
                        "score": consensus_score,
                    }
                ),
                "provenance": item["release_metadata"]["provenance"],
                "verification": {
                    "method": (
                        "allowlisted-maintainer-bypass-v1"
                        if consensus.get("waiver", {}).get("policy")
                        == "allowlisted-maintainer-bypass-v1"
                        else (
                            "maintainer-waiver-one-independent-v1"
                            if consensus.get("waiver") is not None
                            else "community-two-independent-v1"
                        )
                    ),
                    "consensus_path": consensus_path,
                    "consensus_sha256": sha256_file(root / consensus_path),
                    "verifiers": consensus["verifiers"],
                    **(
                        {"benchmark_source": "author-benchmark-v1"}
                        if author_benchmark
                        else {}
                    ),
                    **(
                        {"waiver": consensus["waiver"]}
                        if consensus.get("waiver") is not None
                        else {}
                    ),
                },
            }
            existing = releases.get(runtime["version"])
            if existing is not None and existing != release:
                comparable = copy.deepcopy(existing)
                existing_verification = comparable.get("verification")
                if (
                    author_benchmark
                    and isinstance(existing_verification, dict)
                    and "benchmark_source" not in existing_verification
                ):
                    existing_verification["benchmark_source"] = (
                        "author-benchmark-v1"
                    )
                if comparable != release:
                    raise ManifestError(
                        f"immutable release changed for {candidate}@{runtime['version']}"
                    )
            releases[runtime["version"]] = release
        releases = {
            version: release
            for version, release in releases.items()
            if (
                release["source"].rsplit("@", 1)[-1],
                release.get("verification", {}).get("consensus_sha256"),
            )
            not in revoked
        }
        if not releases:
            continue
        existing_target = targets.setdefault(target_id, {"match": target})
        if existing_target["match"] != target:
            raise ManifestError(f"target contract differs across candidates: {target_id}")
        model_target = models.setdefault(
            runtime["logical_model"], {"targets": {}}
        )["targets"].setdefault(
            target_id, {"recommended": None, "candidates": {}}
        )
        latest = max(releases, key=_version_key)
        model_target["candidates"][candidate] = {
            "latest": latest,
            "releases": dict(sorted(releases.items(), key=lambda item: _version_key(item[0]))),
        }
    orphaned_migrations = sorted(
        set(migration_entries)
        - consumed_migrations
        - retained_historical_migrations
    )
    if orphaned_migrations:
        raise ManifestError(
            "runtime contract migration does not identify a current candidate: "
            + orphaned_migrations[0]
        )
    for model in models.values():
        for target in model["targets"].values():
            qualified = [
                (
                    release["benchmark"]["score"],
                    _version_key(version),
                    candidate,
                    version,
                )
                for candidate, record in target["candidates"].items()
                for version, release in record["releases"].items()
                if release["benchmark"] is not None
                and release["benchmark"]["suite"]
                == RECOMMENDATION_POLICY["benchmark_suite"]
            ]
            if qualified:
                _score, _version_sort, candidate, version = max(qualified)
                target["recommended"] = {
                    "candidate": candidate,
                    "version": version,
                }
    return {
        "schema_version": SCHEMA_VERSION,
        "recommendation_policy": RECOMMENDATION_POLICY,
        "targets": targets,
        "models": models,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    result.add_argument("--source", action="append", default=[])
    result.add_argument("--previous", type=pathlib.Path)
    result.add_argument("--output", type=pathlib.Path)
    result.add_argument("--check", action="store_true")
    result.add_argument("--validate-only", action="store_true")
    return result


def main() -> int:
    arguments = parser().parse_args()
    root = arguments.root.resolve(strict=True)
    if arguments.check and arguments.validate_only:
        raise ManifestError("--check and --validate-only are mutually exclusive")
    previous_path = (
        arguments.previous.resolve(strict=True)
        if arguments.previous is not None
        else None
    )
    previous = read_object(previous_path) if previous_path is not None else None
    sources = sources_from_manifest(previous_path) if previous_path is not None else {}
    explicit = source_map(arguments.source)
    sources.update(explicit)
    if arguments.validate_only:
        items = candidates(root, sources, require_sources=False)
        print(f"VALID candidates={len(items)}")
        return 0
    document = generate(root, sources, previous)
    data = canonical_bytes(document)
    output = arguments.output or root / "manifest.json"
    if arguments.check:
        try:
            current = output.read_bytes()
        except OSError as error:
            raise ManifestError(f"cannot read generated manifest {output}: {error}") from error
        if current != data:
            raise ManifestError("manifest.json is not the canonical generated output")
    else:
        output.write_bytes(data)
    print(
        f"VALID candidates={sum(len(t['candidates']) for m in document['models'].values() for t in m['targets'].values())} "
        f"manifest_sha256={hashlib.sha256(data).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
