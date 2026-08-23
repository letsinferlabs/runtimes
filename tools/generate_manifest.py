#!/usr/bin/env python3
"""Validate flat runtime candidates and generate the public selection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any


SCHEMA_VERSION = 6
RUNTIME_SCHEMA_VERSION = 4
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


def benchmark_subject(runtime: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "candidate_id": runtime.get("id"),
        "runtime_version": runtime.get("version"),
        "model_uri": runtime.get("model", {}).get("uri"),
        "model_revision": primary.get("revision"),
        "engine_oci": runtime.get("engine", {}).get("oci", {}).get("reference"),
        "target": target.get("id"),
        "target_contract_sha256": hashlib.sha256(canonical_bytes(target)).hexdigest(),
    }


def validate_benchmark_binding(
    runtime: dict[str, Any], record: dict[str, Any]
) -> None:
    if record.get("schema_version") != 4:
        raise ManifestError(f"benchmark schema is unsupported: {runtime['id']}")
    subject = benchmark_subject(runtime)
    if record.get("subject") != subject:
        raise ManifestError(f"benchmark subject differs from runtime: {runtime['id']}")
    contract_sha = hashlib.sha256(
        canonical_bytes(runtime["benchmark"]["contract"])
    ).hexdigest()
    if record.get("benchmark_contract_sha256") != contract_sha:
        raise ManifestError(
            f"benchmark contract identity differs from runtime: {runtime['id']}"
        )
    results = record.get("results")
    if not isinstance(results, list) or not results:
        raise ManifestError(f"benchmark results are unavailable: {runtime['id']}")
    results_sha = hashlib.sha256(canonical_bytes(results)).hexdigest()
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
    if (
        consensus.get("schema_version") != 1
        or consensus.get("candidate_id") != candidate
        or consensus.get("runtime_version") != runtime["version"]
        or consensus.get("qualification", {}).get("passed") is not True
        or not isinstance(consensus.get("verifications"), list)
        or len(consensus["verifications"]) < 3
        or not isinstance(consensus.get("verifiers"), list)
        or not consensus["verifiers"]
    ):
        raise ManifestError(f"runtime consensus is not qualified: {candidate}")
    subject = consensus.get("subject")
    if not isinstance(subject, dict):
        raise ManifestError(f"runtime consensus subject is invalid: {candidate}")
    if (
        subject.get("candidate_id") != candidate
        or subject.get("runtime_version") != runtime["version"]
        or subject.get("engine_oci_manifest_digest")
        != runtime["engine"]["oci"]["reference"].rsplit("@", 1)[-1]
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
    for entry in value["revocations"]:
        identity = (
            entry.get("runtime_oci_digest") if isinstance(entry, dict) else None,
            entry.get("consensus_sha256") if isinstance(entry, dict) else None,
        )
        if (
            not isinstance(entry, dict)
            or not isinstance(identity[0], str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", identity[0]) is None
            or not isinstance(identity[1], str)
            or SHA256_RE.fullmatch(identity[1]) is None
            or identity in result
            or (previous is not None and identity < previous)
        ):
            raise ManifestError("revocation ledger release identity is invalid")
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
        migration.get("schema_version") != 1
        or migration.get("method") != "maintainer-qualified-pre-community-v1"
        or not isinstance(migration_releases, dict)
    ):
        raise ManifestError("pre-community qualification migration is invalid")
    targets: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for item in items:
        runtime = item["runtime"]
        candidate = runtime["id"]
        target = runtime["target"]
        target_id = target["id"]
        existing_target = targets.setdefault(target_id, {"match": target})
        if existing_target["match"] != target:
            raise ManifestError(f"target contract differs across candidates: {target_id}")
        model_target = models.setdefault(
            runtime["logical_model"], {"targets": {}}
        )["targets"].setdefault(
            target_id, {"recommended": None, "candidates": {}}
        )
        releases = history.get((runtime["logical_model"], target_id, candidate), {})
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
        if item["qualified"]:
            consensus = item["consensus"]
            consensus_path = f"{candidate}/benchmark.consensus.json"
            release = {
                "authors": item["release_metadata"]["authors"],
                "source": item["source"],
                "engine": runtime["engine"]["id"],
                "engine_oci": runtime["engine"]["oci"]["reference"],
                "model_uri": runtime["model"]["uri"],
                "license": item["release_metadata"]["license"],
                "benchmark": {
                    "id": consensus["consensus_id"],
                    "suite": runtime["benchmark"]["contract"]["suite"],
                    "score": consensus["score"]["aggregate_tps"],
                },
                "provenance": item["release_metadata"]["provenance"],
                "verification": {
                    "method": "community-consensus-v1",
                    "consensus_path": consensus_path,
                    "consensus_sha256": sha256_file(root / consensus_path),
                    "verifiers": consensus["verifiers"],
                },
            }
            existing = releases.get(runtime["version"])
            if existing is not None and existing != release:
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
        latest = max(releases, key=_version_key)
        model_target["candidates"][candidate] = {
            "latest": latest,
            "releases": dict(sorted(releases.items(), key=lambda item: _version_key(item[0]))),
        }
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
