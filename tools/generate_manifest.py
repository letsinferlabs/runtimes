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


SCHEMA_VERSION = 5
RUNTIME_SCHEMA_VERSION = 3
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OCI_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
AUTHOR_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,38})")
LICENSE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,126}")
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


def source_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        candidate, separator, source = value.partition("=")
        if not separator or not CANDIDATE_RE.fullmatch(candidate) or not OCI_RE.fullmatch(source):
            raise ManifestError("--source must be CANDIDATE=digest-pinned-OCI")
        if candidate in result:
            raise ManifestError(f"duplicate runtime publication source: {candidate}")
        result[candidate] = source
    return result


def sources_from_manifest(path: pathlib.Path) -> dict[str, str]:
    document = read_object(path)
    sources: dict[str, str] = {}
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
                if isinstance(record, dict) and isinstance(record.get("releases"), dict):
                    latest = record.get("latest")
                    record = record["releases"].get(latest)
                if (
                    not CANDIDATE_RE.fullmatch(str(candidate))
                    or not isinstance(record, dict)
                    or not OCI_RE.fullmatch(str(record.get("source")))
                    or candidate in sources
                ):
                    raise ManifestError("existing manifest publication source is invalid")
                sources[candidate] = record["source"]
    return sources


def evidence_from_manifest(path: pathlib.Path) -> dict[str, str]:
    document = read_object(path)
    evidence: dict[str, str] = {}
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
                if not isinstance(record, dict) or not isinstance(record.get("releases"), dict):
                    continue
                latest = record.get("latest")
                release = record["releases"].get(latest)
                benchmark = release.get("benchmark") if isinstance(release, dict) else None
                source = benchmark.get("evidence") if isinstance(benchmark, dict) else None
                if isinstance(source, str) and OCI_RE.fullmatch(source):
                    evidence[candidate] = source
    return evidence


def candidates(
    root: pathlib.Path,
    sources: dict[str, str],
    evidence: dict[str, str] | None = None,
    *,
    require_sources: bool = True,
) -> list[dict[str, Any]]:
    nested = sorted(root.glob("*/*/runtime.json"))
    if nested:
        raise ManifestError(f"nested runtime hierarchy is forbidden: {nested[0]}")
    found: list[dict[str, Any]] = []
    evidence = evidence or {}
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
            set(release_metadata) != {"schema_version", "authors", "license"}
            or release_metadata.get("schema_version") != 1
            or not isinstance(release_metadata.get("authors"), list)
            or not release_metadata["authors"]
            or len(release_metadata["authors"]) > 32
            or any(
                not isinstance(author, str) or not AUTHOR_RE.fullmatch(author)
                for author in release_metadata["authors"]
            )
            or len(release_metadata["authors"])
            != len(set(release_metadata["authors"]))
            or not LICENSE_RE.fullmatch(str(release_metadata.get("license")))
        ):
            raise ManifestError(f"runtime release metadata is invalid: {candidate}")
        adapter = directory / "adapter" / "engine-adapter"
        dockerfile = directory / "image" / "Dockerfile"
        if not adapter.is_file() or adapter.is_symlink():
            raise ManifestError(f"Engine protocol adapter is missing: {candidate}")
        if not dockerfile.is_file() or dockerfile.is_symlink():
            raise ManifestError(f"Engine OCI Dockerfile is missing: {candidate}")
        if require_sources and candidate not in sources:
            raise ManifestError(f"published OCI source is missing for {candidate}")
        benchmark_ref = runtime.get("benchmark", {}).get("record")
        benchmark = None
        score = None
        if benchmark_ref is None:
            if runtime.get("status") == "qualified":
                raise ManifestError(
                    f"qualified runtime benchmark reference is missing: {candidate}"
                )
        elif isinstance(benchmark_ref, dict):
            benchmark_path = directory / str(benchmark_ref.get("path"))
            if not benchmark_path.is_file() or benchmark_path.parent != directory:
                raise ManifestError(f"runtime benchmark record is missing: {candidate}")
            if sha256_file(benchmark_path) != benchmark_ref.get("sha256"):
                raise ManifestError(f"runtime benchmark record hash differs: {candidate}")
            benchmark = read_object(benchmark_path)
            if benchmark.get("id") != benchmark_ref.get("id") or not SHA256_RE.fullmatch(str(benchmark.get("id"))):
                raise ManifestError(f"runtime benchmark identity differs: {candidate}")
            validate_benchmark_binding(runtime, benchmark)
            score = benchmark_score(benchmark)
        else:
            raise ManifestError(f"runtime benchmark reference is invalid: {candidate}")
        engine_oci = runtime.get("engine", {}).get("oci", {}).get("reference")
        if not OCI_RE.fullmatch(str(engine_oci)):
            raise ManifestError(f"Engine OCI is not digest-pinned: {candidate}")
        contract = runtime.get("benchmark", {}).get("contract")
        if not isinstance(contract, dict) or not isinstance(contract.get("suite"), str):
            raise ManifestError(f"runtime benchmark contract is invalid: {candidate}")
        found.append(
            {
                "runtime": runtime,
                "source": sources.get(candidate),
                "benchmark": benchmark,
                "score": score,
                "evidence": evidence.get(candidate),
                "release_metadata": release_metadata,
            }
        )
    if require_sources and set(sources) != {item["runtime"]["id"] for item in found}:
        raise ManifestError("publication sources contain unknown runtime candidates")
    if not found:
        raise ManifestError("repository contains no runtime candidates")
    return found


def _previous_releases(previous: dict[str, Any] | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if previous is None:
        return {}
    if previous.get("schema_version") != SCHEMA_VERSION:
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


def generate(
    root: pathlib.Path,
    sources: dict[str, str],
    evidence: dict[str, str] | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = evidence or {}
    items = candidates(root, sources, evidence)
    history = _previous_releases(previous)
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
        benchmark_record = (
            None
            if item["benchmark"] is None
            else {
                "id": item["benchmark"]["id"],
                "suite": runtime["benchmark"]["contract"]["suite"],
                "score": item["score"],
                "evidence": item["evidence"],
            }
        )
        if benchmark_record is not None and not OCI_RE.fullmatch(
            str(benchmark_record["evidence"])
        ):
            raise ManifestError(
                f"published benchmark evidence is missing for {candidate}"
            )
        release = {
            "authors": item["release_metadata"]["authors"],
            "source": item["source"],
            "qualified": runtime["status"] == "qualified",
            "revoked": False,
            "engine": runtime["engine"]["id"],
            "engine_oci": runtime["engine"]["oci"]["reference"],
            "model_uri": runtime["model"]["uri"],
            "license": item["release_metadata"]["license"],
            "benchmark": benchmark_record,
        }
        releases = history.get((runtime["logical_model"], target_id, candidate), {})
        existing = releases.get(runtime["version"])
        if existing is not None and existing != release:
            metadata_upgrade = (
                set(existing).issubset(release)
                and all(release[key] == value for key, value in existing.items())
                and set(release) - set(existing) <= {"authors", "license"}
            )
            if not metadata_upgrade:
                raise ManifestError(
                    f"immutable release changed for {candidate}@{runtime['version']}"
                )
        releases[runtime["version"]] = release
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
                if release["qualified"]
                and not release["revoked"]
                and release["benchmark"] is not None
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
    result.add_argument("--evidence", action="append", default=[])
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
    evidence = evidence_from_manifest(previous_path) if previous_path is not None else {}
    explicit = source_map(arguments.source)
    sources.update(explicit)
    evidence.update(source_map(arguments.evidence))
    if arguments.validate_only:
        items = candidates(root, sources, evidence, require_sources=False)
        print(f"VALID candidates={len(items)}")
        return 0
    document = generate(root, sources, evidence, previous)
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
