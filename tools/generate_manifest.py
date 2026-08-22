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


SCHEMA_VERSION = 4
RUNTIME_SCHEMA_VERSION = 3
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OCI_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
CANDIDATE_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]*--[a-z0-9][a-z0-9._-]*--"
    r"[a-z0-9][a-z0-9._-]*--[a-z0-9][a-z0-9._-]*"
)


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
                if (
                    not CANDIDATE_RE.fullmatch(str(candidate))
                    or not isinstance(record, dict)
                    or not OCI_RE.fullmatch(str(record.get("source")))
                    or candidate in sources
                ):
                    raise ManifestError("existing manifest publication source is invalid")
                sources[candidate] = record["source"]
    return sources


def candidates(
    root: pathlib.Path,
    sources: dict[str, str],
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
        if runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise ManifestError(f"unsupported runtime schema in {directory.name}")
        candidate = normalized_candidate(runtime)
        if runtime.get("id") != candidate or directory.name != candidate:
            raise ManifestError(f"candidate directory and runtime.id differ: {directory.name}")
        if (directory / "release.json").exists():
            raise ManifestError(f"legacy release.json is forbidden: {candidate}")
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
            }
        )
    if require_sources and set(sources) != {item["runtime"]["id"] for item in found}:
        raise ManifestError("publication sources contain unknown runtime candidates")
    if not found:
        raise ManifestError("repository contains no runtime candidates")
    return found


def generate(root: pathlib.Path, sources: dict[str, str]) -> dict[str, Any]:
    items = candidates(root, sources)
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
        model_target["candidates"][candidate] = {
            "version": runtime["version"],
            "source": item["source"],
            "qualified": runtime["status"] == "qualified",
            "engine": runtime["engine"]["id"],
            "engine_oci": runtime["engine"]["oci"]["reference"],
            "model_uri": runtime["model"]["uri"],
            "benchmark": (
                None
                if item["benchmark"] is None
                else {
                    "id": item["benchmark"]["id"],
                    "suite": runtime["benchmark"]["contract"]["suite"],
                    "score": item["score"],
                }
            ),
        }
    for model in models.values():
        for target in model["targets"].values():
            qualified = [
                (record["benchmark"]["score"], candidate)
                for candidate, record in target["candidates"].items()
                if record["qualified"] and record["benchmark"] is not None
            ]
            if qualified:
                target["recommended"] = max(qualified)[1]
    return {"schema_version": SCHEMA_VERSION, "targets": targets, "models": models}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    result.add_argument("--source", action="append", default=[])
    result.add_argument("--sources-from", type=pathlib.Path)
    result.add_argument("--output", type=pathlib.Path)
    result.add_argument("--check", action="store_true")
    result.add_argument("--validate-only", action="store_true")
    return result


def main() -> int:
    arguments = parser().parse_args()
    root = arguments.root.resolve(strict=True)
    if arguments.check and arguments.validate_only:
        raise ManifestError("--check and --validate-only are mutually exclusive")
    sources = (
        sources_from_manifest(arguments.sources_from.resolve(strict=True))
        if arguments.sources_from is not None
        else {}
    )
    explicit = source_map(arguments.source)
    overlap = set(sources).intersection(explicit)
    if overlap:
        raise ManifestError(
            "publication source supplied twice: " + ", ".join(sorted(overlap))
        )
    sources.update(explicit)
    if arguments.validate_only:
        items = candidates(root, sources, require_sources=False)
        print(f"VALID candidates={len(items)}")
        return 0
    document = generate(root, sources)
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
