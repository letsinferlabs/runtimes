#!/usr/bin/env python3
"""Create and validate the immutable artifact consumed by runtime verifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import sys
from collections.abc import Mapping
from typing import Any

if __package__:
    from tools import candidate_policy, engine_sbom, generate_manifest, oci_artifact, oci_layout
else:
    import candidate_policy
    import engine_sbom
    import generate_manifest
    import oci_artifact
    import oci_layout


SCHEMA_VERSION = 1
REPOSITORY = "letsinferlabs/runtimes"
BUNDLE_FILES = {
    "runtime.letsinfer",
    "runtime-plan.json",
    "candidate-audit.json",
    "runtime.spdx.json",
    "provenance.json",
}
ENGINE_FILES = {"engine.oci.tar", "engine.spdx.json"}
RAW_COMMON_FILES = {
    "build.json",
    "classification.json",
    "candidate-audit.json",
    "runtime-a.letsinfer",
    "runtime-b.letsinfer",
    "runtime-plan.json",
    "runtime-trusted.letsinfer",
}
RAW_ENGINE_FILES = {
    "engine.oci.tar",
    "engine-a-plan.json",
    "engine-inventory.json",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class BundleError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{path.name} must contain an object")
    return value


def _copy_exact(source: pathlib.Path, destination: pathlib.Path) -> None:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise BundleError(f"bundle input is not a regular file: {source.name}")
    if destination.exists():
        raise BundleError(f"refusing to overwrite bundle file: {destination.name}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)


def _require_exact_raw_files(raw: pathlib.Path, mode: str) -> None:
    expected = RAW_COMMON_FILES | (RAW_ENGINE_FILES if mode == "build-engine" else set())
    actual: set[str] = set()
    for path in raw.iterdir():
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(metadata.st_mode):
            raise BundleError(f"raw verifier artifact contains a non-regular entry: {path.name}")
        actual.add(path.name)
    if actual != expected:
        raise BundleError(
            f"raw verifier artifact file set differs (extra={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)})"
        )


def _core_subject(runtime: Mapping[str, Any], pack: pathlib.Path) -> dict[str, Any]:
    root = pathlib.Path(os.environ.get("LETSINFER_CORE_ROOT", ".core")).resolve()
    if not (root / "core" / "benchmark_verification.py").is_file():
        raise BundleError("trusted core verification contract is unavailable")
    sys.path.insert(0, str(root))
    try:
        from core.benchmark_verification import execution_subject
    except ImportError as error:
        raise BundleError("cannot load trusted core verification contract") from error
    try:
        return execution_subject(
            runtime,
            pack_sha256=sha256_file(pack),
            pack_bytes=pack.stat().st_size,
        )
    except Exception as error:
        raise BundleError(f"runtime execution subject is invalid: {error}") from error


def runtime_spdx(
    *, candidate: str, pack_sha256: str, pack_bytes: int, tree_sha256: str
) -> dict[str, Any]:
    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: letsinfer-verifier-bundle/1"],
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": f"https://letsinfer.ai/spdx/runtime/{pack_sha256}",
        "name": f"{candidate}-runtime-{pack_sha256[:12]}",
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-Runtime",
                "checksums": [{"algorithm": "SHA256", "checksumValue": pack_sha256}],
                "copyrightText": "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "name": candidate,
                "supplier": "Organization: letsinferlabs",
                "versionInfo": tree_sha256,
            }
        ],
        "relationships": [
            {
                "relatedSpdxElement": "SPDXRef-Package-Runtime",
                "relationshipType": "DESCRIBES",
                "spdxElementId": "SPDXRef-DOCUMENT",
            }
        ],
        "spdxVersion": "SPDX-2.3",
        "annotations": {"runtime_pack_bytes": pack_bytes},
    }


def create(
    *,
    root: pathlib.Path,
    raw: pathlib.Path,
    output: pathlib.Path,
    candidate: str,
    mode: str,
    pull_request: int,
    proposal_head_sha: str,
    proposal_base_sha: str,
    proposal_tree_sha: str,
    build_run_id: int,
    build_workflow_sha: str,
    finalizer_run_id: int,
    finalizer_workflow_sha: str,
) -> dict[str, Any]:
    if (
        mode not in {"reuse-engine", "build-engine", "build-native-engine"}
        or pull_request <= 0
        or build_run_id <= 0
        or finalizer_run_id <= 0
        or any(
            COMMIT_RE.fullmatch(value) is None
            for value in (
                proposal_head_sha,
                proposal_base_sha,
                build_workflow_sha,
                finalizer_workflow_sha,
            )
        )
        or SHA256_RE.fullmatch(proposal_tree_sha) is None
    ):
        raise BundleError("bundle workflow identity is invalid")
    root = root.resolve(strict=True)
    raw = raw.resolve(strict=True)
    _require_exact_raw_files(raw, mode)
    if output.exists():
        raise BundleError("bundle output already exists")
    output.mkdir(parents=True, mode=0o700)
    runtime_a = raw / "runtime-a.letsinfer"
    runtime_b = raw / "runtime-b.letsinfer"
    trusted_pack = raw / "runtime-trusted.letsinfer"
    digests = {sha256_file(path) for path in (runtime_a, runtime_b, trusted_pack)}
    if len(digests) != 1:
        raise BundleError("untrusted and trusted runtime pack bytes differ")
    runtime_path = root / candidate / "runtime.json"
    runtime = generate_manifest.read_object(runtime_path)
    release = generate_manifest.read_object(root / candidate / "release.json")
    authors = release.get("authors")
    if not isinstance(authors, list) or not authors:
        raise BundleError("runtime authors are unavailable")
    for index, author in enumerate(authors):
        generate_manifest.github_identity(
            author, f"{candidate}.authors[{index}]", allow_organization=True
        )
    version = runtime.get("version")
    if not isinstance(version, str):
        raise BundleError("runtime version is invalid")
    audit = candidate_policy.audit_candidate(root, candidate, mode)
    if read_object(raw / "classification.json") != {
        "candidate": candidate,
        "mode": mode,
    }:
        raise BundleError("trusted and build-stage candidate classifications differ")
    if read_object(raw / "candidate-audit.json") != audit:
        raise BundleError("untrusted and trusted candidate audits differ")
    expected_plan = oci_artifact.plan(
        trusted_pack,
        repository=candidate_policy.runtime_repository(
            root, candidate, proposal_base_sha
        ),
        candidate=candidate,
        version=version,
    ).document()
    if read_object(raw / "runtime-plan.json") != expected_plan:
        raise BundleError("runtime OCI plan differs from exact source pack")
    distribution = generate_manifest.engine_distribution(runtime)
    engine_value: dict[str, Any] = {
        "mode": mode,
        "kind": distribution["kind"],
    }
    if distribution["kind"] == "oci-container":
        engine_value |= {
            "reference": audit["engine_reference"],
            "config_digest": audit["engine_config_digest"],
        }
        if distribution.get("payload_id") is not None:
            engine_value["payload_digest"] = distribution["payload_id"]
    else:
        engine_value |= {
            "payload_digest": distribution["payload_id"],
            "platform": distribution["platform"],
            "source_revision": distribution["source_revision"],
        }
    if mode == "build-engine":
        first = oci_layout.inspect_archive(raw / "engine.oci.tar", audit["target_platform"])
        expected_engine_plan = first | {"reference": audit["engine_reference"]}
        if read_object(raw / "engine-a-plan.json") != expected_engine_plan:
            raise BundleError("Engine build plan differs from its OCI layout")
        if (
            first["manifest_digest"] != str(audit["engine_reference"]).rsplit("@", 1)[-1]
            or first["config_digest"] != audit["engine_config_digest"]
            or (
                audit.get("engine_payload_digest") is not None
                and first.get("payload_digest") != audit["engine_payload_digest"]
            )
        ):
            raise BundleError(
                "runtime.json must pin the deterministic production Engine identity before verification"
            )
        engine_value |= first
    subject = _core_subject(runtime, trusted_pack)
    subject_base = dict(subject)
    subject_base.update(
        {
            "artifact_schema_version": SCHEMA_VERSION,
            "repository": REPOSITORY,
            "pull_request": pull_request,
            "proposal_head_sha": proposal_head_sha,
            "proposal_base_sha": proposal_base_sha,
            "proposal_tree_sha256": proposal_tree_sha,
            "engine_mode": mode,
            "build_workflow_run_id": build_run_id,
        }
    )
    subject_base.pop("execution_sha256", None)
    subject = subject_base | {
        "execution_sha256": hashlib.sha256(canonical_bytes(subject_base)).hexdigest()
    }
    _copy_exact(trusted_pack, output / "runtime.letsinfer")
    (output / "runtime-plan.json").write_bytes(canonical_bytes(expected_plan))
    (output / "candidate-audit.json").write_bytes(canonical_bytes(audit))
    (output / "runtime.spdx.json").write_bytes(
        canonical_bytes(
            runtime_spdx(
                candidate=candidate,
                pack_sha256=sha256_file(trusted_pack),
                pack_bytes=trusted_pack.stat().st_size,
                tree_sha256=proposal_tree_sha,
            )
        )
    )
    if mode == "build-engine":
        _copy_exact(raw / "engine.oci.tar", output / "engine.oci.tar")
        inventory = engine_sbom.read_inventory(raw / "engine-inventory.json")
        sbom = engine_sbom.spdx(
            inventory, candidate, str(audit["engine_reference"]), str(audit["engine_config_digest"])
        )
        (output / "engine.spdx.json").write_bytes(canonical_bytes(sbom))
    provenance = {
        "schema_version": 1,
        "builder": "https://github.com/letsinferlabs/runtimes/.github/workflows/finalize-verifier.yml",
        "build_type": "https://letsinfer.ai/build-types/runtime-verifier-bundle/v1",
        "invocation": {
            "repository": REPOSITORY,
            "pull_request": pull_request,
            "proposal_head_sha": proposal_head_sha,
            "proposal_base_sha": proposal_base_sha,
            "proposal_tree_sha256": proposal_tree_sha,
            "build_workflow_run_id": build_run_id,
            "build_workflow_sha": build_workflow_sha,
            "finalizer_workflow_run_id": finalizer_run_id,
            "finalizer_workflow_sha": finalizer_workflow_sha,
        },
        "subject": subject,
        "engine": engine_value,
    }
    (output / "provenance.json").write_bytes(canonical_bytes(provenance))
    expected = set(BUNDLE_FILES) | (ENGINE_FILES if mode == "build-engine" else set())
    checksums = {
        name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
        for name in sorted(expected)
    }
    (output / "checksums.json").write_bytes(canonical_bytes(checksums))
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "repository": REPOSITORY,
        "pull_request": pull_request,
        "proposal_head_sha": proposal_head_sha,
        "proposal_base_sha": proposal_base_sha,
        "proposal_tree_sha256": proposal_tree_sha,
        "candidate": candidate,
        "runtime_authors": authors,
        "mode": mode,
        "artifact_name": f"verification-bundle-pr-{pull_request}-{proposal_head_sha}",
        "build_workflow": {
            "path": ".github/workflows/build-verifier.yml",
            "run_id": build_run_id,
            "workflow_sha": build_workflow_sha,
        },
        "finalizer_workflow": {
            "path": ".github/workflows/finalize-verifier.yml",
            "run_id": finalizer_run_id,
            "workflow_sha": finalizer_workflow_sha,
        },
        "subject": subject,
        "engine": engine_value,
        "runtime": expected_plan,
        "checksums_sha256": sha256_file(output / "checksums.json"),
    }
    (output / "bundle.json").write_bytes(canonical_bytes(bundle))
    validate(output, expected_pr=pull_request, expected_head=proposal_head_sha)
    return bundle


def validate(
    directory: pathlib.Path,
    *,
    expected_pr: int | None = None,
    expected_head: str | None = None,
) -> dict[str, Any]:
    directory = directory.resolve(strict=True)
    if directory.is_symlink() or not directory.is_dir():
        raise BundleError("bundle must be a regular directory")
    bundle = read_object(directory / "bundle.json")
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("repository") != REPOSITORY:
        raise BundleError("unsupported verifier bundle")
    build_workflow = bundle.get("build_workflow")
    finalizer_workflow = bundle.get("finalizer_workflow")
    proposal_base = bundle.get("proposal_base_sha")
    if (
        COMMIT_RE.fullmatch(str(bundle.get("proposal_head_sha"))) is None
        or COMMIT_RE.fullmatch(str(proposal_base)) is None
        or SHA256_RE.fullmatch(str(bundle.get("proposal_tree_sha256"))) is None
        or not isinstance(build_workflow, dict)
        or build_workflow.get("path") != ".github/workflows/build-verifier.yml"
        or not isinstance(build_workflow.get("run_id"), int)
        or build_workflow["run_id"] <= 0
        or build_workflow.get("workflow_sha") != proposal_base
        or not isinstance(finalizer_workflow, dict)
        or finalizer_workflow.get("path") != ".github/workflows/finalize-verifier.yml"
        or not isinstance(finalizer_workflow.get("run_id"), int)
        or finalizer_workflow["run_id"] <= 0
        or COMMIT_RE.fullmatch(str(finalizer_workflow.get("workflow_sha"))) is None
        or bundle.get("artifact_name")
        != f"verification-bundle-pr-{bundle.get('pull_request')}-{bundle.get('proposal_head_sha')}"
    ):
        raise BundleError("verifier bundle workflow provenance is invalid")
    mode = bundle.get("mode")
    expected = set(BUNDLE_FILES) | (ENGINE_FILES if mode == "build-engine" else set())
    entries = list(directory.iterdir())
    if any(
        path.is_symlink()
        or not path.is_file()
        or not stat.S_ISREG(path.lstat().st_mode)
        for path in entries
    ):
        raise BundleError("verifier bundle contains a non-regular entry")
    actual = {path.name for path in entries}
    if actual != expected | {"bundle.json", "checksums.json"}:
        raise BundleError("verifier bundle file set differs")
    checksums_path = directory / "checksums.json"
    if sha256_file(checksums_path) != bundle.get("checksums_sha256"):
        raise BundleError("verifier bundle checksum manifest differs")
    checksums = read_object(checksums_path)
    if set(checksums) != expected:
        raise BundleError("verifier bundle checksum file set differs")
    for name, record in checksums.items():
        path = directory / name
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "bytes"}
            or SHA256_RE.fullmatch(str(record.get("sha256"))) is None
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise BundleError(f"verifier bundle payload differs: {name}")
    if expected_pr is not None and bundle.get("pull_request") != expected_pr:
        raise BundleError("verifier bundle pull request differs")
    if expected_head is not None and bundle.get("proposal_head_sha") != expected_head:
        raise BundleError("verifier bundle proposal head differs")
    subject = bundle.get("subject")
    if not isinstance(subject, dict):
        raise BundleError("verifier bundle subject is invalid")
    without_execution = dict(subject)
    execution = without_execution.pop("execution_sha256", None)
    if execution != hashlib.sha256(canonical_bytes(without_execution)).hexdigest():
        raise BundleError("verifier bundle execution subject differs")
    provenance = read_object(directory / "provenance.json")
    if provenance.get("subject") != subject or provenance.get("engine") != bundle.get("engine"):
        raise BundleError("verifier bundle provenance differs")
    plan = read_object(directory / "runtime-plan.json")
    if plan != bundle.get("runtime") or plan.get("layer_digest") != "sha256:" + sha256_file(directory / "runtime.letsinfer"):
        raise BundleError("verifier bundle runtime plan differs")
    if mode == "build-engine":
        engine = bundle.get("engine")
        if not isinstance(engine, dict):
            raise BundleError("verifier bundle Engine identity is invalid")
        layout = oci_layout.inspect_archive(directory / "engine.oci.tar", str(engine.get("platform")))
        if any(layout.get(key) != engine.get(key) for key in ("platform", "manifest_digest", "config_digest", "layer_digests")):
            raise BundleError("verifier bundle Engine layout differs")
    elif mode == "build-native-engine":
        engine = bundle.get("engine")
        if (
            not isinstance(engine, dict)
            or engine.get("kind")
            not in {"native-archive", "python-standalone", "embedded-application"}
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(engine.get("payload_digest")))
            is None
            or re.fullmatch(r"[0-9a-f]{40}", str(engine.get("source_revision")))
            is None
            or re.fullmatch(r"[a-z0-9._-]+/[a-z0-9._-]+", str(engine.get("platform")))
            is None
        ):
            raise BundleError("verifier bundle native Engine identity is invalid")
    elif mode != "reuse-engine":
        raise BundleError("verifier bundle Engine mode is invalid")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    finalize.add_argument("--raw", type=pathlib.Path, required=True)
    finalize.add_argument("--output", type=pathlib.Path, required=True)
    finalize.add_argument("--candidate", required=True)
    finalize.add_argument("--mode", required=True)
    finalize.add_argument("--pull-request", type=int, required=True)
    finalize.add_argument("--proposal-head-sha", required=True)
    finalize.add_argument("--proposal-base-sha", required=True)
    finalize.add_argument("--proposal-tree-sha", required=True)
    finalize.add_argument("--build-run-id", type=int, required=True)
    finalize.add_argument("--build-workflow-sha", required=True)
    finalize.add_argument("--finalizer-run-id", type=int, required=True)
    finalize.add_argument("--finalizer-workflow-sha", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--bundle", type=pathlib.Path, required=True)
    check.add_argument("--pull-request", type=int)
    check.add_argument("--proposal-head-sha")
    arguments = parser.parse_args()
    if arguments.command == "finalize":
        value = create(
            root=arguments.root,
            raw=arguments.raw,
            output=arguments.output,
            candidate=arguments.candidate,
            mode=arguments.mode,
            pull_request=arguments.pull_request,
            proposal_head_sha=arguments.proposal_head_sha,
            proposal_base_sha=arguments.proposal_base_sha,
            proposal_tree_sha=arguments.proposal_tree_sha,
            build_run_id=arguments.build_run_id,
            build_workflow_sha=arguments.build_workflow_sha,
            finalizer_run_id=arguments.finalizer_run_id,
            finalizer_workflow_sha=arguments.finalizer_workflow_sha,
        )
    else:
        value = validate(
            arguments.bundle,
            expected_pr=arguments.pull_request,
            expected_head=arguments.proposal_head_sha,
        )
    print(canonical_bytes(value).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        BundleError,
        candidate_policy.CandidatePolicyError,
        engine_sbom.SbomError,
        generate_manifest.ManifestError,
        oci_artifact.OciError,
        oci_layout.LayoutError,
    ) as error:
        raise SystemExit(f"FATAL: {error}")
