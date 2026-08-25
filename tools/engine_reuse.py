#!/usr/bin/env python3
"""Reuse a finalizer-attested Engine build for identical Engine source bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

if __package__:
    from tools import engine_sbom, oci_layout, verifier_bundle
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from tools import engine_sbom, oci_layout, verifier_bundle


REPOSITORY = "letsinferlabs/runtimes"
SCHEMA_VERSION = 1
PROOF_FILES = {"engine-proof.json", "engine-inventory.json"}
CONTRACT_PATHS = (
    ".github/workflows/build-verifier.yml",
    "tools/candidate_policy.py",
    "tools/engine_reuse.py",
    "tools/engine_sbom.py",
    "tools/oci_layout.py",
    "tools/verifier_bundle.py",
)
FINALIZER_CERT_IDENTITY = (
    "https://github.com/letsinferlabs/runtimes/"
    ".github/workflows/finalize-verifier.yml@refs/heads/main"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ReuseError(RuntimeError):
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
        raise ReuseError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReuseError(f"{path} must contain an object")
    return value


def builder_contract(root: pathlib.Path) -> str:
    root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for name in CONTRACT_PATHS:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ReuseError(f"Engine build contract input is missing: {name}")
        records.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return hashlib.sha256(
        canonical_bytes({"schema_version": 1, "inputs": records})
    ).hexdigest()


def _identity(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ReuseError(f"Engine reuse {label} is invalid")
    return value


def _validate_proof(
    directory: pathlib.Path,
    *,
    audit: Mapping[str, Any] | None = None,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    directory = directory.resolve(strict=True)
    entries = list(directory.iterdir())
    if (
        {path.name for path in entries} != PROOF_FILES
        or any(
            path.is_symlink()
            or not path.is_file()
            or not stat.S_ISREG(path.lstat().st_mode)
            for path in entries
        )
    ):
        raise ReuseError("Engine proof file set is invalid")
    proof = read_object(directory / "engine-proof.json")
    expected = {
        "schema_version",
        "candidate",
        "engine_source_sha256",
        "builder_contract_sha256",
        "candidate_audit_sha256",
        "engine_reference",
        "engine_config_digest",
        "target_platform",
        "engine_archive_sha256",
        "engine_archive_bytes",
        "engine_spdx_sha256",
        "engine_spdx_bytes",
        "inventory_sha256",
        "bundle",
        "finalizer",
    }
    bundle = proof.get("bundle")
    finalizer = proof.get("finalizer")
    pull_request = (
        finalizer.get("pull_request") if isinstance(finalizer, dict) else None
    )
    if (
        set(proof) != expected
        or proof.get("schema_version") != SCHEMA_VERSION
        or not isinstance(proof.get("candidate"), str)
        or not proof["candidate"]
        or any(
            SHA256_RE.fullmatch(str(proof.get(key))) is None
            for key in (
                "engine_source_sha256",
                "builder_contract_sha256",
                "candidate_audit_sha256",
                "engine_archive_sha256",
                "engine_spdx_sha256",
                "inventory_sha256",
            )
        )
        or not isinstance(proof.get("engine_archive_bytes"), int)
        or proof["engine_archive_bytes"] <= 0
        or not isinstance(proof.get("engine_spdx_bytes"), int)
        or proof["engine_spdx_bytes"] <= 0
        or not isinstance(proof.get("target_platform"), str)
        or DIGEST_RE.fullmatch(str(proof.get("engine_config_digest"))) is None
        or re.fullmatch(
            r"ghcr\.io/letsinferlabs/(?:engines/[^@]+|engine-images)"
            r"@sha256:[0-9a-f]{64}",
            str(proof.get("engine_reference")),
        )
        is None
        or not isinstance(bundle, dict)
        or set(bundle)
        != {"artifact_id", "artifact_digest", "artifact_name", "proposal_head_sha"}
        or not isinstance(bundle.get("artifact_id"), int)
        or bundle["artifact_id"] <= 0
        or DIGEST_RE.fullmatch(str(bundle.get("artifact_digest"))) is None
        or bundle.get("artifact_name")
        != f"verification-bundle-pr-{pull_request}-{bundle.get('proposal_head_sha')}"
        or COMMIT_RE.fullmatch(str(bundle.get("proposal_head_sha"))) is None
        or not isinstance(finalizer, dict)
        or set(finalizer)
        != {"pull_request", "run_id", "workflow_sha"}
        or not isinstance(finalizer.get("pull_request"), int)
        or finalizer["pull_request"] <= 0
        or not isinstance(finalizer.get("run_id"), int)
        or finalizer["run_id"] <= 0
        or COMMIT_RE.fullmatch(str(finalizer.get("workflow_sha"))) is None
    ):
        raise ReuseError("Engine proof is invalid")
    inventory_path = directory / "engine-inventory.json"
    if proof["inventory_sha256"] != sha256_file(inventory_path):
        raise ReuseError("Engine proof inventory digest differs")
    engine_sbom.read_inventory(inventory_path)
    if audit is not None and any(
        proof.get(proof_key) != audit.get(audit_key)
        for proof_key, audit_key in (
            ("candidate", "candidate"),
            ("engine_source_sha256", "engine_source_sha256"),
            ("engine_reference", "engine_reference"),
            ("engine_config_digest", "engine_config_digest"),
            ("target_platform", "target_platform"),
        )
    ):
        raise ReuseError("Engine proof differs from current Engine source or identity")
    if contract_sha256 is not None and proof["builder_contract_sha256"] != contract_sha256:
        raise ReuseError("Engine proof uses a different build contract")
    return proof


def create_proof(
    *,
    trusted_root: pathlib.Path,
    raw: pathlib.Path,
    bundle_root: pathlib.Path,
    output: pathlib.Path,
    bundle_artifact_id: int,
    bundle_artifact_digest: str,
    bundle_artifact_name: str,
    finalizer_run_id: int,
    finalizer_workflow_sha: str,
) -> dict[str, Any]:
    raw = raw.resolve(strict=True)
    bundle_root = bundle_root.resolve(strict=True)
    audit_path = raw / "candidate-audit.json"
    audit = read_object(audit_path)
    build = read_object(raw / "build.json")
    contract_sha256 = _identity(
        build.get("engine_build_contract_sha256"), SHA256_RE, "build contract"
    )
    if contract_sha256 != builder_contract(trusted_root):
        raise ReuseError("Engine proof build contract differs from trusted tooling")
    if SHA256_RE.fullmatch(bundle_artifact_digest):
        bundle_artifact_digest = "sha256:" + bundle_artifact_digest
    bundle = verifier_bundle.validate(bundle_root)
    inventory_path = raw / "engine-inventory.json"
    engine_sbom.read_inventory(inventory_path)
    engine_archive = bundle_root / "engine.oci.tar"
    engine_spdx = bundle_root / "engine.spdx.json"
    if (
        bundle.get("mode") != "build-engine"
        or bundle.get("candidate") != audit.get("candidate")
        or bundle_artifact_id <= 0
        or DIGEST_RE.fullmatch(bundle_artifact_digest) is None
        or bundle_artifact_name != bundle.get("artifact_name")
        or finalizer_run_id <= 0
        or COMMIT_RE.fullmatch(finalizer_workflow_sha) is None
        or bundle.get("finalizer_workflow", {}).get("run_id") != finalizer_run_id
        or bundle.get("finalizer_workflow", {}).get("workflow_sha")
        != finalizer_workflow_sha
    ):
        raise ReuseError("Engine proof inputs are inconsistent")
    engine = bundle["engine"]
    proof = {
        "schema_version": SCHEMA_VERSION,
        "candidate": audit["candidate"],
        "engine_source_sha256": audit["engine_source_sha256"],
        "builder_contract_sha256": contract_sha256,
        "candidate_audit_sha256": sha256_file(audit_path),
        "engine_reference": audit["engine_reference"],
        "engine_config_digest": audit["engine_config_digest"],
        "target_platform": audit["target_platform"],
        "engine_archive_sha256": sha256_file(engine_archive),
        "engine_archive_bytes": engine_archive.stat().st_size,
        "engine_spdx_sha256": sha256_file(engine_spdx),
        "engine_spdx_bytes": engine_spdx.stat().st_size,
        "inventory_sha256": sha256_file(inventory_path),
        "bundle": {
            "artifact_id": bundle_artifact_id,
            "artifact_digest": bundle_artifact_digest,
            "artifact_name": bundle_artifact_name,
            "proposal_head_sha": bundle["proposal_head_sha"],
        },
        "finalizer": {
            "pull_request": bundle["pull_request"],
            "run_id": finalizer_run_id,
            "workflow_sha": finalizer_workflow_sha,
        },
    }
    if any(
        engine.get(key) != value
        for key, value in (
            ("reference", proof["engine_reference"]),
            ("config_digest", proof["engine_config_digest"]),
            ("platform", proof["target_platform"]),
        )
    ):
        raise ReuseError("Engine proof bundle identity differs")
    if output.exists():
        raise ReuseError("Engine proof output already exists")
    output.mkdir(parents=True, mode=0o700)
    (output / "engine-proof.json").write_bytes(canonical_bytes(proof))
    shutil.copyfile(inventory_path, output / "engine-inventory.json")
    _validate_proof(output, audit=audit, contract_sha256=proof["builder_contract_sha256"])
    return proof


def _run(
    command: Sequence[str],
    *,
    token: str,
    output: pathlib.Path | None = None,
) -> bytes:
    environment = dict(os.environ)
    environment["GH_TOKEN"] = token
    if output is None:
        result = subprocess.run(
            list(command), capture_output=True, check=False, env=environment
        )
        stdout = result.stdout
    else:
        with output.open("xb") as handle:
            result = subprocess.run(
                list(command), stdout=handle, stderr=subprocess.PIPE, check=False, env=environment
            )
        stdout = b""
    if result.returncode:
        if output is not None:
            output.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReuseError(detail or f"command failed: {command[0]}")
    return stdout


def _gh_json(endpoint: str, *, token: str) -> Any:
    try:
        return json.loads(_run(["gh", "api", endpoint], token=token))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReuseError(f"GitHub returned invalid JSON for {endpoint}") from error


def _extract_zip(
    archive: pathlib.Path,
    output: pathlib.Path,
    *,
    max_files: int,
    max_bytes: int,
) -> None:
    output.mkdir(mode=0o700)
    try:
        source = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as error:
        raise ReuseError("GitHub artifact is not a zip") from error
    with source:
        members = source.infolist()
        if not members or len(members) > max_files:
            raise ReuseError("GitHub artifact file count is invalid")
        total = 0
        names: set[str] = set()
        for member in members:
            path = pathlib.PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            total += member.file_size
            if (
                path.is_absolute()
                or len(path.parts) != 1
                or any(part in {"", ".", ".."} for part in path.parts)
                or member.filename in names
                or member.is_dir()
                or (mode and not stat.S_ISREG(mode))
                or total > max_bytes
            ):
                raise ReuseError("GitHub artifact contains an unsafe or oversized entry")
            names.add(member.filename)
            target = output / member.filename
            with source.open(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)


def _download_artifact(artifact_id: int, output: pathlib.Path, *, token: str) -> None:
    archive = output.with_suffix(".zip")
    _run(
        ["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"],
        token=token,
        output=archive,
    )
    try:
        _extract_zip(archive, output, max_files=32, max_bytes=20 << 30)
    finally:
        archive.unlink(missing_ok=True)


def _verify_attestations(root: pathlib.Path, *, token: str) -> None:
    for path in sorted(root.iterdir()):
        _run(
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
            token=token,
        )


def _artifact(value: Any, *, name: str | None = None) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("id"), int)
        or value["id"] <= 0
        or value.get("expired") is not False
        or DIGEST_RE.fullmatch(str(value.get("digest"))) is None
        or (name is not None and value.get("name") != name)
    ):
        raise ReuseError("GitHub artifact metadata is invalid")
    return value


def _verify_finalizer(artifact: Mapping[str, Any], proof: Mapping[str, Any], *, token: str) -> None:
    workflow = artifact.get("workflow_run")
    run_id = workflow.get("id") if isinstance(workflow, Mapping) else None
    if run_id != proof["finalizer"]["run_id"]:
        raise ReuseError("Engine proof finalizer run differs")
    run = _gh_json(f"repos/{REPOSITORY}/actions/runs/{run_id}", token=token)
    if (
        not isinstance(run, dict)
        or run.get("event") != "workflow_run"
        or run.get("path") != ".github/workflows/finalize-verifier.yml"
        or run.get("conclusion") != "success"
        or run.get("head_branch") != "main"
        or run.get("head_sha") != proof["finalizer"]["workflow_sha"]
    ):
        raise ReuseError("Engine proof was not produced by the trusted finalizer")


def _copy_engine(
    *,
    proof_root: pathlib.Path,
    bundle_root: pathlib.Path,
    raw: pathlib.Path,
    audit: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    bundle = verifier_bundle.validate(bundle_root)
    archive = bundle_root / "engine.oci.tar"
    spdx = bundle_root / "engine.spdx.json"
    prior_audit = read_object(bundle_root / "candidate-audit.json")
    if (
        bundle.get("mode") != "build-engine"
        or bundle.get("candidate") != audit.get("candidate")
        or bundle.get("proposal_head_sha") != proof["bundle"]["proposal_head_sha"]
        or bundle.get("finalizer_workflow", {}).get("run_id")
        != proof["finalizer"]["run_id"]
        or prior_audit.get("engine_source_sha256") != audit.get("engine_source_sha256")
        or proof["candidate_audit_sha256"]
        != sha256_file(bundle_root / "candidate-audit.json")
        or proof["engine_archive_sha256"] != sha256_file(archive)
        or proof["engine_archive_bytes"] != archive.stat().st_size
        or proof["engine_spdx_sha256"] != sha256_file(spdx)
        or proof["engine_spdx_bytes"] != spdx.stat().st_size
    ):
        raise ReuseError("attested Engine bundle differs from its proof")
    target = raw / "engine.oci.tar"
    shutil.copyfile(archive, target)
    inventory = proof_root / "engine-inventory.json"
    shutil.copyfile(inventory, raw / "engine-inventory.json")
    layout = oci_layout.inspect_archive(target, str(audit["target_platform"]))
    expected = layout | {"reference": audit["engine_reference"]}
    if (
        layout.get("manifest_digest")
        != str(audit["engine_reference"]).rsplit("@", 1)[-1]
        or layout.get("config_digest") != audit["engine_config_digest"]
    ):
        raise ReuseError("attested Engine bundle differs from the current pin")
    for name in ("engine-a-plan.json", "engine-b-plan.json"):
        (raw / name).write_bytes(canonical_bytes(expected))


def restore(
    *,
    trusted_root: pathlib.Path,
    raw: pathlib.Path,
    token: str,
) -> bool:
    audit = read_object(raw / "candidate-audit.json")
    source = _identity(audit.get("engine_source_sha256"), SHA256_RE, "source")
    contract = builder_contract(trusted_root)
    name = f"engine-proof-{source}"
    try:
        response = _gh_json(
            f"repos/{REPOSITORY}/actions/artifacts?name={name}&per_page=100",
            token=token,
        )
    except ReuseError:
        return False
    records = response.get("artifacts") if isinstance(response, dict) else None
    if not isinstance(records, list):
        raise ReuseError("GitHub Engine proof response is invalid")
    candidates = sorted(
        (
            item
            for item in records
            if isinstance(item, dict)
            and item.get("name") == name
            and item.get("expired") is False
        ),
        key=lambda item: int(item.get("id", 0)),
        reverse=True,
    )
    for value in candidates:
        try:
            artifact = _artifact(value, name=name)
            with tempfile.TemporaryDirectory(prefix="letsinfer-engine-reuse-") as temporary:
                temporary_root = pathlib.Path(temporary)
                proof_root = temporary_root / "proof"
                _download_artifact(artifact["id"], proof_root, token=token)
                _verify_attestations(proof_root, token=token)
                proof = _validate_proof(
                    proof_root, audit=audit, contract_sha256=contract
                )
                _verify_finalizer(artifact, proof, token=token)
                bundle_metadata = _artifact(
                    _gh_json(
                        f"repos/{REPOSITORY}/actions/artifacts/"
                        f"{proof['bundle']['artifact_id']}",
                        token=token,
                    ),
                    name=proof["bundle"]["artifact_name"],
                )
                if (
                    bundle_metadata["digest"] != proof["bundle"]["artifact_digest"]
                    or bundle_metadata.get("workflow_run", {}).get("id")
                    != proof["finalizer"]["run_id"]
                ):
                    raise ReuseError("Engine proof bundle artifact metadata differs")
                bundle_root = temporary_root / "bundle"
                _download_artifact(bundle_metadata["id"], bundle_root, token=token)
                _verify_attestations(bundle_root, token=token)
                _copy_engine(
                    proof_root=proof_root,
                    bundle_root=bundle_root,
                    raw=raw,
                    audit=audit,
                    proof=proof,
                )
                marker = {
                    "schema_version": SCHEMA_VERSION,
                    "proof_artifact_id": artifact["id"],
                    "proof_artifact_digest": artifact["digest"],
                    "proof_artifact_name": name,
                    "proof_sha256": sha256_file(proof_root / "engine-proof.json"),
                    "engine_source_sha256": source,
                }
                (raw / "engine-reuse.json").write_bytes(canonical_bytes(marker))
                return True
        except (ReuseError, verifier_bundle.BundleError, engine_sbom.SbomError, oci_layout.LayoutError):
            continue
    return False


def verify_restored(
    *,
    trusted_root: pathlib.Path,
    raw: pathlib.Path,
    token: str,
) -> dict[str, Any] | None:
    marker_path = raw / "engine-reuse.json"
    if not marker_path.is_file():
        return None
    marker = read_object(marker_path)
    if (
        set(marker)
        != {
            "schema_version",
            "proof_artifact_id",
            "proof_artifact_digest",
            "proof_artifact_name",
            "proof_sha256",
            "engine_source_sha256",
        }
        or marker.get("schema_version") != SCHEMA_VERSION
        or not isinstance(marker.get("proof_artifact_id"), int)
        or marker["proof_artifact_id"] <= 0
        or DIGEST_RE.fullmatch(str(marker.get("proof_artifact_digest"))) is None
        or marker.get("proof_artifact_name")
        != f"engine-proof-{marker.get('engine_source_sha256')}"
        or SHA256_RE.fullmatch(str(marker.get("proof_sha256"))) is None
        or SHA256_RE.fullmatch(str(marker.get("engine_source_sha256"))) is None
    ):
        raise ReuseError("restored Engine marker is invalid")
    metadata = _artifact(
        _gh_json(
            f"repos/{REPOSITORY}/actions/artifacts/{marker['proof_artifact_id']}",
            token=token,
        ),
        name=marker["proof_artifact_name"],
    )
    if metadata["digest"] != marker["proof_artifact_digest"]:
        raise ReuseError("restored Engine proof artifact digest differs")
    with tempfile.TemporaryDirectory(prefix="letsinfer-engine-proof-") as temporary:
        proof_root = pathlib.Path(temporary) / "proof"
        _download_artifact(metadata["id"], proof_root, token=token)
        _verify_attestations(proof_root, token=token)
        audit = read_object(raw / "candidate-audit.json")
        proof = _validate_proof(
            proof_root,
            audit=audit,
            contract_sha256=builder_contract(trusted_root),
        )
        _verify_finalizer(metadata, proof, token=token)
        if (
            marker["proof_sha256"] != sha256_file(proof_root / "engine-proof.json")
            or marker["engine_source_sha256"] != proof["engine_source_sha256"]
            or proof["engine_archive_sha256"] != sha256_file(raw / "engine.oci.tar")
            or proof["engine_archive_bytes"] != (raw / "engine.oci.tar").stat().st_size
            or proof["inventory_sha256"]
            != sha256_file(raw / "engine-inventory.json")
        ):
            raise ReuseError("restored Engine output differs from its attested proof")
    marker_path.unlink()
    return proof


def _write_github_output(path: pathlib.Path | None, **values: Any) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    contract = commands.add_parser("contract")
    contract.add_argument("--root", type=pathlib.Path, required=True)
    contract.add_argument("--github-output", type=pathlib.Path)
    restore_command = commands.add_parser("restore")
    restore_command.add_argument("--root", type=pathlib.Path, required=True)
    restore_command.add_argument("--raw", type=pathlib.Path, required=True)
    restore_command.add_argument("--github-output", type=pathlib.Path)
    verify = commands.add_parser("verify-restored")
    verify.add_argument("--root", type=pathlib.Path, required=True)
    verify.add_argument("--raw", type=pathlib.Path, required=True)
    create = commands.add_parser("create-proof")
    create.add_argument("--root", type=pathlib.Path, required=True)
    create.add_argument("--raw", type=pathlib.Path, required=True)
    create.add_argument("--bundle", type=pathlib.Path, required=True)
    create.add_argument("--output", type=pathlib.Path, required=True)
    create.add_argument("--bundle-artifact-id", type=int, required=True)
    create.add_argument("--bundle-artifact-digest", required=True)
    create.add_argument("--bundle-artifact-name", required=True)
    create.add_argument("--finalizer-run-id", type=int, required=True)
    create.add_argument("--finalizer-workflow-sha", required=True)
    arguments = parser.parse_args()
    token = os.environ.get("GH_TOKEN", "")
    if arguments.command == "contract":
        value = builder_contract(arguments.root)
        _write_github_output(arguments.github_output, engine_build_contract_sha256=value)
        print(value)
    elif arguments.command == "restore":
        if not token:
            raise ReuseError("GitHub token is unavailable")
        reused = restore(trusted_root=arguments.root, raw=arguments.raw, token=token)
        _write_github_output(arguments.github_output, reused=reused)
        print(canonical_bytes({"reused": reused}).decode(), end="")
    elif arguments.command == "verify-restored":
        if not token:
            raise ReuseError("GitHub token is unavailable")
        value = verify_restored(
            trusted_root=arguments.root, raw=arguments.raw, token=token
        )
        print(canonical_bytes({"reused": value is not None}).decode(), end="")
    else:
        value = create_proof(
            trusted_root=arguments.root,
            raw=arguments.raw,
            bundle_root=arguments.bundle,
            output=arguments.output,
            bundle_artifact_id=arguments.bundle_artifact_id,
            bundle_artifact_digest=arguments.bundle_artifact_digest,
            bundle_artifact_name=arguments.bundle_artifact_name,
            finalizer_run_id=arguments.finalizer_run_id,
            finalizer_workflow_sha=arguments.finalizer_workflow_sha,
        )
        print(canonical_bytes(value).decode(), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ReuseError,
        engine_sbom.SbomError,
        oci_layout.LayoutError,
        verifier_bundle.BundleError,
    ) as error:
        raise SystemExit(f"FATAL: {error}")
