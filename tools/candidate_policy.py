#!/usr/bin/env python3
"""Classify and audit one runtime candidate without executing candidate code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

if __package__:
    from tools import changed_candidates, generate_manifest, readme_onboarding
else:
    import changed_candidates
    import generate_manifest
    import readme_onboarding


ENGINE_INPUT_DIRECTORIES = frozenset(
    {"adapter", "engine", "image", "kernels", "patches", "scripts"}
)
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "id_rsa",
        "id_ed25519",
        "credentials.json",
        "service-account.json",
    }
)
FORBIDDEN_SUFFIXES = (
    ".gguf",
    ".img",
    ".iso",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
)
GENERATED_DIRECTORIES = frozenset(
    {".git", "__pycache__", "build", "dist", "node_modules", "target"}
)
PRIVATE_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
)
MAX_CANDIDATE_FILES = 10_000
MAX_CANDIDATE_BYTES = 1 << 30
MAX_SINGLE_FILE_BYTES = 128 << 20
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
REFERENCE_RE = re.compile(
    r"(?P<repository>ghcr\.io/letsinferlabs/[a-z0-9][a-z0-9._/-]*)"
    r"@sha256:[0-9a-f]{64}"
)
NEW_ENGINE_REPOSITORY = "ghcr.io/letsinferlabs/engine-images"
NEW_RUNTIME_REPOSITORY = "ghcr.io/letsinferlabs/runtime-artifacts"
OFFICIAL_ENGINE_RE = re.compile(
    r"ghcr\.io/letsinferlabs/(?:engines/[a-z0-9][a-z0-9._/-]*|engine-images)"
    r"@sha256:[0-9a-f]{64}"
)


class CandidatePolicyError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def engine_source_sha256(candidate: str, records: Sequence[Mapping[str, Any]]) -> str:
    engine_tree = {
        "schema_version": 1,
        "candidate": candidate,
        "files": [
            dict(record)
            for record in records
            if pathlib.PurePosixPath(str(record.get("path", ""))).parts
            and pathlib.PurePosixPath(str(record["path"])).parts[0]
            in ENGINE_INPUT_DIRECTORIES
        ],
    }
    return hashlib.sha256(canonical_bytes(engine_tree)).hexdigest()


def _tracked_paths(root: pathlib.Path, candidate: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", candidate],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise CandidatePolicyError("cannot enumerate tracked candidate source")
    values = result.stdout.split(b"\0")
    try:
        paths = [value.decode("utf-8") for value in values if value]
    except UnicodeDecodeError as error:
        raise CandidatePolicyError("candidate contains a non-UTF-8 Git path") from error
    return sorted(paths)


def classify_paths(
    *, candidate: str, paths: Sequence[str], candidate_is_new: bool, root: pathlib.Path
) -> str:
    prefix = candidate + "/"
    relative = [value.removeprefix(prefix) for value in paths if value.startswith(prefix)]
    if not relative:
        raise CandidatePolicyError("pull request does not change the selected candidate")
    engine_change = any(
        pathlib.PurePosixPath(value).parts
        and pathlib.PurePosixPath(value).parts[0] in ENGINE_INPUT_DIRECTORIES
        for value in relative
    )
    if candidate_is_new:
        directory = root / candidate
        engine_change = engine_change or any(
            (directory / name).exists() for name in ENGINE_INPUT_DIRECTORIES
        )
    if not engine_change:
        return "reuse-engine"
    try:
        runtime = generate_manifest.read_object(root / candidate / "runtime.json")
        distribution = generate_manifest.engine_distribution(runtime)
    except generate_manifest.ManifestError:
        distribution = {}
    return (
        "build-native-engine"
        if isinstance(distribution, Mapping)
        and distribution.get("kind")
        in {"native-archive", "python-standalone", "embedded-application"}
        else "build-engine"
    )


def classify(
    root: pathlib.Path, *, base: str, head: str, candidate: str | None = None
) -> tuple[str, str, list[str]]:
    paths = changed_candidates.git_paths(root, base, head)
    selected = changed_candidates.changed(root, paths, all_on_shared_change=False)
    if candidate is not None:
        if candidate not in selected:
            raise CandidatePolicyError("requested candidate is not changed by this pull request")
        selected = [candidate]
    if len(selected) != 1:
        rendered = ", ".join(selected) if selected else "none"
        raise CandidatePolicyError(
            f"verifier artifacts require exactly one changed candidate (found {rendered})"
        )
    selected_candidate = selected[0]
    existed = subprocess.run(
        ["git", "cat-file", "-e", f"{base}:{selected_candidate}/runtime.json"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    mode = classify_paths(
        candidate=selected_candidate,
        paths=paths,
        candidate_is_new=not existed,
        root=root,
    )
    return selected_candidate, mode, paths


def _manifest_at(root: pathlib.Path, base: str | None) -> dict[str, Any]:
    if base is None:
        return generate_manifest.read_object(root / "manifest.json")
    if re.fullmatch(r"[0-9a-f]{40}", base) is None:
        raise CandidatePolicyError("repository resolution base commit is invalid")
    result = subprocess.run(
        ["git", "show", f"{base}:manifest.json"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise CandidatePolicyError("cannot read the base manifest for repository resolution")
    try:
        value = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidatePolicyError("base manifest is invalid") from error
    if not isinstance(value, dict):
        raise CandidatePolicyError("base manifest must contain an object")
    return value


def _candidate_release(
    root: pathlib.Path, candidate: str, base: str | None = None
) -> Mapping[str, Any] | None:
    manifest = _manifest_at(root, base)
    matches: list[Mapping[str, Any]] = []
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise CandidatePolicyError("manifest model map is invalid")
    for model in models.values():
        targets = model.get("targets") if isinstance(model, Mapping) else None
        if not isinstance(targets, Mapping):
            continue
        for target in targets.values():
            candidates = target.get("candidates") if isinstance(target, Mapping) else None
            record = candidates.get(candidate) if isinstance(candidates, Mapping) else None
            if isinstance(record, Mapping):
                matches.append(record)
    if not matches:
        return None
    if len(matches) != 1:
        raise CandidatePolicyError("candidate appears multiple times in the generated manifest")
    record = matches[0]
    latest = record.get("latest")
    releases = record.get("releases")
    release = releases.get(latest) if isinstance(latest, str) and isinstance(releases, Mapping) else None
    if not isinstance(release, Mapping):
        raise CandidatePolicyError("existing candidate release is unavailable")
    return release


def runtime_repository(root: pathlib.Path, candidate: str, base: str | None = None) -> str:
    """Keep an existing public package; route a new candidate to the shared package."""

    release = _candidate_release(root, candidate, base)
    if release is None:
        return NEW_RUNTIME_REPOSITORY
    source = release.get("source") if isinstance(release, Mapping) else None
    match = REFERENCE_RE.fullmatch(str(source))
    if match is None or (
        not match["repository"].startswith("ghcr.io/letsinferlabs/runtimes/")
        and match["repository"] != NEW_RUNTIME_REPOSITORY
    ):
        raise CandidatePolicyError("existing candidate runtime repository is not official")
    return match["repository"]


def engine_publication(
    root: pathlib.Path, candidate: str, base: str | None = None
) -> tuple[str, str | None]:
    """Reuse an existing Engine package so its immutable base blobs remain local."""

    release = _candidate_release(root, candidate, base)
    if release is None:
        return NEW_ENGINE_REPOSITORY, None
    distribution = release.get("engine_distribution")
    reference = (
        distribution.get("reference")
        if isinstance(distribution, Mapping)
        else release.get("engine_oci")
    )
    match = REFERENCE_RE.fullmatch(str(reference))
    if match is None or OFFICIAL_ENGINE_RE.fullmatch(str(reference)) is None:
        raise CandidatePolicyError("existing candidate Engine repository is not official")
    return match["repository"], str(reference)


def publication_repositories(
    root: pathlib.Path, candidate: str, base: str | None = None
) -> dict[str, str]:
    engine_repository, existing_reference = engine_publication(root, candidate, base)
    result = {
        "candidate": candidate,
        "runtime_repository": runtime_repository(root, candidate, base),
        "engine_repository": engine_repository,
    }
    if existing_reference is not None:
        result["engine_existing_reference"] = existing_reference
    return result


def _candidate_files(root: pathlib.Path, candidate: str) -> list[pathlib.Path]:
    directory = root / candidate
    if directory.is_symlink() or not directory.is_dir():
        raise CandidatePolicyError("candidate must be a regular directory")
    paths: list[pathlib.Path] = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if GENERATED_DIRECTORIES.intersection(relative.parts):
            raise CandidatePolicyError(
                f"candidate contains generated directory content: {relative}"
            )
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CandidatePolicyError(f"candidate source cannot contain symlinks: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise CandidatePolicyError(
                f"candidate source contains a special file: {relative}"
            )
        paths.append(path)
    return paths


def _dockerfile_bases(path: pathlib.Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise CandidatePolicyError(f"cannot read Engine Dockerfile: {error}") from error
    found = False
    aliases: set[str] = set()
    pattern = re.compile(
        r"^FROM(?:\s+--platform=\S+)?\s+(\S+)(?:\s+AS\s+([A-Za-z0-9._-]+))?\s*$",
        re.IGNORECASE,
    )
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or not line.upper().startswith("FROM "):
            continue
        found = True
        match = pattern.fullmatch(line)
        if match is None:
            raise CandidatePolicyError("Engine Dockerfile has an unsupported FROM instruction")
        image, alias = match.groups()
        if image != "scratch" and image.lower() not in aliases and "@sha256:" not in image:
            raise CandidatePolicyError(f"Engine Dockerfile has a mutable base image: {image}")
        if alias:
            aliases.add(alias.lower())
    if not found:
        raise CandidatePolicyError("Engine Dockerfile must contain a FROM instruction")


def audit_candidate(root: pathlib.Path, candidate: str, mode: str) -> dict[str, Any]:
    if mode not in {"reuse-engine", "build-engine", "build-native-engine"}:
        raise CandidatePolicyError("candidate Engine mode is invalid")
    directory = root / candidate
    runtime_path = directory / "runtime.json"
    release_path = directory / "release.json"
    readme_path = directory / "README.md"
    for path in (runtime_path, release_path, readme_path):
        if path.is_symlink() or not path.is_file():
            raise CandidatePolicyError(f"candidate is missing {path.name}")
    runtime = generate_manifest.read_object(runtime_path)
    if runtime.get("id") != candidate:
        raise CandidatePolicyError("candidate directory and runtime ID differ")
    release = generate_manifest.read_object(release_path)
    if not isinstance(release.get("license"), str) or not release["license"]:
        raise CandidatePolicyError("candidate release license is missing")
    readme_onboarding.validate(
        readme_path.read_text(encoding="utf-8"), str(runtime.get("logical_model"))
    )
    engine = runtime.get("engine")
    if not isinstance(engine, dict):
        raise CandidatePolicyError("runtime Engine contract is missing")
    try:
        distribution = generate_manifest.engine_distribution(runtime)
    except generate_manifest.ManifestError as error:
        raise CandidatePolicyError(str(error)) from error
    if distribution["kind"] == "oci-container":
        reference = distribution.get("reference")
        immutable_id = distribution.get("immutable_id")
        if OFFICIAL_ENGINE_RE.fullmatch(str(reference)) is None:
            raise CandidatePolicyError(
                "runtime must pin an official Engine OCI manifest digest"
            )
        if SHA256_RE.fullmatch(str(immutable_id)) is None:
            raise CandidatePolicyError(
                "runtime must pin an Engine configuration digest"
            )
        payload_id = distribution.get("payload_id")
        if payload_id is not None and SHA256_RE.fullmatch(str(payload_id)) is None:
            raise CandidatePolicyError("runtime Engine payload identity is invalid")
        if mode == "build-native-engine":
            raise CandidatePolicyError("OCI candidate cannot use build-native-engine")
    elif mode != "build-native-engine":
        raise CandidatePolicyError(
            "native Engine candidate must use build-native-engine"
        )
    else:
        reference = (
            distribution.get("archive", {}).get("url")
            if isinstance(distribution.get("archive"), Mapping)
            else distribution.get("bundle_id")
            or distribution.get("python", {}).get("archive", {}).get("url")
        )
        immutable_id = distribution.get("payload_id")
    protocol = engine.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("version") != 2:
        raise CandidatePolicyError("runtime must use Engine protocol 2")

    if mode == "build-engine":
        for path in (
            directory / "adapter" / "engine-adapter",
            directory / "image" / "Dockerfile",
            directory / "LICENSE",
        ):
            if path.is_symlink() or not path.is_file():
                raise CandidatePolicyError(
                    f"build-engine candidate is missing {path.relative_to(directory)}"
                )
        _dockerfile_bases(directory / "image" / "Dockerfile")
    elif mode == "build-native-engine":
        for path in (
            directory / "adapter" / "engine-adapter",
            directory / "LICENSE",
        ):
            if path.is_symlink() or not path.is_file():
                raise CandidatePolicyError(
                    f"build-native-engine candidate is missing {path.relative_to(directory)}"
                )
        if (directory / "image" / "Dockerfile").exists():
            raise CandidatePolicyError(
                "build-native-engine candidate cannot contain image/Dockerfile"
            )

    paths = _candidate_files(root, candidate)
    if len(paths) > MAX_CANDIDATE_FILES:
        raise CandidatePolicyError("candidate exceeds the source file-count limit")
    total = 0
    engine_total = 0
    records: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        if size > MAX_SINGLE_FILE_BYTES:
            raise CandidatePolicyError(f"candidate source file is too large: {relative}")
        total += size
        if total > MAX_CANDIDATE_BYTES:
            raise CandidatePolicyError("candidate exceeds the 1 GiB source limit")
        if pathlib.PurePosixPath(relative).parts[0] in ENGINE_INPUT_DIRECTORIES:
            engine_total += size
        lowered = relative.lower()
        if path.name in FORBIDDEN_NAMES or lowered.endswith(FORBIDDEN_SUFFIXES):
            raise CandidatePolicyError(f"candidate contains generated or secret material: {relative}")
        content = path.read_bytes()
        if any(marker in content for marker in PRIVATE_MARKERS):
            raise CandidatePolicyError(f"candidate contains private-key material: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": size,
                "mode": 0o755 if os.access(path, os.X_OK) else 0o644,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    tree = {
        "schema_version": 1,
        "candidate": candidate,
        "files": records,
    }
    # Git tree order and recursive filesystem order differ when a candidate
    # mixes files and nested directories. The inventory is a set identity;
    # normalize both traversals before comparing it.
    tracked = sorted(_tracked_paths(root, candidate))
    actual = sorted(f"{candidate}/{record['path']}" for record in records)
    if not tracked:
        raise CandidatePolicyError("candidate source is not tracked by Git")
    if actual != tracked:
        extra = sorted(set(actual) - set(tracked))
        missing = sorted(set(tracked) - set(actual))
        raise CandidatePolicyError(
            f"candidate working tree differs from tracked source (extra={extra}, missing={missing})"
        )
    return {
        "schema_version": 1,
        "candidate": candidate,
        "mode": mode,
        "file_count": len(records),
        "candidate_bytes": total,
        "engine_source_bytes": engine_total,
        "largest_files": sorted(
            ({"path": item["path"], "bytes": item["bytes"]} for item in records),
            key=lambda item: (-item["bytes"], item["path"]),
        )[:10],
        "candidate_tree_sha256": hashlib.sha256(canonical_bytes(tree)).hexdigest(),
        "engine_source_sha256": engine_source_sha256(candidate, records),
        "engine_reference": reference,
        "engine_config_digest": immutable_id,
        "engine_payload_digest": payload_id,
        "engine_base_reference": (
            oci.get("base") if isinstance(oci, dict) else None
        ),
        "target_platform": runtime.get("target", {}).get("platform"),
    }


def _write_output(path: pathlib.Path, values: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("classify", "audit", "repositories"))
    parser.add_argument(
        "--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--candidate")
    parser.add_argument(
        "--mode",
        choices=("reuse-engine", "build-engine", "build-native-engine"),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--github-output", type=pathlib.Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    if arguments.command == "classify":
        if not arguments.base:
            raise CandidatePolicyError("classify requires --base")
        candidate, mode, _paths = classify(
            root,
            base=arguments.base,
            head=arguments.head,
            candidate=arguments.candidate,
        )
        value = {"candidate": candidate, "mode": mode}
    elif arguments.command == "audit":
        if not arguments.candidate or not arguments.mode:
            raise CandidatePolicyError("audit requires --candidate and --mode")
        value = audit_candidate(root, arguments.candidate, arguments.mode)
    else:
        if not arguments.candidate:
            raise CandidatePolicyError("repositories requires --candidate")
        value = publication_repositories(root, arguments.candidate, arguments.base)
    data = canonical_bytes(value)
    if arguments.output:
        arguments.output.write_bytes(data)
    if arguments.github_output:
        _write_output(arguments.github_output, value)
    print(data.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        CandidatePolicyError,
        changed_candidates.ChangeError,
        generate_manifest.ManifestError,
        readme_onboarding.ReadmeError,
    ) as error:
        raise SystemExit(f"FATAL: {error}")
