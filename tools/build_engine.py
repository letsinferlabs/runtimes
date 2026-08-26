#!/usr/bin/env python3
"""Build one Engine with Let's Infer's canonical BuildKit contract."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

if __package__:
    from tools import candidate_policy, oci_layout, pin_engine
else:
    import candidate_policy
    import oci_layout
    import pin_engine


BUILDKIT_IMAGE = (
    "moby/buildkit@sha256:"
    "28a898719c18a33f4e8000685287fa36fd0dd9560c6440227d3a732d79bb41d8"
)


class BuildError(RuntimeError):
    pass


def read_object(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BuildError(f"{path} must contain an object")
    return value


def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    # stdout is a strict machine-readable channel owned by main(). Send every
    # child command's progress and intermediate document to stderr.
    kwargs.setdefault("stdout", sys.stderr)
    try:
        return subprocess.run(arguments, check=True, **kwargs)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuildError(f"command failed: {' '.join(arguments)}") from error


def build(
    root: pathlib.Path,
    candidate: str,
    output: pathlib.Path,
    *,
    pin: bool,
    inventory_output: pathlib.Path | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    tool_root = pathlib.Path(__file__).resolve().parents[1]
    directory = (root / candidate).resolve(strict=True)
    if directory.parent != root:
        raise BuildError("candidate must be one direct repository child")
    audit = candidate_policy.audit_candidate(root, candidate, "build-engine")
    runtime_path = directory / "runtime.json"
    runtime = read_object(runtime_path)
    oci = runtime["engine"]["oci"]
    base = oci.get("base")
    if not isinstance(base, str):
        raise BuildError("build-engine runtime must pin engine.oci.base")
    repository, _existing = candidate_policy.engine_publication(
        root, candidate, None
    )
    platform = str(audit["target_platform"])
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    builder = os.environ.get("LETSINFER_CANONICAL_BUILDER", "letsinfer-canonical")
    inspected = subprocess.run(
        ["docker", "buildx", "inspect", builder],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspected.returncode != 0:
        run(
            [
                "docker",
                "buildx",
                "create",
                "--name",
                builder,
                "--driver",
                "docker-container",
                "--driver-opt",
                f"image={BUILDKIT_IMAGE}",
            ]
        )
    run(["docker", "buildx", "inspect", builder, "--bootstrap"])
    build_command = [
        "docker",
        "buildx",
        "build",
        str(directory),
        "--builder",
        builder,
        "--file",
        str(directory / "image" / "Dockerfile"),
        "--platform",
        platform,
        "--build-context",
        f"letsinfer-tools={tool_root}",
        "--build-arg",
        "SOURCE_DATE_EPOCH=0",
        "--output",
        (
            "type=oci,dest=-,oci-mediatypes=true,"
            "rewrite-timestamp=true,compression=gzip"
        ),
    ]
    thin_command = [
        sys.executable,
        str(tool_root / "tools" / "oci_layout.py"),
        "thin",
        "--archive",
        "-",
        "--platform",
        platform,
        "--repository",
        repository,
        "--existing-reference",
        base,
        "--output",
        str(output),
    ]
    producer = subprocess.Popen(build_command, stdout=subprocess.PIPE)
    assert producer.stdout is not None
    try:
        run(thin_command, stdin=producer.stdout)
    finally:
        producer.stdout.close()
    if producer.wait() != 0:
        raise BuildError("canonical BuildKit build failed")
    if inventory_output is not None:
        inventory_output = inventory_output.resolve()
        run(
            [
                "docker",
                "buildx",
                "build",
                str(directory),
                "--builder",
                builder,
                "--file",
                str(directory / "image" / "Dockerfile"),
                "--platform",
                platform,
                "--target",
                "letsinfer-engine-inventory",
                "--build-context",
                f"letsinfer-tools={tool_root}",
                "--build-arg",
                "SOURCE_DATE_EPOCH=0",
                "--output",
                f"type=local,dest={inventory_output}",
            ]
        )
    plan = oci_layout.inspect_archive(output, platform)
    plan["reference"] = f"{repository}@{plan['manifest_digest']}"
    if pin:
        pin_engine.pin_runtime(
            runtime_path,
            plan["reference"],
            plan["config_digest"],
            plan["payload_digest"],
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--inventory-output", type=pathlib.Path)
    arguments = parser.parse_args()
    plan = build(
        arguments.root,
        arguments.candidate,
        arguments.output,
        pin=arguments.pin,
        inventory_output=arguments.inventory_output,
    )
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, candidate_policy.CandidatePolicyError, oci_layout.LayoutError, pin_engine.PinError) as error:
        raise SystemExit(f"FATAL: {error}")
