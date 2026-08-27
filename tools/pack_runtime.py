#!/usr/bin/env python3
"""Build one deterministic runtime pack with an exact checked-out Core contract."""

from __future__ import annotations

import argparse
import pathlib
import sys


class PackError(RuntimeError):
    pass


def pack(core_root: pathlib.Path, candidate: pathlib.Path, output: pathlib.Path) -> None:
    core_root = core_root.resolve(strict=True)
    candidate = candidate.resolve(strict=True)
    if not (core_root / "core" / "runtime_packs.py").is_file():
        raise PackError("exact Core runtime-pack contract is unavailable")
    sys.path.insert(0, str(core_root))
    try:
        from core.runtime_packs import RuntimePackError, build_archive
    except ImportError as error:
        raise PackError("cannot import exact Core runtime-pack contract") from error
    try:
        build_archive(candidate, output)
    except RuntimePackError as error:
        raise PackError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise PackError("runtime pack output already exists")
    pack(arguments.core_root, arguments.candidate, arguments.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackError as error:
        raise SystemExit(f"FATAL: {error}")
