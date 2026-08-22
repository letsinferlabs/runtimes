#!/usr/bin/env python3
"""Resolve candidate directories affected by a Git revision range."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess


class ChangeError(RuntimeError):
    pass


def candidates(root: pathlib.Path) -> set[str]:
    return {
        path.parent.name
        for path in root.glob("*/runtime.json")
        if path.is_file() and not path.is_symlink()
    }


def changed(
    root: pathlib.Path,
    paths: list[str],
    *,
    all_on_shared_change: bool = True,
) -> list[str]:
    known = candidates(root)
    selected: set[str] = set()
    shared = False
    for raw in paths:
        path = pathlib.PurePosixPath(raw)
        if not path.parts:
            continue
        if path.parts[0] in known:
            selected.add(path.parts[0])
        elif path.parts[0] in {"tools", "tests", ".github"} or path.name in {
            "manifest.json",
            "README.md",
        }:
            shared = True
    if shared and all_on_shared_change:
        selected = known
    return sorted(selected)


def git_paths(root: pathlib.Path, base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", base, head, "--"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ChangeError(result.stderr.strip() or "git diff failed")
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--github-output", type=pathlib.Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    known = candidates(root)
    if arguments.candidate:
        unknown = set(arguments.candidate) - known
        if unknown:
            raise ChangeError("unknown runtime candidate: " + ", ".join(sorted(unknown)))
        selected = sorted(set(arguments.candidate))
    else:
        if not arguments.base or not arguments.head:
            raise ChangeError("--base and --head are required without --candidate")
        selected = changed(root, git_paths(root, arguments.base, arguments.head))
    document = json.dumps(selected, separators=(",", ":"))
    if arguments.github_output is not None:
        with arguments.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"candidates={document}\n")
            handle.write(f"count={len(selected)}\n")
    print(document)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ChangeError as error:
        raise SystemExit(f"FATAL: {error}")
