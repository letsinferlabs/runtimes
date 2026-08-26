#!/usr/bin/env python3
"""Create and validate the canonical Let's Infer candidate README header."""

from __future__ import annotations

import argparse
import json
import pathlib
import re

if __package__:
    from tools import changed_candidates
else:
    import changed_candidates


SITE_URL = "https://letsinfer.ai/"
INSTALL_COMMAND = "curl -fsSL https://letsinfer.ai/install.sh | sh"
VISIBLE_MARKER = f"> **Run this model with [Let's Infer]({SITE_URL}).**"
MODEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")


class ReadmeError(ValueError):
    pass


def launch_block(logical_model: str) -> str:
    if MODEL_RE.fullmatch(logical_model) is None:
        raise ReadmeError("runtime logical_model is invalid")
    fence = chr(96) * 3
    return (
        f"{VISIBLE_MARKER}\n"
        ">\n"
        "> Install Let's Infer first:\n"
        ">\n"
        f"> {fence}sh\n"
        f"> {INSTALL_COMMAND}\n"
        f"> {fence}\n"
        ">\n"
        "> Then install this model:\n"
        ">\n"
        f"> {fence}sh\n"
        f"> letsinfer model install {logical_model}\n"
        f"> {fence}\n"
    )


def validate(readme: str, logical_model: str) -> None:
    expected = launch_block(logical_model) + "\n"
    if not readme.startswith(expected):
        raise ReadmeError(
            "runtime README must begin with the canonical Let's Infer install block "
            f"for logical model {logical_model}"
        )


def prepend(readme: str, logical_model: str) -> str:
    if readme.startswith(VISIBLE_MARKER):
        validate(readme, logical_model)
        return readme
    if VISIBLE_MARKER in readme:
        raise ReadmeError("runtime README contains a misplaced Let's Infer install block")
    return launch_block(logical_model) + "\n" + readme


def read_runtime(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReadmeError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("logical_model"), str):
        raise ReadmeError(f"{path} has no logical_model")
    return value


def process(root: pathlib.Path, candidates: list[str], *, write: bool) -> None:
    known = changed_candidates.candidates(root)
    unknown = set(candidates) - known
    if unknown:
        raise ReadmeError("unknown runtime candidate: " + ", ".join(sorted(unknown)))
    for candidate in candidates:
        directory = root / candidate
        runtime = read_runtime(directory / "runtime.json")
        readme_path = directory / "README.md"
        if readme_path.is_symlink() or not readme_path.is_file():
            raise ReadmeError(f"runtime README is missing: {candidate}")
        try:
            readme = readme_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ReadmeError(f"cannot read {readme_path}: {error}") from error
        if write:
            updated = prepend(readme, runtime["logical_model"])
            if updated != readme:
                readme_path.write_text(updated, encoding="utf-8")
        else:
            validate(readme, runtime["logical_model"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    if arguments.candidate:
        selected = sorted(set(arguments.candidate))
    else:
        if not arguments.base or not arguments.head:
            raise ReadmeError("--base and --head are required without --candidate")
        paths = changed_candidates.git_paths(root, arguments.base, arguments.head)
        selected = changed_candidates.changed(
            root, paths, all_on_shared_change=False
        )
    process(root, selected, write=arguments.write)
    print(json.dumps(selected, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReadmeError, changed_candidates.ChangeError) as error:
        raise SystemExit(f"FATAL: {error}")
