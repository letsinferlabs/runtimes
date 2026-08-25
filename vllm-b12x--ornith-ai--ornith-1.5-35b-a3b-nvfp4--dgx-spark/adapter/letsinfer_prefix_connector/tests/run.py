# SPDX-License-Identifier: AGPL-3.0-only
"""Dependency-free build gate for the vLLM connector tests."""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
from types import ModuleType
from typing import Any


class MonkeyPatch:
    """Small subset of pytest's fixture used by this test module."""

    def __init__(self) -> None:
        self._changes: list[tuple[Any, str, Any]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        self._changes.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, original in reversed(self._changes):
            setattr(target, name, original)


def load_tests() -> ModuleType:
    path = pathlib.Path(__file__).with_name("test_connector.py")
    spec = importlib.util.spec_from_file_location("test_connector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load connector tests from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_tests()
    tests = [
        value
        for name, value in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    ]
    for test in tests:
        patch = MonkeyPatch()
        parameters = inspect.signature(test).parameters
        unexpected = set(parameters) - {"monkeypatch"}
        if unexpected:
            raise RuntimeError(
                f"unsupported fixtures for {test.__name__}: "
                f"{', '.join(sorted(unexpected))}"
            )
        try:
            test(**({"monkeypatch": patch} if parameters else {}))
        finally:
            patch.undo()
    print(f"connector tests passed: {len(tests)}")


if __name__ == "__main__":
    main()
