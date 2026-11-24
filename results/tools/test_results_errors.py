# ruff: noqa: D103
"""Tests of the error analysis tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_errors.py")
    spec = importlib.util.spec_from_file_location("results_errors", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _cells(rows: list[tuple[str, str, str]]) -> list[dict]:
    return [{"task_id": t, "condition": c, "outcome": o} for t, c, o in rows]


def test_discordant_tasks() -> None:
    cells = _cells(
        [
            ("a", "T", "pass"),
            ("a", "E", "fail"),
            ("b", "T", "fail"),
            ("b", "E", "fail"),
            ("c", "T", "pass"),
        ]
    )
    assert tool.discordant(cells, ("T", "E")) == {"a"}


def test_flip_overlap_against_independence() -> None:
    first = {"a", "b", "c", "d"}
    second = {"c", "d", "e"}
    row = tool.flip_overlap(first, second, 20)
    assert row["first"] == 4 and row["second"] == 3 and row["both"] == 2
    assert row["expected_if_independent"] == pytest.approx(4 * 3 / 20)
    assert row["ratio"] == pytest.approx(2 / 0.6)


def test_flip_overlap_without_flips() -> None:
    row = tool.flip_overlap(set(), {"a"}, 10)
    assert row["both"] == 0 and row["ratio"] is None
