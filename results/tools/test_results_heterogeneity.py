# ruff: noqa: D103
"""Tests of the heterogeneity tables (domains, contamination strata, frozen subgroups)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pstparser.empirical.inference import tango_interval


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_heterogeneity.py")
    spec = importlib.util.spec_from_file_location("results_heterogeneity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def test_domains_put_a_task_in_every_domain_of_its_libraries() -> None:
    mapping = {"numpy": "Computation", "matplotlib": "Visualization", "os": "System"}
    libs = {"a": ["numpy", "matplotlib"], "b": ["os"], "c": ["unknown"], "d": []}
    domains, unmapped = tool.domains(libs, mapping)
    assert domains == {"Computation": {"a"}, "Visualization": {"a"}, "System": {"b"}}
    assert unmapped == {"unknown": 1}


def _cells(rows: list[tuple[str, str, str]]) -> list[dict]:
    return [{"task_id": t, "condition": c, "outcome": o} for t, c, o in rows]


def test_paired_counts_small_level_reports_counts_only() -> None:
    cells = _cells(
        [("a", "T", "pass"), ("a", "H-gold", "fail"), ("b", "T", "fail"), ("b", "H-gold", "pass")]
    )
    row = tool.paired_counts(cells, ("T", "H-gold"), {"a", "b"})
    assert (row["n"], row["b"], row["c"]) == (2, 1, 1)
    assert row["reported"] == "counts only" and row["delta"] is None and row["ci95"] is None


def test_paired_counts_estimate_uses_the_frozen_tango_interval() -> None:
    rows = []
    for i in range(30):
        t = "pass" if i < 12 else "fail"
        h = "pass" if (i < 10 or 12 <= i < 16) else "fail"
        rows += [(f"t{i}", "T", t), (f"t{i}", "H-gold", h)]
    row = tool.paired_counts(_cells(rows), ("T", "H-gold"), {f"t{i}" for i in range(30)})
    assert (row["n"], row["b"], row["c"]) == (30, 2, 4)
    assert row["delta"] == pytest.approx(2 / 30)
    assert row["ci95"] == pytest.approx(list(tango_interval(2, 4, 30)))


def test_contaminated_tasks_are_high_in_either_arm() -> None:
    overlap = [
        {"task_id": "a", "condition": "T", "high_similarity": True},
        {"task_id": "a", "condition": "H-gold", "high_similarity": False},
        {"task_id": "b", "condition": "E", "high_similarity": True},
        {"task_id": "c", "condition": "H-gold", "high_similarity": True},
    ]
    assert tool.contaminated(overlap, ("T", "H-gold")) == {"a", "c"}


def test_logistic_per_tenth_rescales_the_log_odds() -> None:
    term = {"log_odds": -10.0, "se": 4.0, "odds_ratio": math.exp(-10), "or_ci95": [0, 0]}
    scaled = tool.per_step(term, 0.1)
    assert scaled["odds_ratio"] == pytest.approx(math.exp(-1.0))
    assert scaled["or_ci95"][0] == pytest.approx(math.exp(-1.0 - 1.959963984540054 * 0.4))
    assert scaled["or_ci95"][1] == pytest.approx(math.exp(-1.0 + 1.959963984540054 * 0.4))
