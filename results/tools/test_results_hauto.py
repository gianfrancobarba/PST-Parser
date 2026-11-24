# ruff: noqa: D103
"""Tests of the unified H-auto table (r3, attach conversion, r4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_hauto.py")
    spec = importlib.util.spec_from_file_location("results_hauto", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _report(midp: float, ci90: list[float]) -> dict:
    return {
        "conditions": ["T", "H-auto"],
        "included_tasks": 600,
        "mcnemar_table": {
            "both_fail": 300,
            "baseline_only": 30,
            "treatment_only": 32,
            "both_pass": 238,
        },
        "delta": 2 / 600,
        "tango_ci": [-0.02, 0.03],
        "tango_ci_90": ci90,
        "midp_p": midp,
    }


def test_row_reads_e1_to_e4_at_three_margins() -> None:
    row = tool.row(_report(0.8, [-0.02, 0.025]))
    assert (row["n"], row["b"], row["c"]) == (600, 30, 32)
    assert row["reading"] == {"3pp": "E3", "5pp": "E3", "10pp": "E3"}
    wide = tool.row(_report(0.8, [-0.04, 0.045]))
    assert wide["reading"] == {"3pp": "E4", "5pp": "E3", "10pp": "E3"}
    different = tool.row(_report(0.01, [-0.07, -0.02]))
    assert different["reading"]["5pp"] == "E1" and different["reading"]["10pp"] == "E2"


def test_row_without_a_test_has_no_reading() -> None:
    report = _report(0.5, [-0.01, 0.01])
    report["midp_p"] = None
    assert tool.row(report)["reading"] is None


def test_decomposition_rows_keep_the_identity() -> None:
    level = {
        "level": "valid",
        "tasks": 100,
        "reported": "estimate",
        "contrasts": {
            "delta_auto": {"delta": -0.03, "ci95": [-0.1, 0.0], "counts": {}},
            "delta_gold": {"delta": 0.01, "ci95": [-0.05, 0.06], "counts": {}},
            "delta_par": {"delta": -0.04, "ci95": [-0.09, 0.0], "counts": {}},
        },
    }
    row = tool.decomposition_row("validity", level)
    assert row["delta_auto"] == pytest.approx(row["delta_gold"] + row["delta_par"])
    assert row["grouping"] == "validity" and row["tasks"] == 100
