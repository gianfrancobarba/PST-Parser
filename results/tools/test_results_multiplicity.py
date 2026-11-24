# ruff: noqa: D103
"""Tests of the multiplicity sensitivity over the exploratory p-values."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from scipy.stats import false_discovery_control


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_multiplicity.py")
    spec = importlib.util.spec_from_file_location("results_multiplicity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def test_benjamini_hochberg_matches_scipy() -> None:
    p = [0.007, 0.268, 0.256, 0.01, 0.04, 0.9, 0.5, 0.03]
    assert tool.benjamini_hochberg(p) == pytest.approx(
        list(false_discovery_control(p, method="bh"))
    )


def test_benjamini_hochberg_keeps_order_and_caps_at_one() -> None:
    adjusted = tool.benjamini_hochberg([0.9, 0.01, 0.95])
    assert adjusted[1] == pytest.approx(0.03) and max(adjusted) <= 1.0


def test_summary_counts() -> None:
    rows = [{"p": 0.007}, {"p": 0.2}, {"p": 0.04}, {"p": 0.6}]
    summary = tool.summary(rows, q=0.05)
    assert summary["contrasts"] == 4 and summary["raw_below_0.05"] == 2
    assert summary["expected_false_at_0.05"] == pytest.approx(0.2)
    assert summary["bh_below_q"] == 1
