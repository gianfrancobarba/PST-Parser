# ruff: noqa: D103
"""Tests of the longitudinal table of the parser versions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_parser.py")
    spec = importlib.util.spec_from_file_location("results_parser", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _score(path: Path, field_f1: dict, **extra: float) -> Path:
    payload = {
        "total": 10,
        "valid": 9,
        "json_validity_rate": 0.9,
        "mean_coverage_score": 0.99,
        "mean_hallucination_rate": 0.01,
        "field_f1": field_f1,
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_score_reads_structured_leaves(tmp_path: Path) -> None:
    path = _score(
        tmp_path / "r.json",
        {
            "prompt.context.data": {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 6},
            "prompt.reasoning.paths": {"precision": None, "recall": None, "f1": None, "support": 0},
        },
    )
    score = tool.load_score(path)
    assert score["total"] == 10 and score["valid"] == 9
    assert score["validity"] == 0.9 and score["coverage"] == 0.99
    assert score["leaves"]["context.data"]["f1"] == pytest.approx(2 / 3)
    assert score["leaves"]["context.data"]["support"] == 6
    assert score["leaves"]["reasoning.paths"]["f1"] is None


def test_load_score_reads_the_old_float_format_and_renames_constrains(tmp_path: Path) -> None:
    path = _score(tmp_path / "r.json", {"prompt.context.constrains": 0.48})
    score = tool.load_score(path)
    assert score["leaves"] == {
        "context.constraints": {
            "f1": 0.48,
            "precision": None,
            "recall": None,
            "support": None,
            "predicted": None,
        }
    }


def test_row_f1_leaves_out_the_leaves_without_support() -> None:
    leaves = {
        "a": {"f1": 0.5, "support": 4, "predicted": 6},
        "b": {"f1": 0.0, "support": 0, "predicted": 12},
        "c": {"f1": 0.7, "support": None, "predicted": None},
    }
    f1, unsupported = tool.row_f1(leaves)
    assert f1 == {"a": 0.5, "b": None, "c": 0.7}
    assert unsupported == {"b": 12}


def test_macro_f1_is_the_mean_over_leaves_with_support() -> None:
    leaves = {
        "a": {"f1": 0.5, "support": 4},
        "b": {"f1": 1.0, "support": 2},
        "c": {"f1": None, "support": 0},
    }
    assert tool.macro_f1(leaves) == pytest.approx(0.75)
    assert tool.macro_f1({"c": {"f1": None, "support": 0}}) is None


def test_same_leaves_detects_a_difference(tmp_path: Path) -> None:
    leaf = {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 6}
    a = tool.load_score(_score(tmp_path / "a.json", {"prompt.examples": leaf}))
    b = tool.load_score(_score(tmp_path / "b.json", {"prompt.examples": {**leaf, "f1": 0.6}}))
    assert tool.same_leaves(a, a) == []
    assert tool.same_leaves(a, b) == ["examples: f1 0.6666666666666666 != 0.6"]


def test_documented_rows_carry_their_source() -> None:
    for row in tool.DOCUMENTED:
        assert row["source"] and row["n"] > 0
        assert all(0 <= v <= 1 for v in row["f1"].values())
    scorziello = next(r for r in tool.DOCUMENTED if r["id"] == "scorziello")
    assert scorziello["f1"]["context.constraints"] == 0.3453
    assert scorziello["n"] == 109


def test_table_rows_have_one_column_per_leaf() -> None:
    rows = [
        {"id": "x", "f1": {"examples": 1.0}},
        {"id": "y", "f1": {"context.data": 0.5}},
    ]
    columns, table = tool.leaf_table(rows)
    assert columns == ["id", *tool.LEAVES]
    assert table[0][columns.index("examples")] == 1.0
    assert table[1][columns.index("examples")] is None
