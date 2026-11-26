# ruff: noqa: D103
"""Tests of the task and representation features associated with outcome changes."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_rq2_features.py")
    spec = importlib.util.spec_from_file_location("results_rq2_features", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def test_category_shares_weigh_each_leaf_by_its_characters() -> None:
    spans = [
        {"path": "main_instruction", "start": 0, "end": 30},
        {"path": "context.constraints", "start": 31, "end": 61},
        {"path": "context.format", "start": 62, "end": 82},
        {"path": "examples", "start": 83, "end": 103},
    ]
    shares = tool.category_shares(spans)
    assert shares["share_main_instruction"] == pytest.approx(0.3)
    assert shares["share_constraints"] == pytest.approx(0.3)
    assert shares["share_format"] == pytest.approx(0.2)
    assert shares["share_data"] == 0.0
    assert shares["has_examples"] == 1


def test_category_shares_without_examples() -> None:
    shares = tool.category_shares([{"path": "context.data", "start": 0, "end": 10}])
    assert shares["share_data"] == 1.0 and shares["has_examples"] == 0


def test_solution_lines_count_non_blank_non_comment_lines() -> None:
    code = "    x = 1\n\n    # comment\n    return x\n"
    assert tool.solution_lines(code) == 2


def test_test_methods_count_the_test_functions() -> None:
    source = "class T:\n    def test_a(self):\n        pass\n    def helper(self):\n" * 1
    source += "    def test_b(self):\n        pass\n"
    assert tool.test_methods(source) == 2


def test_stability_classes_over_four_forms() -> None:
    stable = {"T": "pass", "H-gold": "pass", "E": "pass", "F": "pass"}
    only_pst = {"T": "fail", "H-gold": "pass", "E": "fail", "F": "fail"}
    shared = {"T": "pass", "H-gold": "fail", "E": "fail", "F": "pass"}
    assert tool.stability(stable) == {"unstable": 0, "pst_change": 0, "pst_only": 0}
    assert tool.stability(only_pst) == {"unstable": 1, "pst_change": 1, "pst_only": 1}
    assert tool.stability(shared) == {"unstable": 1, "pst_change": 1, "pst_only": 0}


def test_standardize_has_mean_zero_and_unit_sd() -> None:
    values = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 6.0}
    z = tool.standardize(values)
    assert sum(z.values()) == pytest.approx(0.0)
    sd = math.sqrt(sum(v * v for v in z.values()) / (len(z) - 1))
    assert sd == pytest.approx(1.0)


def test_binary_association_matches_the_two_by_two_odds_ratio() -> None:
    # 2 x 2 table: x = 1 -> 8 yes, 2 no; x = 0 -> 3 yes, 7 no; MLE OR = (8*7)/(2*3).
    rows = [(1.0, 1)] * 8 + [(1.0, 0)] * 2 + [(0.0, 1)] * 3 + [(0.0, 0)] * 7
    result = tool.association([x for x, _ in rows], [y for _, y in rows])
    assert result["read"] is True
    assert result["odds_ratio"] == pytest.approx(56 / 6, rel=1e-6)
    se = math.sqrt(1 / 8 + 1 / 2 + 1 / 3 + 1 / 7)
    assert result["or_ci95"][0] == pytest.approx(56 / 6 * math.exp(-1.959963984540054 * se))


def test_association_reports_no_variation() -> None:
    result = tool.association([1.0, 1.0, 1.0], [0, 1, 0])
    assert result["read"] is False and result["reason"] == "no variation"


def test_cross_strata_counts_wins_and_losses_per_cell() -> None:
    pairs = [
        {"task_id": "a", "winner": "H-gold"},
        {"task_id": "b", "winner": "T"},
        {"task_id": "c", "winner": "T"},
    ]
    first = {"a": "T1", "b": "T1", "c": "T2", "d": "T3"}
    second = {"a": "T3", "b": "T3", "c": "T3", "d": "T1"}
    cells = tool.cross_strata(pairs, first, second, "H-gold")
    by_cell = {(c["first"], c["second"]): c for c in cells}
    assert by_cell[("T1", "T3")]["pst_wins"] == 1 and by_cell[("T1", "T3")]["pst_losses"] == 1
    assert by_cell[("T1", "T3")]["tasks"] == 2
    assert by_cell[("T3", "T1")]["tasks"] == 1 and by_cell[("T3", "T1")]["pst_wins"] == 0
    assert len(cells) == 9


def test_response_groups_split_by_paired_outcome() -> None:
    outcome = {"a": ("pass", "fail"), "b": ("fail", "pass"), "c": ("pass", "pass")}
    lines = {("a", "T"): 10, ("a", "H-gold"): 6, ("b", "T"): 8, ("b", "H-gold"): 9}
    lines |= {("c", "T"): 5, ("c", "H-gold"): 5}
    groups = tool.response_groups(outcome, lines, "T", "H-gold")
    assert groups["pst_loses"] == [-4]
    assert groups["pst_wins"] == [1]
    assert groups["both_pass"] == [0]
    assert groups["both_fail"] == []


def test_response_groups_skip_cells_without_a_value() -> None:
    outcome = {"a": ("pass", "fail")}
    groups = tool.response_groups(outcome, {("a", "T"): 3, ("a", "H-gold"): None}, "T", "H-gold")
    assert groups["pst_loses"] == []


def test_summary_gives_mean_median_and_a_seeded_bootstrap_interval() -> None:
    summary = tool.summary([1.0, 2.0, 3.0, 4.0, 10.0])
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(4.0) and summary["median"] == 3.0
    low, high = summary["mean_ci95"]
    assert low < 4.0 < high
    assert tool.summary([1.0, 2.0, 3.0, 4.0, 10.0]) == summary
    assert tool.summary([])["n"] == 0
