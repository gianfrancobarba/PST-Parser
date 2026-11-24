# ruff: noqa: D103
"""Tests of the correctness reading tables (H1, mechanism, concordance, anchor)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
from scipy.stats import binomtest


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_correctness.py")
    spec = importlib.util.spec_from_file_location("results_correctness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def test_odds_ratio_of_the_discordant_pairs_with_the_exact_conditional_interval() -> None:
    result = tool.discordant_odds_ratio(32, 35)
    assert result["odds_ratio"] == pytest.approx(35 / 32)
    low, high = binomtest(35, 67).proportion_ci(method="exact")
    assert result["ci95"][0] == pytest.approx(low / (1 - low))
    assert result["ci95"][1] == pytest.approx(high / (1 - high))
    assert result["ci95"][0] < 1 < result["ci95"][1]


def test_odds_ratio_is_unbounded_without_baseline_only_pairs() -> None:
    result = tool.discordant_odds_ratio(0, 4)
    assert result["odds_ratio"] is None and result["ci95"][1] is None
    assert tool.discordant_odds_ratio(0, 0) == {"odds_ratio": None, "ci95": [None, None]}


def test_risk_ratio_from_the_mcnemar_table() -> None:
    table = {"both_fail": 324, "baseline_only": 32, "treatment_only": 35, "both_pass": 209}
    assert tool.risk_ratio(table) == pytest.approx(244 / 241)


def _cells(rows: list[tuple[str, str, str, str]]) -> list[dict]:
    return [
        {"task_id": t, "condition": c, "replicate_id": 0, "outcome": o, "reason": r}
        for t, c, o, r in rows
    ]


def test_concordance_matrix() -> None:
    cells = _cells(
        [
            ("a", "T", "pass", "pass"),
            ("a", "E", "pass", "pass"),
            ("b", "T", "fail", "test_exception"),
            ("b", "E", "pass", "pass"),
        ]
    )
    matrix = tool.concordance(cells, ("T", "E"))
    assert matrix["T"]["T"] == 1.0 and matrix["T"]["E"] == 0.5 and matrix["E"]["T"] == 0.5
    assert matrix["n"]["T"]["E"] == 2


def test_concordance_uses_only_tasks_present_in_both_arms() -> None:
    cells = _cells(
        [("a", "T", "pass", "pass"), ("a", "H-auto", "pass", "pass"), ("b", "T", "fail", "x")]
    )
    matrix = tool.concordance(cells, ("T", "H-auto"))
    assert matrix["n"]["T"]["H-auto"] == 1 and matrix["T"]["H-auto"] == 1.0


def test_per_protocol_tasks_drop_answers_without_code_in_either_arm() -> None:
    cells = _cells(
        [
            ("a", "T", "pass", "pass"),
            ("a", "H-gold", "fail", "no_code"),
            ("b", "T", "fail", "no_code"),
            ("b", "E", "pass", "pass"),
            ("c", "T", "pass", "pass"),
            ("c", "H-gold", "pass", "pass"),
        ]
    )
    assert tool.per_protocol_tasks(cells, ("T", "H-gold")) == {"c"}
    assert tool.per_protocol_tasks(cells, ("T", "E")) == set()


def test_per_protocol_tasks_drop_parser_failures() -> None:
    cells = _cells(
        [
            ("a", "T", "pass", "pass"),
            ("a", "H-auto", "fail", "parser_failure:content_gap"),
            ("b", "T", "pass", "pass"),
            ("b", "H-auto", "fail", "test_exception"),
        ]
    )
    assert tool.per_protocol_tasks(cells, ("T", "H-auto")) == {"b"}


def test_clopper_pearson() -> None:
    low, high = tool.clopper_pearson(241, 600)
    ref = binomtest(241, 600).proportion_ci(method="exact")
    assert (low, high) == pytest.approx((ref.low, ref.high))


def test_leaderboard_row() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {"model": ["A", "Qwen2.5-Coder-7B-Instruct"], "instruct": [10.0, 40.4], "date": ["x", "y"]}
    )
    assert tool.leaderboard_value(frame, "Qwen2.5-Coder-7B-Instruct") == 40.4
    with pytest.raises(ValueError):
        tool.leaderboard_value(frame, "missing")


def test_format_spread() -> None:
    assert tool.format_spread({"T": 241, "E": 240, "F": 240, "H-gold": 244}, 600) == {
        "min": 240 / 600,
        "max": 244 / 600,
        "range_pp": pytest.approx(100 * 4 / 600),
    }
    assert math.isclose(tool.format_spread({"T": 1}, 2)["range_pp"], 0.0)
