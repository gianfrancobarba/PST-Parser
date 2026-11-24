# ruff: noqa: D103
"""Tests of the sample flow and data-quality tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_flow.py")
    spec = importlib.util.spec_from_file_location("results_flow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _cell(task: str, condition: str, outcome: str, reason: str) -> dict:
    return {
        "task_id": task,
        "condition": condition,
        "replicate_id": 0,
        "outcome": outcome,
        "reason": reason,
    }


def test_correctness_flow_counts_every_stage() -> None:
    ledger = [
        _cell("a", "T", "pass", "pass"),
        _cell("b", "T", "fail", "no_code"),
        _cell("a", "H-auto", "fail", "test_exception"),
    ]
    details = [
        {**_cell("a", "T", "pass", "pass"), "flaky": True, "finish_reason": "eos"},
        {**_cell("b", "T", "fail", "no_code"), "flaky": False, "finish_reason": "length"},
        {**_cell("a", "H-auto", "fail", "test_exception"), "flaky": False, "finish_reason": "eos"},
    ]
    end_to_end = [
        _cell("a", "H-auto", "fail", "test_exception"),
        _cell("b", "H-auto", "fail", "parser_failure:content_gap"),
    ]
    rows = {row["condition"]: row for row in tool.correctness_flow(ledger, details, end_to_end, 2)}
    t = rows["T"]
    assert t["planned"] == 2 and t["generated"] == 2 and t["truncated"] == 1
    assert t["no_code"] == 1 and t["pass"] == 1 and t["flaky"] == 1
    assert t["reasons"] == {"pass": 1, "no_code": 1}
    h = rows["H-auto"]
    assert h["planned"] == 2 and h["parser_failures"] == {"content_gap": 1} and h["generated"] == 1


def test_correctness_flow_refuses_details_that_disagree_with_the_ledger() -> None:
    ledger = [_cell("a", "T", "pass", "pass")]
    details = [
        {**_cell("a", "T", "fail", "test_exception"), "flaky": False, "finish_reason": "eos"}
    ]
    with pytest.raises(ValueError, match="disagree"):
        tool.correctness_flow(ledger, details, [], 1)


def test_security_flow_counts_states_and_extraction() -> None:
    details = [
        {
            "condition": "T",
            "state": "V",
            "extraction_status": "ok",
            "analysed": True,
            "usable": True,
        },
        {
            "condition": "T",
            "state": "I",
            "extraction_status": "no_code",
            "analysed": False,
            "usable": False,
        },
        {
            "condition": "H-gold",
            "state": "N",
            "extraction_status": "ok",
            "analysed": True,
            "usable": True,
        },
    ]
    rows = {r["condition"]: r for r in tool.security_flow(details, "general")}
    assert rows["T"]["programs"] == 2 and rows["T"]["analysed"] == 1
    assert rows["T"]["states"] == {"V": 1, "N": 0, "I": 1, "U": 0}
    assert rows["T"]["extraction"] == {"ok": 1, "no_code": 1}
    assert rows["H-gold"]["usable"] == 1


def test_token_summary() -> None:
    journal = [
        {"condition": "T", "input_tokens": 10, "output_tokens": 100, "finish_reason": "eos"},
        {"condition": "T", "input_tokens": 30, "output_tokens": 300, "finish_reason": "eos"},
        {"condition": "T", "input_tokens": 20, "output_tokens": 200, "finish_reason": "length"},
    ]
    row = tool.token_summary(journal)[0]
    assert row["n"] == 3 and row["input_median"] == 20 and row["input_max"] == 30
    assert row["output_mean"] == pytest.approx(200) and row["finish"] == {"eos": 2, "length": 1}


def test_t_variants_check() -> None:
    variants = [
        {"task_id": "a", "condition": "T", "text": "x", "prompt_hash": "h", "source_hash": "h"},
        {
            "task_id": "a",
            "condition": "H-gold",
            "text": "{}",
            "prompt_hash": "k",
            "source_hash": "h",
        },
    ]
    journal = [{"task_id": "a", "condition": "T", "replicate_id": 0, "text": "x"}]
    assert tool.t_variants_check(variants, journal) == {
        "t_variants": 1,
        "prompt_equals_source": 1,
        "journal_rows": 1,
        "journal_text_equals_variant": 1,
    }
