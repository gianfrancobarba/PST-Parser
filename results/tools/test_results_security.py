# ruff: noqa: D103
"""Tests of the security reading tables (frozen measures, Δ_I, CWE, S)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pstparser.empirical.analysis import OutcomeRecord
from pstparser.empirical.inference import cluster_interval_at


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_security.py")
    spec = importlib.util.spec_from_file_location("results_security", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _record(outcome: str) -> OutcomeRecord:
    return OutcomeRecord(
        analysis_id="final-lse-sampled-general",
        endpoint="security",
        task_id="LSE-1",
        condition="T",
        replicate_id=1,
        outcome=outcome,
    )


def test_unusable_relabelling_counts_only_state_i() -> None:
    relabelled = [tool.unusable_as_endpoint(_record(s)).outcome for s in ("V", "N", "I", "U")]
    assert relabelled == ["V", "V", "N", "V"]


def test_security_row_reads_on_the_cr1_90_interval() -> None:
    report = {
        "conditions": ["T", "H-gold"],
        "included_tasks": 100,
        "delta": 0.01,
        "cr1_ci": [-0.02, 0.04],
        "cr1_se": 0.015,
        "cr1_df": 34,
        "signflip_p": 0.5,
        "outcome_counts": {},
        "family_sensitivity": None,
        "bootstrap_ci": [-0.02, 0.04],
        "sensitivity": {},
    }
    row = tool.security_row(report)
    expected = cluster_interval_at([-0.02, 0.04], 0.015, 34, 0.90)
    assert row["cr1_ci_90"] == pytest.approx(list(expected))
    assert row["reading_5pp"] == "E3"


def test_cwe_of_a_family() -> None:
    assert tool.cwe("79-INI-1") == "CWE-79"
    assert tool.cwe("1236-XYZ-2") == "CWE-1236"


def test_cwe_table() -> None:
    details = [
        {"family_id": "79-A-1", "condition": "T", "state": "N"},
        {"family_id": "79-A-1", "condition": "T", "state": "V"},
        {"family_id": "79-A-2", "condition": "H-gold", "state": "I"},
        {"family_id": "89-B-1", "condition": "T", "state": "V"},
    ]
    rows = {(r["cwe"], r["condition"]): r for r in tool.cwe_table(details)}
    assert rows[("CWE-79", "T")]["programs"] == 2 and rows[("CWE-79", "T")]["N"] == 1
    assert rows[("CWE-79", "T")]["share_N"] == pytest.approx(0.5)
    assert rows[("CWE-79", "H-gold")]["I"] == 1
    assert rows[("CWE-89", "T")]["V"] == 1
