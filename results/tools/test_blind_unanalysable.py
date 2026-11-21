# ruff: noqa: D103
"""Tests of the count of the reviewer's judgments on the unanalysable (U) cases."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pstparser.empirical.auto import clopper_pearson


def _tool() -> ModuleType:
    path = Path(__file__).with_name("blind_unanalysable.py")
    spec = importlib.util.spec_from_file_location("blind_unanalysable", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _key(case: str, condition: str, targeted: str | None) -> dict:
    states = {"general": "N"} if targeted is None else {"general": "N", "targeted": targeted}
    return {"case_id": case, "condition": condition, "states": states, "usable": True}


def _row(case: str, vulnerable: bool | None) -> dict:
    return {
        "case_id": case,
        "scenario_vulnerable": vulnerable,
        "general_alert_true": None,
        "aligned": True,
        "guessed_condition": "unknown",
    }


KEY = [
    _key("a", "T", "U"),
    _key("b", "T", "U"),
    _key("c", "T", "U"),
    _key("d", "H-gold", "U"),
    _key("e", "H-gold", "U"),
    _key("f", "T", "V"),
    _key("g", "H-gold", None),
]
SHEET = [
    _row("a", True),
    _row("b", False),
    _row("c", None),
    _row("d", False),
    _row("e", False),
    _row("f", True),
    _row("g", True),
]


def test_counts_only_the_u_cases_per_condition() -> None:
    report = tool.count_unanalysable(SHEET, KEY)
    t = report["per_condition"]["T"]
    assert (t["cases"], t["yes"], t["no"], t["undecided"]) == (3, 1, 1, 1)
    assert t["share_yes"] == pytest.approx(0.5)
    assert t["share_yes_ci95"] == pytest.approx(list(clopper_pearson(1, 2)))
    h = report["per_condition"]["H-gold"]
    assert (h["cases"], h["yes"], h["no"], h["undecided"]) == (2, 0, 2, 0)
    assert h["share_yes"] == 0.0
    total = report["total"]
    assert (total["cases"], total["yes"], total["no"], total["undecided"]) == (5, 1, 3, 1)


def test_no_decided_case_gives_no_share() -> None:
    report = tool.count_unanalysable([_row("a", None)], [_key("a", "T", "U")])
    cell = report["per_condition"]["T"]
    assert cell["share_yes"] is None and cell["share_yes_ci95"] is None


def test_refuses_a_sheet_that_does_not_match_the_key() -> None:
    with pytest.raises(ValueError, match="does not match"):
        tool.count_unanalysable(SHEET[:-1], KEY)


def test_cli_writes_a_new_report_with_input_hashes(tmp_path: Path) -> None:
    sheet, key, out = tmp_path / "sheet.jsonl", tmp_path / "key.jsonl", tmp_path / "u.json"
    sheet.write_text("".join(json.dumps(r) + "\n" for r in SHEET), encoding="utf-8")
    key.write_text("".join(json.dumps(r) + "\n" for r in KEY), encoding="utf-8")
    tool.main([str(sheet), str(key), "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["total"]["cases"] == 5
    assert set(report["inputs"]) == {"sheet_sha256", "key_sha256"}
    assert out.read_bytes().endswith(b"}\n") and b"\r\n" not in out.read_bytes()
    with pytest.raises(FileExistsError):
        tool.main([str(sheet), str(key), "--out", str(out)])
