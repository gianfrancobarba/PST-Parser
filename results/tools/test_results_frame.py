# ruff: noqa: D103
"""Tests of the results frame and of the reconciliation with the frozen reports."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    analyze_paired,
    write_outcomes,
)
from pstparser.empirical.tables import Provenance, paired_tables, to_csv


def _tool() -> ModuleType:
    path = Path(__file__).with_name("results_frame.py")
    spec = importlib.util.spec_from_file_location("results_frame", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()

TASKS = [f"BigCodeBench/{i}" for i in range(60)]


def _design() -> AnalysisDesign:
    return AnalysisDesign(
        analysis_id="final-bcb-core",
        endpoint="functional",
        tasks=tuple(TaskSpec(task_id=t) for t in TASKS),
        replicate_ids=(0,),
        conditions=("T", "H-gold"),
    )


def _records(seed: int = 5) -> list[OutcomeRecord]:
    rng = random.Random(seed)
    rows = []
    for task in TASKS:
        for condition in ("T", "H-gold", "E"):
            outcome = "pass" if rng.random() < 0.45 else "fail"
            rows.append(
                OutcomeRecord(
                    analysis_id="final-bcb-core",
                    endpoint="functional",
                    task_id=task,
                    condition=condition,
                    replicate_id=0,
                    outcome=outcome,
                    reason=outcome if outcome == "pass" else "test_exception",
                )
            )
    return rows


def _write_design(path: Path, design: AnalysisDesign) -> None:
    payload = {
        "analysis_id": design.analysis_id,
        "endpoint": design.endpoint,
        "tasks": [{"task_id": t.task_id} for t in design.tasks],
        "replicate_ids": list(design.replicate_ids),
        "conditions": list(design.conditions),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- compare -----------------------------------------------------------------


def test_compare_accepts_equal_and_tolerated_values() -> None:
    frozen = {"delta": 0.005, "ci": [-0.022, 0.032], "name": "x", "none": None}
    fresh = {"delta": 0.005 + 1e-12, "ci": [-0.022, 0.032], "name": "x", "none": None}
    assert tool.compare(frozen, fresh) == []


def test_compare_reports_every_kind_of_difference() -> None:
    frozen = {"delta": 0.005, "ci": [-0.022, 0.032], "name": "x", "gone": 1, "none": None}
    fresh = {"delta": 0.006, "ci": [-0.022], "name": "y", "new": 1, "none": 0.0}
    diffs = tool.compare(frozen, fresh)
    joined = "\n".join(diffs)
    for where in ("delta", "ci", "name", "gone", "new", "none"):
        assert where in joined
    assert len(diffs) == 6


def test_compare_treats_bool_and_int_as_different() -> None:
    assert tool.compare({"flag": True}, {"flag": 1}) != []


# --- report reproduction --------------------------------------------------------


def test_reproduce_report_equals_the_frozen_estimator(tmp_path: Path) -> None:
    design = _design()
    records = _records()
    design_path = tmp_path / "design.json"
    ledger = tmp_path / "outcomes.jsonl"
    _write_design(design_path, design)
    write_outcomes(records, ledger)
    expected = analyze_paired(design, records, ledger_conditions=("E", "H-gold", "T")).as_dict()
    fresh = tool.reproduce_report(design_path, ledger)
    assert tool.compare(json.loads(json.dumps(expected)), fresh) == []


def test_reconcile_report_flags_an_altered_frozen_report(tmp_path: Path) -> None:
    design = _design()
    records = _records()
    design_path = tmp_path / "design.json"
    ledger = tmp_path / "outcomes.jsonl"
    _write_design(design_path, design)
    write_outcomes(records, ledger)
    frozen = json.loads(json.dumps(tool.reproduce_report(design_path, ledger)))
    frozen["delta"] += 0.01
    report = tmp_path / "report.json"
    report.write_text(json.dumps(frozen), encoding="utf-8")
    diffs = tool.reconcile_report(design_path, ledger, report)
    assert len(diffs) == 1 and "delta" in diffs[0]


def test_ledger_conditions_are_declared_only_when_the_ledger_has_more_than_two(
    tmp_path: Path,
) -> None:
    pair = tmp_path / "pair.jsonl"
    write_outcomes([r for r in _records() if r.condition != "E"], pair)
    assert tool.ledger_conditions(pair) is None
    three = tmp_path / "three.jsonl"
    write_outcomes(_records(), three)
    assert tool.ledger_conditions(three) == ("E", "H-gold", "T")


# --- paired table reproduction ------------------------------------------------


def test_compare_table_round_trips_the_frozen_csv(tmp_path: Path) -> None:
    design = _design()
    report = analyze_paired(design, _records(), ledger_conditions=("E", "H-gold", "T")).as_dict()
    reports = {"analysis/bcb-h1/report.json": json.loads(json.dumps(report))}
    table = paired_tables(reports)[0]
    provenance = Provenance(run_id="run", source_digest="0" * 64, inputs=())
    csv_path = tmp_path / "paired.csv"
    csv_path.write_text(to_csv(table, provenance, "paired"), encoding="utf-8", newline="\n")
    assert tool.compare_table(table, csv_path) == []


def test_compare_table_flags_an_altered_cell(tmp_path: Path) -> None:
    design = _design()
    report = analyze_paired(design, _records(), ledger_conditions=("E", "H-gold", "T")).as_dict()
    table = paired_tables({"r": json.loads(json.dumps(report))})[0]
    provenance = Provenance(run_id="run", source_digest="0" * 64, inputs=())
    text = to_csv(table, provenance, "paired")
    lines = text.split("\n")
    header = next(line for line in lines if not line.startswith("#")).split(",")
    data_index = next(i for i, line in enumerate(lines) if line and not line.startswith("#")) + 1
    cells = lines[data_index].split(",")
    cells[header.index("b")] = str(int(cells[header.index("b")]) + 1)
    lines[data_index] = ",".join(cells)
    csv_path = tmp_path / "paired.csv"
    csv_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    diffs = tool.compare_table(table, csv_path)
    assert len(diffs) == 1 and "b" in diffs[0]


# --- task frame ---------------------------------------------------------------


def test_task_frame_joins_outcomes_with_task_attributes() -> None:
    records = [r for r in _records() if r.condition in ("T", "H-gold")]
    attributes = {
        TASKS[0]: {"hard": True, "n_judgement_units": 2},
        TASKS[1]: {"hard": False, "n_judgement_units": 0},
    }
    frame = tool.task_frame(records, attributes)
    assert len(frame) == len(records)
    first = next(r for r in frame if r["task_id"] == TASKS[0] and r["condition"] == "T")
    assert first["hard"] is True and first["n_judgement_units"] == 2
    other = next(r for r in frame if r["task_id"] == TASKS[5])
    assert "hard" not in other
    assert {r["outcome"] for r in frame} <= {"pass", "fail"}


def test_task_frame_joins_cell_attributes_by_task_condition_replicate() -> None:
    records = [r for r in _records() if r.condition in ("T", "H-gold")]
    cells = {(TASKS[2], "H-gold", 0): {"output_tokens": 311, "flaky": False}}
    frame = tool.task_frame(records, {}, cells)
    hit = next(r for r in frame if r["task_id"] == TASKS[2] and r["condition"] == "H-gold")
    miss = next(r for r in frame if r["task_id"] == TASKS[2] and r["condition"] == "T")
    assert hit["output_tokens"] == 311 and hit["flaky"] is False
    assert "output_tokens" not in miss


def test_task_frame_refuses_cell_attributes_without_an_outcome() -> None:
    records = [r for r in _records() if r.condition == "T"]
    with pytest.raises(ValueError, match="no outcome"):
        tool.task_frame(records, {}, {("BigCodeBench/999", "T", 0): {"flaky": False}})


def test_task_frame_refuses_a_duplicate_cell() -> None:
    records = _records()
    with pytest.raises(ValueError, match="duplicate"):
        tool.task_frame([*records, records[0]], {})


def test_task_frame_refuses_an_attribute_that_overwrites_a_ledger_field() -> None:
    with pytest.raises(ValueError, match="overwrite"):
        tool.task_frame(_records(), {TASKS[0]: {"outcome": "pass"}})


def test_parse_libs_reads_the_python_list_of_the_dataset() -> None:
    assert tool.parse_libs("['random', 'itertools']") == ["itertools", "random"]
    assert tool.parse_libs("[]") == []


@pytest.mark.parametrize("text", ["random", "[1, 2]", "__import__('os')", "{'a': 1}"])
def test_parse_libs_refuses_anything_but_a_list_of_names(text: str) -> None:
    with pytest.raises(ValueError):
        tool.parse_libs(text)


def test_sources_cover_the_fourteen_frozen_designs() -> None:
    assert len(tool.SOURCES) == 14
    assert {name for name in tool.SOURCES if name.startswith("lse-")} == {
        "lse-general",
        "lse-general-python",
        "lse-greedy-general",
        "lse-targeted",
        "lse-targeted-fp",
    }
