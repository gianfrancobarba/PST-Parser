# ruff: noqa: D103
"""Tests of the decomposition of Δ_auto by parser quality, on synthetic data."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pstparser.data.predictions import PredictionRecord
from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    write_outcomes,
)
from pstparser.evaluation.metrics.field_f1 import field_f1_scores


def _tool() -> ModuleType:
    path = Path(__file__).with_name("hauto_decomposition.py")
    spec = importlib.util.spec_from_file_location("hauto_decomposition", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()

N = 90
TASKS = [f"BigCodeBench/{i}" for i in range(N)]
INVALID = set(TASKS[:3])
GAPS = set(TASKS[3:12])
S_PARSE = [t for t in TASKS if t not in INVALID | GAPS]


def _tree(main: str, data: str = "") -> str:
    context = {"data": [data]} if data else {}
    return json.dumps({"prompt": {"main_instruction": [main], "context": context}})


def _predictions() -> list[PredictionRecord]:
    rng = random.Random(3)
    rows = []
    for index, task in enumerate(TASKS):
        target = _tree("load the file and return rows", "a csv path")
        if task in INVALID:
            prediction = "{not json"
        else:
            words = ["load", "the", "file", "and", "return", "rows"]
            kept = " ".join(w for w in words if rng.random() < 0.3 + 0.7 * index / N)
            prediction = _tree(kept or "nothing", "a csv path" if rng.random() < 0.5 else "")
        rows.append(PredictionRecord(index, f"prompt {task}", target, prediction))
    return rows


def _score(rows: list[PredictionRecord]) -> dict:
    scores = field_f1_scores([r.prediction for r in rows], [r.target for r in rows])
    return {"field_f1": {name: value.as_dict() for name, value in scores.items()}}


def _failures() -> dict[str, str]:
    return {**{t: "invalid_output" for t in INVALID}, **{t: "content_gap" for t in GAPS}}


def _design() -> AnalysisDesign:
    return AnalysisDesign(
        analysis_id="final-bcb-end-to-end",
        endpoint="functional",
        tasks=tuple(TaskSpec(task_id=t) for t in TASKS),
        replicate_ids=(0,),
        conditions=("T", "H-auto"),
    )


def _ledgers(seed: int = 5, missing: str | None = None) -> tuple[list, list]:
    rng = random.Random(seed)
    e2e, core = [], []
    for task in TASKS:
        t = rng.random() < 0.5
        gold = rng.random() < 0.6
        auto = False if task not in S_PARSE else rng.random() < 0.55
        outcome = "pass" if t else "fail"
        e2e.append(OutcomeRecord("final-bcb-end-to-end", "functional", task, "T", 0, outcome))
        core.append(OutcomeRecord("final-bcb-core", "functional", task, "T", 0, outcome))
        h_outcome = "missing" if task == missing else ("pass" if gold else "fail")
        reason = "harness_error" if task == missing else ""
        core.append(
            OutcomeRecord("final-bcb-core", "functional", task, "H-gold", 0, h_outcome, reason)
        )
        reason = "" if task in S_PARSE else "parser_failure:content_gap"
        e2e.append(
            OutcomeRecord(
                "final-bcb-end-to-end",
                "functional",
                task,
                "H-auto",
                0,
                "pass" if auto else "fail",
                reason,
            )
        )
    return e2e, core


def test_task_quality_is_the_gate_macro_f1_on_one_task() -> None:
    rows = _predictions()
    quality = tool.task_quality(rows, TASKS, _failures(), set(S_PARSE), _score(rows))
    assert [quality[t]["valid"] for t in TASKS[:4]] == [False, False, False, True]
    assert quality[TASKS[0]]["f1"] is None
    one = rows[50]
    single = field_f1_scores([one.prediction], [one.target])
    leaves = [v for v in single.values() if v.support > 0]
    expected = sum(v.f1 or 0.0 for v in leaves) / len(leaves)
    assert quality[TASKS[50]]["f1"] == pytest.approx(expected)
    assert quality[TASKS[50]]["s_parse"] and not quality[TASKS[5]]["s_parse"]
    assert quality[TASKS[5]]["cause"] == "content_gap"


def test_task_quality_refuses_predictions_that_are_not_the_scored_ones() -> None:
    rows = _predictions()
    score = _score(rows)
    score["field_f1"]["prompt.main_instruction"]["f1"] += 0.01
    with pytest.raises(ValueError, match="score"):
        tool.task_quality(rows, TASKS, _failures(), set(S_PARSE), score)


def test_combine_takes_h_gold_from_the_core_and_checks_t() -> None:
    e2e, core = _ledgers()
    combined = tool.combine(e2e, core, "decomposition")
    assert {r.condition for r in combined} == {"T", "H-gold", "H-auto"}
    assert {r.analysis_id for r in combined} == {"decomposition"}
    assert len(combined) == 3 * N
    broken = [
        r
        if r.condition != "T" or r.task_id != TASKS[20]
        else OutcomeRecord(
            r.analysis_id, r.endpoint, r.task_id, "T", 0, "fail" if r.outcome == "pass" else "pass"
        )
        for r in core
    ]
    with pytest.raises(ValueError, match="T"):
        tool.combine(e2e, broken, "decomposition")


def test_decompose_holds_the_identity_in_every_level() -> None:
    rows = _predictions()
    quality = tool.task_quality(rows, TASKS, _failures(), set(S_PARSE), _score(rows))
    e2e, core = _ledgers()
    report = tool.decompose(_design(), tool.combine(e2e, core, "d"), quality)
    assert set(report) == {"all", "validity", "strict_reconstruction", "f1_tercile_s_parse"}
    overall = report["all"]["levels"][0]
    deltas = {name: overall["contrasts"][name]["delta"] for name in tool.CONTRASTS}
    assert deltas["delta_auto"] == pytest.approx(deltas["delta_gold"] + deltas["delta_par"])
    validity = {row["level"]: row for row in report["validity"]["levels"]}
    assert validity["invalid"]["tasks"] == 3
    assert validity["invalid"]["reported"] == "counts only"
    assert validity["invalid"]["contrasts"]["delta_auto"]["delta"] is None
    terciles = report["f1_tercile_s_parse"]
    assert sum(row["tasks"] for row in terciles["levels"]) == len(S_PARSE)
    assert {row["level"] for row in terciles["levels"]} == {"T1", "T2", "T3"}


def test_decompose_drops_tasks_incomplete_in_any_arm() -> None:
    rows = _predictions()
    quality = tool.task_quality(rows, TASKS, _failures(), set(S_PARSE), _score(rows))
    e2e, core = _ledgers(missing=TASKS[40])
    report = tool.decompose(_design(), tool.combine(e2e, core, "d"), quality)
    overall = report["all"]["levels"][0]
    assert overall["tasks"] == N - 1
    assert overall["incomplete_tasks"] == [TASKS[40]]
    planned = {overall["contrasts"][name]["included_tasks"] for name in tool.CONTRASTS}
    assert planned == {N - 1}


def test_run_writes_the_reports_once(tmp_path: Path) -> None:
    rows = _predictions()
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "index": r.index,
                    "prompt": r.prompt,
                    "target": r.target,
                    "prediction": r.prediction,
                }
            )
            + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    id_map = tmp_path / "id-map.json"
    id_map.write_text(json.dumps({"tasks": TASKS}), encoding="utf-8")
    score = tmp_path / "score.json"
    score.write_text(json.dumps(_score(rows)), encoding="utf-8")
    failures = tmp_path / "failures.jsonl"
    failures.write_text(
        "".join(json.dumps({"task_id": t, "cause": c}) + "\n" for t, c in _failures().items()),
        encoding="utf-8",
    )
    s_parse = tmp_path / "s-parse.json"
    s_parse.write_text(json.dumps({"tasks": S_PARSE}), encoding="utf-8")
    design = tmp_path / "design.json"
    design.write_text(
        json.dumps(
            {
                "analysis_id": "final-bcb-end-to-end",
                "endpoint": "functional",
                "tasks": [{"task_id": t} for t in TASKS],
                "replicate_ids": [0],
                "conditions": ["T", "H-auto"],
            }
        ),
        encoding="utf-8",
    )
    e2e, core = _ledgers()
    write_outcomes(e2e, tmp_path / "e2e.jsonl")
    write_outcomes(core, tmp_path / "core.jsonl")
    out = tmp_path / "out"
    args = [
        str(design),
        str(tmp_path / "e2e.jsonl"),
        str(tmp_path / "core.jsonl"),
        "--s-parse",
        str(s_parse),
        "--failures",
        str(failures),
        "--predictions",
        str(predictions),
        "--id-map",
        str(id_map),
        "--score",
        str(score),
        "--out",
        str(out),
    ]
    tool.main(args)
    report = json.loads((out / "decomposition.json").read_text("utf-8"))
    assert report["version"] == tool.VERSION
    lines = (out / "parser-quality.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == N
    with pytest.raises(FileExistsError):
        tool.main(args)
