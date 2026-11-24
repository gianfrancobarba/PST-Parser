"""Score the parser against the reviewed or frozen gold, not the drafts.

parse-prompts stores, beside each prediction, the target built from the spans of
the parse sources; for D-01 and the prefix those sources carried the rule drafts.
retarget-predictions replaces the target alone with the gold tree of the given
annotations, after checking task, text and coverage, and leaves every raw output
untouched; auto-annotate, score and the gate then read the retargeted file.
"""

import dataclasses
import json

import pytest
from test_gold_diff import SOURCE, UNITS, reviewed, source
from typer.testing import CliRunner

from pstparser.data.predictions import PredictionRecord, load_predictions, write_predictions
from pstparser.empirical.annotations import BCB_RULE, propose, write_annotations
from pstparser.empirical.benchmarks import write_sources
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import SpanAnnotation
from pstparser.empirical.retarget import retarget_predictions
from pstparser.inference.parse_prompts import gold_target

TASKS = ["BigCodeBench/1", "BigCodeBench/2"]
SOURCES = [SOURCE, source("BigCodeBench/2")]


def gold_rows():
    rows = []
    for record in SOURCES:
        draft = propose(record, BCB_RULE)
        spans = list(draft.spans)
        judged = next(i for i, unit in enumerate(UNITS) if unit.kind == "judgement")
        spans[judged] = dataclasses.replace(spans[judged], path="context.format")
        rows.append(reviewed(draft, tuple(spans)))
    return rows


def predictions(rows) -> list[PredictionRecord]:
    drafts = [propose(record, BCB_RULE) for record in SOURCES]
    return [
        PredictionRecord(
            index,
            record.text,
            gold_target([dataclasses.asdict(s) for s in draft.spans]),
            f"raw output {index}",
            input_tokens=10,
            output_tokens=5,
            finish_reason="eos",
        )
        for index, (record, draft) in enumerate(zip(SOURCES, drafts, strict=True))
    ]


def as_dicts(spans: tuple[SpanAnnotation, ...]) -> list[dict]:
    return [dataclasses.asdict(span) for span in spans]


def test_only_the_target_changes_and_it_is_the_gold_tree() -> None:
    gold = gold_rows()
    before = predictions(gold)
    after = retarget_predictions(before, TASKS, SOURCES, gold, require_status="reviewed")
    for old, new, row in zip(before, after, gold, strict=True):
        assert dataclasses.replace(new, target=old.target) == old
        assert new.target == gold_target(as_dicts(row.spans))
    assert any(old.target != new.target for old, new in zip(before, after, strict=True))


def test_a_text_that_differs_from_the_gold_source_is_refused() -> None:
    gold = gold_rows()
    altered = [dataclasses.replace(predictions(gold)[0], prompt="other"), predictions(gold)[1]]
    with pytest.raises(ValueError, match="text"):
        retarget_predictions(altered, TASKS, SOURCES, gold, require_status="reviewed")


def test_gold_that_does_not_cover_the_parsed_tasks_is_refused() -> None:
    gold = gold_rows()[:1]
    with pytest.raises(ValueError, match="cover"):
        retarget_predictions(
            predictions(gold_rows()), TASKS, SOURCES, gold, require_status="reviewed"
        )


def test_gold_in_another_state_is_refused() -> None:
    with pytest.raises(ValueError, match="frozen"):
        retarget_predictions(
            predictions(gold_rows()), TASKS, SOURCES, gold_rows(), require_status="frozen"
        )


def test_the_command_writes_the_file_and_records_both_hashes(tmp_path) -> None:
    gold = gold_rows()
    write_sources(SOURCES, tmp_path / "sources.jsonl")
    write_annotations(gold, tmp_path / "gold.jsonl")
    write_predictions(predictions(gold), tmp_path / "run" / "predictions.jsonl")
    (tmp_path / "run" / "id-map.json").write_text(
        json.dumps({"format": "id-map-v1", "sources_sha256": "0" * 64, "tasks": TASKS}),
        encoding="utf-8",
    )
    out = tmp_path / "retargeted" / "predictions.jsonl"
    arguments = [
        "retarget-predictions",
        str(tmp_path / "run" / "predictions.jsonl"),
        str(tmp_path / "run" / "id-map.json"),
        str(tmp_path / "sources.jsonl"),
        str(tmp_path / "gold.jsonl"),
        "--require-status",
        "reviewed",
        "--out",
        str(out),
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.exception
    written = load_predictions(out)
    assert [r.prediction for r in written] == ["raw output 0", "raw output 1"]
    manifest = json.loads((out.parent / "empirical-retarget_manifest.json").read_text("utf-8"))
    assert manifest["require_status"] == "reviewed"
    assert manifest["input_predictions_sha256"] != manifest["predictions_sha256"]
    assert CliRunner().invoke(app, arguments).exit_code != 0  # never overwritten
