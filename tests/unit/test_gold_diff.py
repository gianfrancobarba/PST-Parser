"""Prespecified diff between the rule drafts and the reviewed gold (plan of 18/11/2026, P-07)."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.annotations import (
    BCB_RULE,
    AnnotationRow,
    propose,
    proposed_units,
    write_annotations,
)
from pstparser.empirical.benchmarks import SourceRecord, write_sources
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import SpanAnnotation
from pstparser.empirical.golddiff import DIFF_VERSION, gold_diff

CODE = "```\nimport os\ndef task_func(x):\n```"
TEXT = (
    "Compute the sum of a list. It must not modify the input. It must not sort the list.\n"
    "The function should output with:\n    float: The sum.\n"
    "You should write self-contained code starting with:\n" + CODE
)


def source(task: str, text: str = TEXT) -> SourceRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceRecord(
        task_id=task,
        text=text,
        source_sha256=digest,
        original_sha256=digest,
        benchmark="bigcodebench",
        dataset_revision="1" * 40,
        dataset_file_sha256="d" * 64,
        original_id=task,
        family_id=None,
        text_field="instruct_prompt",
        transforms=(),
    )


SOURCE = source("BigCodeBench/1")
UNITS = proposed_units(SOURCE, BCB_RULE)
JUDGED = [index for index, unit in enumerate(UNITS) if unit.kind == "judgement"]


def reviewed(row: AnnotationRow, spans: tuple[SpanAnnotation, ...] | None = None) -> AnnotationRow:
    return dataclasses.replace(
        row,
        annotator="studente",
        reviewer="studente",
        status="reviewed",
        spans=row.spans if spans is None else spans,
    )


def nonspace(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def test_the_example_has_two_judgement_units_after_the_first_sentence() -> None:
    assert len(JUDGED) == 2
    assert [UNITS[index].rule for index in JUDGED] == ["G3", "G3"]


def test_an_unchanged_review_changes_nothing() -> None:
    draft = propose(SOURCE, BCB_RULE)
    report = gold_diff([SOURCE], [draft], [reviewed(draft)], BCB_RULE)
    assert report["diff_version"] == DIFF_VERSION
    assert report["tasks"] == 1 and report["tasks_changed"] == 0
    assert report["characters_changed"] == 0
    assert report["by_rule"]["G3"] == {
        "kind": "judgement",
        "units": 2,
        "units_changed": 0,
        "characters": sum(nonspace(SOURCE.text[UNITS[i].start : UNITS[i].end]) for i in JUDGED),
        "characters_changed": 0,
    }


def test_a_new_leaf_counts_the_characters_and_the_unit_of_its_rule() -> None:
    draft = propose(SOURCE, BCB_RULE)
    index = JUDGED[0]
    spans = list(draft.spans)
    spans[index] = dataclasses.replace(spans[index], path="context.role")
    report = gold_diff([SOURCE], [draft], [reviewed(draft, tuple(spans))], BCB_RULE)
    unit = UNITS[index]
    changed = nonspace(SOURCE.text[unit.start : unit.end])
    assert draft.spans[index].path != "context.role"
    assert report["tasks_changed"] == 1
    assert report["characters_changed"] == changed
    assert report["by_rule"]["G3"]["units_changed"] == 1
    assert report["by_rule"]["G3"]["characters_changed"] == changed
    assert report["transitions"] == {f"{draft.spans[index].path} -> context.role": changed}
    assert report["per_task"]["BigCodeBench/1"]["characters_changed"] == changed


def test_joining_two_units_with_the_same_leaf_changes_units_but_not_characters() -> None:
    draft = propose(SOURCE, BCB_RULE)
    first, second = JUDGED
    spans = list(draft.spans)
    assert spans[first].path == spans[second].path
    joined = dataclasses.replace(
        spans[first],
        end=spans[second].end,
        phrase=SOURCE.text[spans[first].start : spans[second].end],
    )
    spans[first : second + 1] = [joined]
    report = gold_diff([SOURCE], [draft], [reviewed(draft, tuple(spans))], BCB_RULE)
    assert report["characters_changed"] == 0
    assert report["tasks_changed"] == 1
    assert report["by_rule"]["G3"]["units_changed"] == 1


def test_the_reviewer_decided_share_is_the_share_of_judgement_characters() -> None:
    draft = propose(SOURCE, BCB_RULE)
    report = gold_diff([SOURCE], [draft], [reviewed(draft)], BCB_RULE)
    judged = sum(nonspace(SOURCE.text[UNITS[i].start : UNITS[i].end]) for i in JUDGED)
    assert report["characters"] == nonspace(SOURCE.text)
    assert report["reviewer_decided_characters"] == judged
    assert report["reviewer_decided_share"] == pytest.approx(judged / nonspace(SOURCE.text))


def test_a_draft_that_is_not_the_rule_proposal_or_a_gold_still_draft_is_refused() -> None:
    draft = propose(SOURCE, BCB_RULE)
    moved = list(draft.spans)
    moved[JUDGED[0]] = dataclasses.replace(moved[JUDGED[0]], path="context.role")
    edited = dataclasses.replace(draft, spans=tuple(moved))
    with pytest.raises(ValueError, match="not the proposal"):
        gold_diff([SOURCE], [edited], [reviewed(draft)], BCB_RULE)
    with pytest.raises(ValueError, match="still a draft"):
        gold_diff([SOURCE], [draft], [draft], BCB_RULE)


def test_the_gold_may_cover_a_subset_of_the_drafts_but_not_other_tasks() -> None:
    other = source("BigCodeBench/2")
    drafts = [propose(SOURCE, BCB_RULE), propose(other, BCB_RULE)]
    report = gold_diff([SOURCE, other], drafts, [reviewed(drafts[0])], BCB_RULE)
    assert report["tasks"] == 1
    with pytest.raises(ValueError, match="without a draft"):
        gold_diff([SOURCE, other], drafts[:1], [reviewed(drafts[1])], BCB_RULE)


def test_the_command_writes_the_report_and_its_manifest(tmp_path: Path) -> None:
    sources = tmp_path / "in" / "sources.jsonl"
    write_sources([SOURCE], sources)
    draft = propose(SOURCE, BCB_RULE)
    write_annotations([draft], tmp_path / "in" / "draft.jsonl")
    write_annotations([reviewed(draft)], tmp_path / "in" / "gold.jsonl")
    out = tmp_path / "diff" / "gold-diff.json"
    result = CliRunner().invoke(
        app,
        [
            "gold-diff",
            str(sources),
            str(tmp_path / "in" / "draft.jsonl"),
            str(tmp_path / "in" / "gold.jsonl"),
            "--rule",
            BCB_RULE,
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text("utf-8"))["tasks_changed"] == 0
    assert (out.parent / "empirical-gold-diff_manifest.json").is_file()
