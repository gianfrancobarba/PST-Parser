"""Integrity check comparing targets against their source prompts."""

from __future__ import annotations

import json

import pytest

from pstparser.data import check_corpus
from pstparser.pst import LEAF_PATHS, ROOT_KEY, assemble, serialise_target


def target_with(**leaves: list[str]) -> str:
    """Serialise a target carrying the given leaf values."""
    values: dict[str, list[str]] = {path: [] for path in LEAF_PATHS}
    values.update(leaves)
    return serialise_target(assemble(values))


def test_complete_annotation_reports_no_issue() -> None:
    report = check_corpus(
        prompts=["Summarise the code."],
        targets=[target_with(main_instruction=["Summarise the code."])],
        min_coverage_ratio=0.90,
    )

    assert report.passed
    assert report.total == 1


def test_whitespace_differences_do_not_count_as_missing() -> None:
    report = check_corpus(
        prompts=["Summarise\n\tthe   code."],
        targets=[target_with(main_instruction=["Summarise the code."])],
        min_coverage_ratio=1.0,
    )

    assert report.passed


def test_short_annotation_is_reported() -> None:
    report = check_corpus(
        prompts=["Explain this long prompt that was barely annotated at all."],
        targets=[target_with(main_instruction=["Explain"])],
        min_coverage_ratio=0.90,
    )

    assert not report.passed
    issue = report.issues[0]
    assert issue.index == 0
    assert issue.kind == "low_coverage"
    assert issue.coverage is not None
    assert issue.coverage < 0.90


def test_annotation_just_above_the_threshold_passes() -> None:
    prompt = "a" * 100
    report = check_corpus(
        prompts=[prompt],
        targets=[target_with(main_instruction=["a" * 90])],
        min_coverage_ratio=0.90,
    )

    assert report.passed


def test_a_segment_the_prompt_does_not_hold_is_reported() -> None:
    # Length alone cannot see this: the annotation is as long as the prompt.
    report = check_corpus(
        prompts=["Summarise the code."],
        targets=[target_with(main_instruction=["Summarise the text."])],
        min_coverage_ratio=0.0,
    )

    issue = report.issues[0]
    assert issue.kind == "absent_segment"
    assert issue.segment == "Summarise the text."


def test_a_segment_claimed_by_two_nodes_is_reported() -> None:
    # The instruction swallows the passage that the restrictive node also holds.
    report = check_corpus(
        prompts=["Summarise the code. Be brief."],
        targets=[
            target_with(
                main_instruction=["Summarise the code. Be brief."],
                **{"context.constraints": ["Be brief."]},
            )
        ],
        min_coverage_ratio=0.0,
    )

    issue = report.issues[0]
    assert issue.kind == "contested_segment"
    assert issue.segment == "Be brief."


def test_a_repeated_passage_is_not_mistaken_for_a_clash() -> None:
    # The passage really does occur twice, so both segments find a home.
    report = check_corpus(
        prompts=["stop. go. stop."],
        targets=[target_with(main_instruction=["stop.", "stop."])],
        min_coverage_ratio=0.0,
    )

    assert report.passed


def test_unparseable_target_is_reported() -> None:
    report = check_corpus(
        prompts=["anything"],
        targets=["{not valid json"],
        min_coverage_ratio=0.90,
    )

    assert report.issues[0].kind == "unparseable_target"


def test_misaligned_sequences_are_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        check_corpus(prompts=["a", "b"], targets=[target_with()], min_coverage_ratio=0.9)


def test_report_is_serialisable() -> None:
    report = check_corpus(
        prompts=["Explain this long prompt that was barely annotated at all."],
        targets=[target_with(main_instruction=["Explain"])],
        min_coverage_ratio=0.90,
    )

    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["total"] == 1
    assert payload["issue_count"] == 1
    assert payload["affected_records"] == 1
    assert payload["issues"][0]["kind"] == "low_coverage"


def test_issue_indices_follow_corpus_order() -> None:
    prompts = ["ok", "a much longer prompt than its annotation", "fine"]
    targets = [
        target_with(main_instruction=["ok"]),
        target_with(main_instruction=["a"]),
        target_with(main_instruction=["fine"]),
    ]

    report = check_corpus(prompts=prompts, targets=targets, min_coverage_ratio=0.90)

    assert [issue.index for issue in report.issues] == [1]
    assert ROOT_KEY in targets[0]
