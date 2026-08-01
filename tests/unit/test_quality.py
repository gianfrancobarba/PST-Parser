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
    assert issue.source_length is not None
    assert issue.reconstructed_length is not None
    assert issue.reconstructed_length < issue.source_length


def test_annotation_just_above_the_threshold_passes() -> None:
    prompt = "a" * 100
    report = check_corpus(
        prompts=[prompt],
        targets=[target_with(main_instruction=["a" * 90])],
        min_coverage_ratio=0.90,
    )

    assert report.passed


def test_unparseable_target_is_reported() -> None:
    report = check_corpus(
        prompts=["anything"],
        targets=["{not valid json"],
        min_coverage_ratio=0.90,
    )

    assert report.issues[0].kind == "unparseable_target"


def test_target_without_root_key_is_reported() -> None:
    report = check_corpus(
        prompts=["anything"],
        targets=[json.dumps({"unexpected": {}})],
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
