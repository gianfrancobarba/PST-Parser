"""Assembly and persistence of the evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pstparser.config import EvaluationConfig
from pstparser.evaluation import evaluate, write_report
from pstparser.pst import LEAF_PATHS, assemble, serialise_target

CONFIG = EvaluationConfig(ted_failure_penalty=100.0, confusion_top_k=3)


def target(**leaves: list[str]) -> str:
    """Serialise a target carrying the given leaf values."""
    values: dict[str, list[str]] = {path: [] for path in LEAF_PATHS}
    values.update(leaves)
    return serialise_target(assemble(values))


def test_perfect_predictions_score_at_the_top() -> None:
    reference = target(main_instruction=["Summarise the report."])

    report = evaluate([reference], [reference], ["Summarise the report."], CONFIG)

    assert report.total == 1
    assert report.valid == 1
    assert report.json_validity_rate == 1.0
    assert report.mean_tree_edit_distance == 0.0
    assert report.mean_hallucination_rate == 0.0
    assert report.mean_coverage_score == 1.0


def test_invalid_prediction_lowers_validity_but_keeps_the_row() -> None:
    reference = target(main_instruction=["Summarise."])

    report = evaluate(["{broken", reference], [reference, reference], ["Summarise."] * 2, CONFIG)

    assert report.total == 2
    assert report.valid == 1
    assert report.json_validity_rate == 0.5
    assert [detail.valid for detail in report.details] == [False, True]
    assert report.details[0].tree_edit_distance is None
    assert report.details[1].tree_edit_distance == 0.0


def test_details_follow_the_supplied_indices() -> None:
    reference = target(main_instruction=["a"])

    report = evaluate([reference] * 3, [reference] * 3, ["a"] * 3, CONFIG, indices=[7, 3, 11])

    assert [detail.index for detail in report.details] == [7, 3, 11]


def test_details_default_to_positional_indices() -> None:
    reference = target(main_instruction=["a"])

    report = evaluate([reference] * 2, [reference] * 2, ["a"] * 2, CONFIG)

    assert [detail.index for detail in report.details] == [0, 1]


def test_means_are_absent_when_nothing_parses() -> None:
    reference = target(main_instruction=["a"])

    report = evaluate(["{broken"], [reference], ["a"], CONFIG)

    assert report.valid == 0
    assert report.mean_tree_edit_distance is None
    assert report.mean_hallucination_rate is None
    assert report.mean_coverage_score is None
    assert report.field_f1 == {}


def test_a_parseable_object_of_the_wrong_shape_is_not_valid() -> None:
    reference = target(main_instruction=["Summarise."])

    report = evaluate(['{"prompt": {}}'], [reference], ["Summarise."], CONFIG)

    assert report.valid == 0
    assert report.json_validity_rate == 0.0
    assert report.parse_rate == 1.0


def test_details_carry_the_reconstruction_of_each_example() -> None:
    reference = target(main_instruction=["Summarise the report."])
    partial = target(main_instruction=["Summarise"])

    report = evaluate(
        [reference, partial, "{broken"],
        [reference] * 3,
        ["Summarise the report."] * 3,
        CONFIG,
    )

    assert [detail.reconstructed for detail in report.details] == [True, False, None]
    assert [detail.alignment_rate for detail in report.details] == [1.0, 1.0, None]
    assert report.reconstruction.exact_reconstruction_rate == pytest.approx(1 / 3)


def test_misaligned_sequences_are_rejected() -> None:
    reference = target()

    with pytest.raises(ValueError, match="must be aligned"):
        evaluate([reference], [reference, reference], ["a"], CONFIG)


def test_report_serialises_every_metric() -> None:
    reference = target(main_instruction=["Summarise the report."])

    payload = evaluate([reference], [reference], ["Summarise the report."], CONFIG).as_dict()
    restored = json.loads(json.dumps(payload))

    assert set(restored) == {
        "total",
        "valid",
        "json_validity_rate",
        "parse_rate",
        "field_f1",
        "reconstruction",
        "mean_tree_edit_distance",
        "mean_hallucination_rate",
        "mean_coverage_score",
        "confusion",
        "top_confusions",
    }
    assert set(restored["field_f1"]["prompt.main_instruction"]) == {
        "precision",
        "recall",
        "f1",
        "support",
        "predicted",
    }


def test_report_is_written_alongside_its_details(tmp_path: Path) -> None:
    reference = target(main_instruction=["Summarise."])
    report = evaluate([reference, "{broken"], [reference] * 2, ["Summarise."] * 2, CONFIG)

    results_path, details_path = write_report(report, tmp_path / "run")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    details = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert results["total"] == 2
    assert results["valid"] == 1
    assert len(details) == 2
    assert details[1]["valid"] is False
