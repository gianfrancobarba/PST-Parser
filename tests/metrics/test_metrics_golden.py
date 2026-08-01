"""Locked behaviour of every metric on a curated set of cases.

Each case isolates one way a prediction can differ from its reference. The
expected scores are committed, so an unexplained diff is a defect rather than a
silent change of meaning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pstparser.config import EvaluationConfig
from pstparser.evaluation import evaluate
from pstparser.evaluation.metrics import (
    coverage_scores,
    field_confusion,
    field_f1_scores,
    hallucination_rates,
    json_validity_rate,
    tree_edit_distances,
)

CASES_DIR = Path(__file__).parent / "cases"
PENALTY = 100.0


def case_names() -> list[str]:
    """List the committed cases, in file order."""
    return sorted(path.stem for path in CASES_DIR.glob("*.json"))


def load_case(name: str) -> dict[str, Any]:
    """Read one case and serialise its reference tree."""
    payload = json.loads((CASES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    payload["reference"] = json.dumps(payload["reference"])
    return payload


@pytest.mark.parametrize("name", case_names())
def test_case_scores_are_stable(name: str, data_regression: Any) -> None:
    case = load_case(name)
    predictions = [case["prediction"]]
    references = [case["reference"]]
    prompts = [case["prompt"]]

    scores = {
        "json_validity_rate": round(json_validity_rate(predictions), 6),
        "field_f1": {
            field: round(value, 6)
            for field, value in sorted(field_f1_scores(predictions, references).items())
        },
        "tree_edit_distance": [
            round(value, 6) for value in tree_edit_distances(predictions, references, PENALTY)
        ],
        "hallucination_rate": [
            round(value, 6) for value in hallucination_rates(predictions, prompts)
        ],
        "coverage_score": [round(value, 6) for value in coverage_scores(predictions, prompts)],
        "confusion": field_confusion(predictions, references).as_dict(),
    }

    data_regression.check(scores, basename=f"scores_{name}")


def test_identical_trees_score_perfectly() -> None:
    case = load_case("001_perfect_match")

    assert json_validity_rate([case["prediction"]]) == 1.0
    assert tree_edit_distances([case["prediction"]], [case["reference"]], PENALTY) == [0.0]
    assert all(
        value == 1.0
        for value in field_f1_scores([case["prediction"]], [case["reference"]]).values()
    )
    assert hallucination_rates([case["prediction"]], [case["prompt"]]) == [0.0]
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [1.0]


def test_invalid_prediction_is_charged_the_penalty() -> None:
    case = load_case("004_invalid_json")

    assert json_validity_rate([case["prediction"]]) == 0.0
    assert tree_edit_distances([case["prediction"]], [case["reference"]], PENALTY) == [PENALTY]
    # An unparseable prediction is skipped by the faithfulness metric and scores
    # zero on coverage, so the two are not computed over the same population.
    assert hallucination_rates([case["prediction"]], [case["prompt"]]) == []
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [0.0]


def test_hallucinated_words_are_counted() -> None:
    case = load_case("005_hallucinated_text")

    rate = hallucination_rates([case["prediction"]], [case["prompt"]])[0]

    assert rate > 0.0
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [1.0]


def test_unaccounted_prompt_lowers_coverage() -> None:
    case = load_case("006_partial_coverage")

    assert coverage_scores([case["prediction"]], [case["prompt"]])[0] < 1.0
    assert hallucination_rates([case["prediction"]], [case["prompt"]]) == [0.0]


def test_misplaced_segment_appears_in_the_confusion_matrix() -> None:
    case = load_case("003_swapped_fields")

    confusion = field_confusion([case["prediction"]], [case["reference"]])
    misassignments = confusion.top_confusions(5)

    assert ("prompt.context.constrains", "prompt.context.format", 1) in misassignments


def test_report_aggregates_the_whole_case_set() -> None:
    cases = [load_case(name) for name in case_names()]

    report = evaluate(
        predictions=[case["prediction"] for case in cases],
        references=[case["reference"] for case in cases],
        prompts=[case["prompt"] for case in cases],
        config=EvaluationConfig(ted_failure_penalty=PENALTY),
    )

    assert report.total == len(cases)
    assert report.valid == len(cases) - 1
    assert report.json_validity_rate == pytest.approx((len(cases) - 1) / len(cases))
    assert report.mean_tree_edit_distance is not None
    assert len(report.details) == len(cases)
    assert [detail.valid for detail in report.details].count(False) == 1
