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
    field_f1_scores,
    hallucination_rates,
    json_validity_rate,
    reconstruction_scores,
    token_confusion,
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


def rounded(value: float | None, digits: int = 6) -> float | None:
    """Round a score, leaving an undefined one undefined."""
    return None if value is None else round(value, digits)


@pytest.mark.parametrize("name", case_names())
def test_case_scores_are_stable(name: str, data_regression: Any) -> None:
    case = load_case(name)
    predictions = [case["prediction"]]
    references = [case["reference"]]
    prompts = [case["prompt"]]

    scores = {
        "json_validity_rate": rounded(json_validity_rate(predictions)),
        "field_f1": {
            field: {
                "precision": rounded(score.precision),
                "recall": rounded(score.recall),
                "f1": rounded(score.f1),
                "support": score.support,
                "predicted": score.predicted,
            }
            for field, score in sorted(field_f1_scores(predictions, references).items())
        },
        "tree_edit_distance": [
            rounded(value) for value in tree_edit_distances(predictions, references, PENALTY)
        ],
        "hallucination_rate": [
            rounded(value) for value in hallucination_rates(predictions, prompts)
        ],
        "coverage_score": [rounded(value) for value in coverage_scores(predictions, prompts)],
        "confusion": token_confusion(predictions, references, prompts).as_dict(),
        "reconstruction": reconstruction_scores(predictions, prompts).as_dict(),
    }

    data_regression.check(scores, basename=f"scores_{name}")


def test_identical_trees_score_perfectly() -> None:
    case = load_case("001_perfect_match")

    assert json_validity_rate([case["prediction"]]) == 1.0
    assert tree_edit_distances([case["prediction"]], [case["reference"]], PENALTY) == [0.0]
    assert all(
        score.f1 == 1.0
        for score in field_f1_scores([case["prediction"]], [case["reference"]]).values()
        if score.support or score.predicted
    )
    assert hallucination_rates([case["prediction"]], [case["prompt"]]) == [0.0]
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [1.0]


def test_a_field_without_support_is_reported_unscored() -> None:
    case = load_case("001_perfect_match")

    scores = field_f1_scores([case["prediction"]], [case["reference"]])

    paths = scores["prompt.reasoning.paths"]
    assert paths.support == 0
    assert paths.predicted == 0
    assert paths.f1 is None


def test_invalid_prediction_is_charged_the_penalty() -> None:
    case = load_case("004_invalid_json")

    assert json_validity_rate([case["prediction"]]) == 0.0
    assert tree_edit_distances([case["prediction"]], [case["reference"]], PENALTY) == [PENALTY]
    # Faithfulness is a share of what was emitted, so it has no value here;
    # coverage does, and it is zero, because nothing was placed.
    assert hallucination_rates([case["prediction"]], [case["prompt"]]) == [None]
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [0.0]


def test_an_empty_prediction_has_no_faithfulness_to_report() -> None:
    case = load_case("009_empty_prediction")

    assert json_validity_rate([case["prediction"]]) == 1.0
    assert hallucination_rates([case["prediction"]], [case["prompt"]]) == [None]
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [0.0]


def test_a_well_formed_object_of_the_wrong_shape_is_not_valid() -> None:
    case = load_case("015_schema_violation")

    assert json_validity_rate([case["prediction"]]) == 0.0


def test_hallucinated_words_are_counted() -> None:
    case = load_case("005_hallucinated_text")

    rate = hallucination_rates([case["prediction"]], [case["prompt"]])[0]

    assert rate is not None
    assert rate > 0.0
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [1.0]


def test_unaccounted_prompt_lowers_coverage() -> None:
    case = load_case("006_partial_coverage")

    assert coverage_scores([case["prediction"]], [case["prompt"]])[0] < 1.0
    assert hallucination_rates([case["prediction"]], [case["prompt"]]) == [0.0]


def test_a_repeated_token_placed_once_still_counts_as_covered() -> None:
    case = load_case("014_repeated_tokens")

    # "log" and "the" each occur twice in the prompt and are placed once. The
    # prompt holds five distinct tokens and only "warning" goes unaccounted for,
    # which counting occurrences instead would have charged twice over.
    assert coverage_scores([case["prediction"]], [case["prompt"]]) == [4 / 5]


def test_misplaced_segment_appears_in_the_confusion_matrix() -> None:
    case = load_case("003_swapped_fields")

    confusion = token_confusion([case["prediction"]], [case["reference"]], [case["prompt"]])
    misassignments = confusion.top_confusions(5)

    # "Do not use markdown." belongs to the restrictive node and was placed in
    # the presentational one: four tokens, not one misplaced value.
    assert ("prompt.context.constrains", "prompt.context.format", 4) in misassignments


def test_the_whole_prompt_is_recovered_when_nothing_is_left_out() -> None:
    case = load_case("012_exact_reconstruction")

    scores = reconstruction_scores([case["prediction"]], [case["prompt"]])

    assert scores.exact_reconstruction_rate == 1.0
    assert scores.alignment_rate == 1.0
    assert scores.ambiguity_rate == 0.0


def test_the_taxonomy_order_does_not_constrain_the_reconstruction() -> None:
    case = load_case("013_reordered_reconstruction")

    scores = reconstruction_scores([case["prediction"]], [case["prompt"]])

    assert scores.alignment_rate == 1.0
    assert scores.exact_reconstruction_rate == 1.0


def test_a_repeated_phrase_is_reported_as_ambiguous() -> None:
    case = load_case("010_repeated_phrase")

    scores = reconstruction_scores([case["prediction"]], [case["prompt"]])

    assert scores.alignment_rate == 1.0
    assert scores.ambiguity_rate > 0.0
    assert scores.exact_reconstruction_rate == 1.0


def test_a_phrase_absent_from_the_prompt_breaks_the_reconstruction() -> None:
    case = load_case("011_phrase_not_found")

    scores = reconstruction_scores([case["prediction"]], [case["prompt"]])

    assert scores.located == 0
    assert scores.exact_reconstruction_rate == 0.0


def test_report_aggregates_the_whole_case_set() -> None:
    cases = [load_case(name) for name in case_names()]
    unusable = 2  # one prediction does not parse, one does not fit the schema

    report = evaluate(
        predictions=[case["prediction"] for case in cases],
        references=[case["reference"] for case in cases],
        prompts=[case["prompt"] for case in cases],
        config=EvaluationConfig(ted_failure_penalty=PENALTY),
    )

    assert report.total == len(cases)
    assert report.valid == len(cases) - unusable
    assert report.json_validity_rate == pytest.approx((len(cases) - unusable) / len(cases))
    assert report.parse_rate == pytest.approx((len(cases) - 1) / len(cases))
    assert report.mean_tree_edit_distance is not None
    assert len(report.details) == len(cases)
    assert [detail.valid for detail in report.details].count(False) == unusable
