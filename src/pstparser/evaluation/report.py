"""Scoring a set of predictions and persisting the outcome.

Content metrics are computed on the predictions that are well-formed target
objects. A prediction that is not still counts against the validity rate, and
appears in the per-example detail with no scores.

Locating the phrases is the exception: it is attempted on everything that parses,
because recovering the prompt is a property of the text, not of the shape the
text was wrapped in.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pstparser.config import EvaluationConfig
from pstparser.evaluation.metrics import (
    Confusion,
    FieldScore,
    ReconstructionScores,
    align_predictions,
    coverage_scores,
    field_f1_scores,
    hallucination_rates,
    is_valid,
    json_validity_rate,
    parse_rate,
    score_alignments,
    token_confusion,
    tree_edit_distances,
)

RESULTS_FILE = "results.json"
DETAILS_FILE = "eval_details.jsonl"


@dataclass(frozen=True)
class ExampleScore:
    """Per-example scores, for error analysis.

    Attributes:
        index: Position of the record in the corpus.
        valid: Whether the prediction is a well-formed target object.
        tree_edit_distance: Structural distance from the reference.
        hallucination_rate: Share of predicted tokens absent from the prompt.
        coverage_score: Share of prompt tokens present in the prediction.
        reconstructed: Whether the located phrases give the prompt back.
        alignment_rate: Share of the prediction's phrases found in the prompt.
    """

    index: int
    valid: bool
    tree_edit_distance: float | None = None
    hallucination_rate: float | None = None
    coverage_score: float | None = None
    reconstructed: bool | None = None
    alignment_rate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the scores."""
        return {
            "index": self.index,
            "valid": self.valid,
            "tree_edit_distance": self.tree_edit_distance,
            "hallucination_rate": self.hallucination_rate,
            "coverage_score": self.coverage_score,
            "reconstructed": self.reconstructed,
            "alignment_rate": self.alignment_rate,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate outcome of scoring a set of predictions.

    Attributes:
        total: Number of predictions scored.
        valid: Number that are well-formed target objects.
        json_validity_rate: Share that are well-formed target objects.
        parse_rate: Share that read as JSON, whatever their shape. A gap from
            the validity rate is the share that parsed but was not the object
            the taxonomy declares.
        field_f1: Token agreement per field, with its support.
        confusion: Where each prompt token went against where it belonged.
        reconstruction: How well the prompt is recovered from the located
            phrases.
        mean_tree_edit_distance: Mean structural distance, over valid
            predictions.
        mean_hallucination_rate: Mean share of invented tokens, over the valid
            predictions for which the share is defined.
        mean_coverage_score: Mean share of the prompt accounted for, over valid
            predictions.
        details: Per-example scores, in input order.
    """

    total: int
    valid: int
    json_validity_rate: float
    parse_rate: float
    field_f1: dict[str, FieldScore]
    confusion: Confusion
    reconstruction: ReconstructionScores
    mean_tree_edit_distance: float | None
    mean_hallucination_rate: float | None
    mean_coverage_score: float | None
    details: list[ExampleScore]

    def as_dict(self, top_confusions: int = 5) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report.

        Args:
            top_confusions: Number of misassignments listed alongside the
                matrix.

        Returns:
            A mapping suitable for serialisation.
        """
        return {
            "total": self.total,
            "valid": self.valid,
            "json_validity_rate": self.json_validity_rate,
            "parse_rate": self.parse_rate,
            "field_f1": {field: score.as_dict() for field, score in sorted(self.field_f1.items())},
            "reconstruction": self.reconstruction.as_dict(),
            "mean_tree_edit_distance": self.mean_tree_edit_distance,
            "mean_hallucination_rate": self.mean_hallucination_rate,
            "mean_coverage_score": self.mean_coverage_score,
            "confusion": self.confusion.as_dict(),
            "top_confusions": [
                {"expected": expected, "predicted": predicted, "count": count}
                for expected, predicted, count in self.confusion.top_confusions(top_confusions)
            ],
        }


def evaluate(
    predictions: Sequence[str],
    references: Sequence[str],
    prompts: Sequence[str],
    config: EvaluationConfig,
    indices: Sequence[int] | None = None,
) -> EvaluationReport:
    """Score predictions against their references and source prompts.

    Args:
        predictions: The raw model outputs.
        references: The expected targets, aligned with ``predictions``.
        prompts: The source prompts, aligned with ``predictions``.
        config: Scoring options.
        indices: Corpus positions, aligned with ``predictions``. Defaults to
            their offsets.

    Returns:
        The aggregate report together with the per-example scores.

    Raises:
        ValueError: If the sequences have different lengths.
    """
    if not len(predictions) == len(references) == len(prompts):
        raise ValueError(
            "predictions, references and prompts must be aligned, got "
            f"{len(predictions)}, {len(references)} and {len(prompts)}"
        )

    positions = list(indices) if indices is not None else list(range(len(predictions)))
    valid_offsets = [offset for offset, text in enumerate(predictions) if is_valid(text)]

    valid_predictions = [predictions[offset] for offset in valid_offsets]
    valid_references = [references[offset] for offset in valid_offsets]
    valid_prompts = [prompts[offset] for offset in valid_offsets]

    distances = tree_edit_distances(valid_predictions, valid_references, config.ted_failure_penalty)
    hallucinations = hallucination_rates(valid_predictions, valid_prompts)
    coverages = coverage_scores(valid_predictions, valid_prompts)

    alignments = align_predictions(predictions, prompts, config.alignment_levels)
    reconstruction = score_alignments(alignments)

    scored = {
        offset: ExampleScore(
            index=positions[offset],
            valid=True,
            tree_edit_distance=distances[rank],
            hallucination_rate=hallucinations[rank],
            coverage_score=coverages[rank],
        )
        for rank, offset in enumerate(valid_offsets)
    }
    details = []
    for offset in range(len(predictions)):
        score = scored.get(offset, ExampleScore(index=positions[offset], valid=False))
        alignment = alignments[offset]
        details.append(
            ExampleScore(
                index=score.index,
                valid=score.valid,
                tree_edit_distance=score.tree_edit_distance,
                hallucination_rate=score.hallucination_rate,
                coverage_score=score.coverage_score,
                reconstructed=None if alignment is None else alignment.reconstructs(),
                alignment_rate=(
                    None
                    if alignment is None or not alignment.scored
                    else alignment.located / len(alignment.scored)
                ),
            )
        )

    return EvaluationReport(
        total=len(predictions),
        valid=len(valid_offsets),
        json_validity_rate=json_validity_rate(predictions),
        parse_rate=parse_rate(predictions),
        field_f1=field_f1_scores(valid_predictions, valid_references),
        confusion=token_confusion(
            valid_predictions, valid_references, valid_prompts, config.alignment_levels
        ),
        reconstruction=reconstruction,
        mean_tree_edit_distance=_mean(distances),
        mean_hallucination_rate=_mean(hallucinations),
        mean_coverage_score=_mean(coverages),
        details=details,
    )


def write_report(
    report: EvaluationReport,
    directory: str | Path,
    top_confusions: int = 5,
) -> tuple[Path, Path]:
    """Write the aggregate report and the per-example scores.

    Args:
        report: The outcome of :func:`evaluate`.
        directory: Destination directory, created if absent.
        top_confusions: Number of misassignments listed in the report.

    Returns:
        The paths of the results file and the detail file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    results_path = directory / RESULTS_FILE
    results_path.write_text(
        json.dumps(report.as_dict(top_confusions), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    details_path = directory / DETAILS_FILE
    with details_path.open("w", encoding="utf-8", newline="\n") as handle:
        for detail in report.details:
            handle.write(json.dumps(detail.as_dict(), ensure_ascii=False))
            handle.write("\n")

    return results_path, details_path


def _mean(values: Sequence[float | None]) -> float | None:
    """Average the values that are defined, returning ``None`` when none is."""
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None
