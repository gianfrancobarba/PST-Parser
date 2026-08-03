"""Turning a set of predictions into the positional artefact and its rates.

Both sides are located, not only the prediction. A reference phrase that cannot
be found in its own prompt is a defect of the annotation, and reporting the two
separately is what keeps such a defect from being charged to the model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pstparser.data.alignments import ALIGNMENTS_FILE, AlignmentRecord, write_alignments
from pstparser.data.predictions import PredictionRecord
from pstparser.evaluation.metrics.reconstruction import (
    ReconstructionScores,
    align_predictions,
    score_alignments,
)
from pstparser.pst.alignment import DEFAULT_LEVELS, Alignment, MatchLevel


@dataclass(frozen=True)
class AlignmentOutcome:
    """Artefacts and rates of a completed alignment run.

    Attributes:
        records: One record per prediction, in input order.
        prediction_scores: Rates measured on the predicted trees.
        target_scores: Rates measured on the reference trees.
        alignments_path: File the records were written to.
    """

    records: list[AlignmentRecord]
    prediction_scores: ReconstructionScores
    target_scores: ReconstructionScores
    alignments_path: Path


def run_alignment(
    records: Sequence[PredictionRecord],
    directory: str | Path,
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> AlignmentOutcome:
    """Locate every phrase of every prediction and reference, and persist it.

    Args:
        records: The predictions to align, together with their references.
        directory: Destination directory, created if absent.
        levels: Normalisations to try, from the most conservative.

    Returns:
        The records written, and the rates for both sides.
    """
    prompts = [record.prompt for record in records]
    predicted = align_predictions([record.prediction for record in records], prompts, levels)
    expected = align_predictions([record.target for record in records], prompts, levels)

    written = [
        AlignmentRecord(
            index=record.index,
            prompt=record.prompt,
            target=_serialise(reference),
            prediction=_serialise(prediction),
            target_alignment=reference.as_dict() if reference is not None else None,
            prediction_alignment=prediction.as_dict() if prediction is not None else None,
        )
        for record, prediction, reference in zip(records, predicted, expected, strict=True)
    ]

    return AlignmentOutcome(
        records=written,
        prediction_scores=score_alignments(predicted),
        target_scores=score_alignments(expected),
        alignments_path=write_alignments(written, Path(directory) / ALIGNMENTS_FILE),
    )


def _serialise(alignment: Alignment | None) -> str | None:
    """Render the positional tree of an alignment, if there is one."""
    return None if alignment is None else json.dumps(alignment.as_tree(), ensure_ascii=False)
