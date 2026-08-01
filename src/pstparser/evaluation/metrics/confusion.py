"""Which field a segment was assigned to, against where it belonged."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from pstparser.evaluation.metrics._text import flatten, generalise_key, normalise_text


@dataclass(frozen=True)
class Confusion:
    """Counts of predicted against expected field assignments.

    Attributes:
        matrix: Square matrix whose rows are expected fields and columns
            predicted fields.
        labels: Field paths, in the order used by the matrix.
    """

    matrix: Any
    labels: list[str]

    def top_confusions(self, limit: int) -> list[tuple[str, str, int]]:
        """List the most frequent misassignments.

        Args:
            limit: Maximum number of entries returned.

        Returns:
            Triples of expected field, predicted field and count, ordered by
            decreasing count. Correct assignments are excluded.
        """
        if self.matrix.size == 0:
            return []

        off_diagonal = np.array(self.matrix, copy=True)
        np.fill_diagonal(off_diagonal, 0)

        flat_order = np.argsort(off_diagonal, axis=None)[::-1][:limit]
        confusions = []
        for position in flat_order:
            row, column = np.unravel_index(position, off_diagonal.shape)
            count = int(off_diagonal[row, column])
            if count > 0:
                confusions.append((self.labels[row], self.labels[column], count))
        return confusions

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the matrix."""
        return {
            "labels": list(self.labels),
            "matrix": [[int(value) for value in row] for row in self.matrix],
        }


def field_confusion(predictions: Sequence[str], references: Sequence[str]) -> Confusion:
    """Match predicted segments to the field they occupy in the reference.

    A segment contributes a pair only when its normalised text appears verbatim
    in the reference; segments that were reworded, or that do not come from the
    prompt at all, cannot be located and are left out.

    Args:
        predictions: The raw model outputs.
        references: The expected targets, aligned with ``predictions``.

    Returns:
        The matrix together with the field labels it is indexed by. Both are
        empty when no segment could be located.
    """
    expected: list[str] = []
    predicted: list[str] = []
    labels: set[str] = set()

    for prediction, reference in zip(predictions, references, strict=True):
        try:
            predicted_tree = json.loads(prediction)
            reference_tree = json.loads(reference)
        except json.JSONDecodeError:
            continue

        flat_prediction = flatten(predicted_tree)
        by_content = {
            normalise_text(text): path for path, text in flatten(reference_tree).items() if text
        }

        for path, text in flat_prediction.items():
            if not text:
                continue

            predicted_label = generalise_key(path)
            labels.add(predicted_label)

            reference_path = by_content.get(normalise_text(text))
            if reference_path is None:
                continue

            expected_label = generalise_key(reference_path)
            labels.add(expected_label)
            expected.append(expected_label)
            predicted.append(predicted_label)

    if not expected:
        return Confusion(matrix=np.zeros((0, 0), dtype=int), labels=[])

    ordered = sorted(labels)
    return Confusion(
        matrix=confusion_matrix(expected, predicted, labels=ordered),
        labels=ordered,
    )
