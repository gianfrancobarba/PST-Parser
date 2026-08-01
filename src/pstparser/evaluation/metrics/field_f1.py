"""Token-level agreement between prediction and reference, reported per field."""

from __future__ import annotations

import collections
import json
from collections.abc import Sequence

import numpy as np

from pstparser.evaluation.metrics._text import flatten, generalise_key, token_f1


def field_f1_scores(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Average the token F1 of every leaf, grouped by field.

    Both trees are flattened to path-value pairs and compared over the union of
    their paths, so a segment present on one side only scores against an empty
    string. List indices are dropped before grouping, so every segment of a
    field contributes to the same average. Pairs whose prediction does not parse
    are skipped.

    Args:
        predictions: The raw model outputs.
        references: The expected targets, aligned with ``predictions``.

    Returns:
        A mapping from field path to its mean score.
    """
    per_field: dict[str, list[float]] = collections.defaultdict(list)

    for prediction, reference in zip(predictions, references, strict=True):
        try:
            predicted_tree = json.loads(prediction)
            reference_tree = json.loads(reference)
        except json.JSONDecodeError:
            continue

        flat_prediction = flatten(predicted_tree)
        flat_reference = flatten(reference_tree)

        for path in set(flat_prediction) | set(flat_reference):
            score = token_f1(flat_prediction.get(path, ""), flat_reference.get(path, ""))
            per_field[generalise_key(path)].append(score)

    return {field: float(np.mean(scores)) for field, scores in per_field.items()}
