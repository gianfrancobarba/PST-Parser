"""Share of the source prompt that the prediction accounts for."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pstparser.evaluation.metrics._text import flatten, tokenise


def coverage_scores(predictions: Sequence[str], prompts: Sequence[str]) -> list[float]:
    """Fraction of the prompt's distinct tokens that the prediction places.

    Both sides are token sets, so a word the prompt repeats counts once however
    many times it occurs: the question is whether the segmentation accounts for
    it, not how often. Segmenting the whole prompt scores one.

    A prediction that cannot be read scores zero, since it places nothing, and
    an empty prompt scores one, since there is nothing left to place.

    Args:
        predictions: The raw model outputs.
        prompts: The source prompts, aligned with ``predictions``.

    Returns:
        One score per prediction, in input order.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    scores: list[float] = []
    for prediction, prompt in zip(predictions, prompts, strict=True):
        try:
            predicted_tree = json.loads(prediction)
        except json.JSONDecodeError:
            scores.append(0.0)
            continue

        source = set(tokenise(prompt))
        if not source:
            scores.append(1.0)
            continue

        placed = set(tokenise(" ".join(flatten(predicted_tree).values())))
        scores.append(len(source & placed) / len(source))

    return scores
