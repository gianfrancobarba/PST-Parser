"""Share of the source prompt that the prediction accounts for."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pstparser.evaluation.metrics._text import flatten, normalise_text, tokenise


def coverage_scores(predictions: Sequence[str], prompts: Sequence[str]) -> list[float]:
    """Fraction of prompt tokens that appear in the prediction, per prediction.

    Segmenting the whole prompt means every token is placed somewhere, so a
    complete segmentation scores one. Predictions that do not parse score zero,
    and an empty prompt scores one.

    Args:
        predictions: The raw model outputs.
        prompts: The source prompts, aligned with ``predictions``.

    Returns:
        One score per prediction, in input order.
    """
    scores: list[float] = []
    for prediction, prompt in zip(predictions, prompts, strict=True):
        try:
            predicted_tree = json.loads(prediction)
        except json.JSONDecodeError:
            scores.append(0.0)
            continue

        placed = set(tokenise(" ".join(flatten(predicted_tree).values())))
        prompt_tokens = normalise_text(prompt).split()
        if not prompt_tokens:
            scores.append(1.0)
            continue

        covered = [token for token in prompt_tokens if token in placed]
        scores.append(len(covered) / len(prompt_tokens))

    return scores
