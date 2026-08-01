"""Share of predicted text that does not come from the source prompt."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pstparser.evaluation.metrics._text import flatten, normalise_text, tokenise


def hallucination_rates(predictions: Sequence[str], prompts: Sequence[str]) -> list[float]:
    """Fraction of predicted tokens absent from the prompt, per prediction.

    The task is extractive, so every token of the output should already occur in
    the prompt and a faithful model scores zero. Predictions that do not parse
    are omitted, and a prediction holding no text scores zero.

    Args:
        predictions: The raw model outputs.
        prompts: The source prompts, aligned with ``predictions``.

    Returns:
        One rate per parseable prediction, in input order.
    """
    rates: list[float] = []
    for prediction, prompt in zip(predictions, prompts, strict=True):
        try:
            predicted_tree = json.loads(prediction)
        except json.JSONDecodeError:
            continue

        predicted_tokens = tokenise(" ".join(flatten(predicted_tree).values()))
        if not predicted_tokens:
            rates.append(0.0)
            continue

        available = set(normalise_text(prompt).split())
        absent = [token for token in predicted_tokens if token not in available]
        rates.append(len(absent) / len(predicted_tokens))

    return rates
