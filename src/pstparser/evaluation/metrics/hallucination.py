"""Share of predicted text that does not come from the source prompt."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pstparser.evaluation.metrics._text import flatten, tokenise


def hallucination_rates(
    predictions: Sequence[str],
    prompts: Sequence[str],
) -> list[float | None]:
    """Fraction of the prediction's distinct tokens absent from the prompt.

    Both sides are token sets, matching the definition of the coverage score, so
    that the two rates speak of the same quantities. The task is extractive, so
    every token of the output should already occur in the prompt and a faithful
    model scores zero.

    The rate is a share of what the model emitted, so it is undefined when the
    model emitted nothing to judge: a prediction that cannot be read, and one
    whose leaves are all empty, both report ``None``. Reporting zero for the
    latter would credit an empty answer with perfect faithfulness.

    Args:
        predictions: The raw model outputs.
        prompts: The source prompts, aligned with ``predictions``.

    Returns:
        One rate per prediction, in input order, or ``None`` where the rate has
        no meaning.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    rates: list[float | None] = []
    for prediction, prompt in zip(predictions, prompts, strict=True):
        try:
            predicted_tree = json.loads(prediction)
        except json.JSONDecodeError:
            rates.append(None)
            continue

        generated = set(tokenise(" ".join(flatten(predicted_tree).values())))
        if not generated:
            rates.append(None)
            continue

        available = set(tokenise(prompt))
        rates.append(len(generated - available) / len(generated))

    return rates
