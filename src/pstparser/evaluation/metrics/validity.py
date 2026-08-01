"""Share of predictions that parse as JSON."""

from __future__ import annotations

import json
from collections.abc import Sequence


def is_valid(prediction: str) -> bool:
    """Report whether a prediction parses as JSON.

    Args:
        prediction: The raw model output.

    Returns:
        ``True`` when the string is well-formed JSON.
    """
    try:
        json.loads(prediction)
    except json.JSONDecodeError:
        return False
    return True


def json_validity_rate(predictions: Sequence[str]) -> float:
    """Fraction of predictions that parse as JSON.

    Args:
        predictions: The raw model outputs.

    Returns:
        A score between 0 and 1, or 0 when there is nothing to score.
    """
    if not predictions:
        return 0.0
    return sum(is_valid(prediction) for prediction in predictions) / len(predictions)
