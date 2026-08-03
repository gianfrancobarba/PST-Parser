"""Share of predictions that are a well-formed target object.

Parsing alone is not the question the structural metric asks: an empty object
parses and answers nothing. A prediction counts as valid when it reads as JSON
*and* carries the object the taxonomy declares, which is what makes the rate a
statement about the structure the model produced.

The two conditions are also reported apart, because a truncated output and a
well-formed object of the wrong shape are different failures.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pstparser.pst.target import conforms


def parses(prediction: str) -> bool:
    """Report whether a prediction reads as JSON.

    Args:
        prediction: The raw model output.

    Returns:
        ``True`` when the string is well-formed JSON, whatever its shape.
    """
    try:
        json.loads(prediction)
    except json.JSONDecodeError:
        return False
    return True


def is_valid(prediction: str) -> bool:
    """Report whether a prediction is a well-formed target object.

    Args:
        prediction: The raw model output.

    Returns:
        ``True`` when the string parses and satisfies the target schema.
    """
    try:
        payload = json.loads(prediction)
    except json.JSONDecodeError:
        return False
    return conforms(payload)


def json_validity_rate(predictions: Sequence[str]) -> float:
    """Fraction of predictions that are a well-formed target object.

    Args:
        predictions: The raw model outputs.

    Returns:
        A score between 0 and 1, or 0 when there is nothing to score.
    """
    if not predictions:
        return 0.0
    return sum(is_valid(prediction) for prediction in predictions) / len(predictions)


def parse_rate(predictions: Sequence[str]) -> float:
    """Fraction of predictions that read as JSON, whatever their shape.

    Reported beside the validity rate, so that a gap between the two reads as
    outputs that are well-formed but not of the declared shape.

    Args:
        predictions: The raw model outputs.

    Returns:
        A score between 0 and 1, or 0 when there is nothing to score.
    """
    if not predictions:
        return 0.0
    return sum(parses(prediction) for prediction in predictions) / len(predictions)
