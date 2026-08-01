"""Structural distance between the predicted tree and the reference tree."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from apted import APTED
from apted.helpers import Tree

#: Placeholder name given to the elements of a list, which have no key of their own.
_LIST_ITEM = "item"

#: Name given to the outermost node.
_ROOT = "root"


def tree_edit_distances(
    predictions: Sequence[str],
    references: Sequence[str],
    failure_penalty: float,
) -> list[float]:
    """Compute, for each pair, the edit distance between the two trees.

    The distance is the number of node insertions, deletions and renames needed
    to turn one tree into the other, so a lower value means a closer structure.
    A pair that cannot be compared, because the prediction does not parse or
    cannot be expressed as a tree, is charged ``failure_penalty``.

    Args:
        predictions: The raw model outputs.
        references: The expected targets, aligned with ``predictions``.
        failure_penalty: Distance charged when a pair cannot be compared.

    Returns:
        One distance per pair, in input order.
    """
    distances: list[float] = []
    for prediction, reference in zip(predictions, references, strict=True):
        try:
            expected = Tree.from_text(to_bracket_notation(json.loads(reference)))
            predicted = Tree.from_text(to_bracket_notation(json.loads(prediction)))
            distances.append(float(APTED(expected, predicted).compute_edit_distance()))
        except Exception:
            distances.append(failure_penalty)
    return distances


def to_bracket_notation(node: Any, name: str = _ROOT) -> str:
    """Render a tree in the bracket notation the distance algorithm reads.

    Each node becomes ``{name...children}``. Braces occurring inside a text
    segment are removed, since they would otherwise be read as structure.

    Args:
        node: A tree, a subtree, a list of segments or a single segment.
        name: Name given to this node.

    Returns:
        The bracketed representation.
    """
    if isinstance(node, dict):
        children = "".join(to_bracket_notation(value, key) for key, value in node.items())
        return f"{{{name}{children}}}"
    if isinstance(node, list):
        children = "".join(to_bracket_notation(item, _LIST_ITEM) for item in node)
        return f"{{{name}{children}}}"

    leaf = str(node).replace("{", "").replace("}", "")
    return f"{{{name}{{{leaf}}}}}"
