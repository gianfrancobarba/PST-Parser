"""Helpers shared by the content metrics."""

from __future__ import annotations

import collections
import re
from typing import Any

#: Runs of non-word characters collapsed during normalisation.
_NON_WORD = re.compile(r"\W+")

#: List indices appended to a flattened key, removed when aggregating by field.
_LIST_INDEX = re.compile(r"\[\d+\]")


def normalise_text(text: str) -> str:
    """Lowercase a string and reduce punctuation to single spaces.

    Args:
        text: The string to normalise.

    Returns:
        The normalised string, without leading or trailing spaces.
    """
    return _NON_WORD.sub(" ", text.lower()).strip()


def tokenise(text: str) -> list[str]:
    """Split a string into normalised whitespace-separated tokens.

    Args:
        text: The string to tokenise.

    Returns:
        The tokens, in order of appearance.
    """
    return normalise_text(text).split()


def generalise_key(key: str) -> str:
    """Drop list indices from a flattened key so that fields can be aggregated.

    Args:
        key: A flattened key such as ``prompt.context.data[0]``.

    Returns:
        The key without indices, such as ``prompt.context.data``.
    """
    return _LIST_INDEX.sub("", key)


def flatten(node: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a tree into a mapping from path to text segment.

    Dictionary keys extend the path with a dot; list positions extend it with a
    bracketed index. Only text leaves are kept.

    Args:
        node: A tree, a subtree, a list of segments or a single segment.
        prefix: Path accumulated so far.

    Returns:
        A mapping from path to the segment stored there.
    """
    flattened: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten(value, path))
    elif isinstance(node, list):
        for position, item in enumerate(node):
            flattened.update(flatten(item, f"{prefix}[{position}]"))
    elif isinstance(node, str):
        flattened[prefix] = node
    return flattened


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall between two strings.

    Tokens are compared as multisets, so a repeated word counts as many times as
    it appears in both strings. Two empty strings are considered a perfect
    match.

    Args:
        prediction: The predicted text.
        reference: The expected text.

    Returns:
        A score between 0 and 1.
    """
    predicted_tokens = tokenise(prediction)
    reference_tokens = tokenise(reference)

    if not predicted_tokens and not reference_tokens:
        return 1.0

    shared = collections.Counter(predicted_tokens) & collections.Counter(reference_tokens)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
