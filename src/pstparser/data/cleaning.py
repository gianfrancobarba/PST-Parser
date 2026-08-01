"""Normalisation of individual spreadsheet cells into leaf values."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd


def clean_field(value: Any, is_list: bool = True) -> Any:
    """Convert a spreadsheet cell into the value stored at a leaf.

    Empty cells become an empty list, or ``None`` when the leaf is scalar. Text
    that is delimited by square brackets is parsed as a Python literal, so that
    an annotator can enter several segments as a list. Any other text is wrapped
    in a single-element list.

    Args:
        value: Raw cell value as read from the spreadsheet.
        is_list: Whether the destination leaf holds a list of segments.

    Returns:
        The leaf value: a list of segments, ``None``, or the original value when
        it is neither text nor missing.
    """
    if pd.isna(value) or value == "":
        return [] if is_list else None

    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("[") and candidate.endswith("]"):
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                pass

        if is_list:
            return [value]

    return value
