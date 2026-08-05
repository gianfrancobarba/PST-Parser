"""Normalisation of individual spreadsheet cells into leaf values."""

from __future__ import annotations

from typing import Any, Final

import pandas as pd

#: Token an annotator writes between two parts of the same category that are
#: not contiguous in the prompt.
SEGMENT_SEPARATOR: Final = "<sep>"


def clean_field(value: Any, separator: str = SEGMENT_SEPARATOR) -> list[str]:
    """Convert a spreadsheet cell into the segments stored at a leaf.

    A cell holds the text of one category, which may be several parts of the
    prompt that do not sit next to each other. The annotator marks each break
    with the separator, and every part becomes a segment of its own. Keeping
    them fused would put the separator itself into the target, which is a token
    the prompt never contained and which the model would learn to emit.

    Whitespace at the edge of a part belongs between the segments rather than to
    either of them, so it is dropped; a part holding nothing else is not a
    segment at all. A cell that is not text is rendered as text rather than
    refused: the integrity check is what decides whether the result belongs to
    the prompt, and it is better placed to say so than this function.

    Args:
        value: Raw cell value as read from the spreadsheet.
        separator: Token marking a break between two parts.

    Returns:
        The segments, in the order they were written. Empty when the cell is
        empty or missing.
    """
    if pd.isna(value):
        return []

    text = value if isinstance(value, str) else str(value)
    parts = (part.strip() for part in text.split(separator))
    return [part for part in parts if part]
