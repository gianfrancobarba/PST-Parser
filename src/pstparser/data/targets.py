"""Mapping a source record onto the target tree."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pstparser.data.cleaning import clean_field
from pstparser.pst.taxonomy import LEAF_PATHS, assemble


@dataclass(frozen=True)
class RawRecord:
    """One annotated prompt as a source hands it over, before conversion.

    It is what every source format is reduced to, so that the preparation reads
    the same shape whether the annotation came from a worksheet or from a text
    file, and has one code path rather than one per format.

    Attributes:
        prompt: The raw, unsegmented prompt.
        leaves: Segments held at each leaf, keyed by dotted path. Every leaf the
            taxonomy declares is present; one with nothing annotated holds an
            empty list.
        paradigm: The reasoning paradigm the prompt was written to exercise,
            where a source records it. It describes the prompt's provenance, not
            its structure, so it never reaches the target.
    """

    prompt: str
    leaves: dict[str, list[str]]
    paradigm: str | None = None


def build_leaves(row: Mapping[str, Any], column_mapping: Mapping[str, str]) -> dict[str, list[str]]:
    """Read the segments of every leaf out of a spreadsheet row.

    Args:
        row: Spreadsheet row, keyed by column name.
        column_mapping: Mapping from dotted leaf path to source column.

    Returns:
        The segments of each declared leaf, keyed by dotted path.

    Raises:
        KeyError: If ``column_mapping`` does not cover every declared leaf.
    """
    return {path: clean_field(row.get(column_mapping[path])) for path in LEAF_PATHS}


def build_target(row: Mapping[str, Any], column_mapping: Mapping[str, str]) -> dict[str, Any]:
    """Assemble the target tree for a single annotated record.

    Args:
        row: Spreadsheet row, keyed by column name.
        column_mapping: Mapping from dotted leaf path to source column.

    Returns:
        The nested target tree.

    Raises:
        KeyError: If ``column_mapping`` does not cover every declared leaf.
    """
    return assemble(build_leaves(row, column_mapping))
