"""Mapping a spreadsheet row onto the target tree."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pstparser.data.cleaning import clean_field
from pstparser.pst.taxonomy import LEAF_PATHS, assemble


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
    values = {path: clean_field(row.get(column_mapping[path])) for path in LEAF_PATHS}
    return assemble(values)
