"""Reading the annotated corpus from a spreadsheet."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be read or does not have the expected layout."""


def read_corpus(
    source_path: str | Path,
    sheet_name: str,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """Load one worksheet of the annotated corpus.

    Args:
        source_path: Spreadsheet file.
        sheet_name: Worksheet to read.
        required_columns: Columns that must be present.

    Returns:
        The worksheet contents, with the original column names preserved.

    Raises:
        CorpusError: If the file, the worksheet or any required column is
            missing.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise CorpusError(f"corpus not found: {source_path}")

    available = sheet_names(source_path)
    if sheet_name not in available:
        raise CorpusError(
            f"worksheet {sheet_name!r} not found in {source_path}; "
            f"available: {', '.join(available)}"
        )

    frame = pd.read_excel(source_path, sheet_name=sheet_name)

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise CorpusError(
            f"missing columns in {source_path}[{sheet_name}]: {', '.join(sorted(missing))}"
        )

    return frame


def iter_rows(frame: pd.DataFrame) -> Iterator[dict[str, Any]]:
    """Iterate over the rows of a worksheet as plain dictionaries.

    Spreadsheet headers are always text, which lets the rest of the pipeline
    work with string keys rather than the generic label type used by pandas.

    Args:
        frame: The worksheet contents.

    Yields:
        One dictionary per row, keyed by column name.
    """
    for row in frame.to_dict(orient="records"):
        yield cast(dict[str, Any], row)


def sheet_names(source_path: str | Path) -> list[str]:
    """List the worksheets of a spreadsheet without loading their contents.

    Args:
        source_path: Spreadsheet file.

    Returns:
        The worksheet names, in workbook order.
    """
    with pd.ExcelFile(source_path) as workbook:
        return [str(name) for name in workbook.sheet_names]
