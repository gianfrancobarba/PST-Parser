"""Corrections applied to the annotated corpus as it is read.

The spreadsheet is an input: it is the artefact that was handed over, and it
stays byte for byte as it was. Every correction lives here instead, next to the
record it applies to and the reason it was made, so that what was changed can be
read without diffing a binary.

A fix that no longer applies is an error rather than a silent no-op. That is the
point: a correction and the corpus it corrects cannot drift apart unnoticed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pstparser.data.cleaning import SEGMENT_SEPARATOR
from pstparser.data.errors import CorpusError


@dataclass(frozen=True)
class AnnotationFix:
    """One correction to one cell of the corpus.

    Exactly one operation is declared. ``find`` replaces a passage that must
    occur once and only once; ``unwrap`` removes one leading and one trailing
    occurrence of a character the annotation was enclosed in; ``clear`` empties
    a cell that holds no annotation at all; ``set`` writes a cell whose content
    the spreadsheet destroyed, and states what was there instead; ``append``
    adds a further segment to a cell that was left incomplete.

    Attributes:
        row: Position of the record in the worksheet, counting from zero.
        column: Heading of the cell to correct.
        reason: Why the cell is wrong, in the annotator's terms.
        find: Passage to replace.
        replace: What to put in its place.
        unwrap: Character the cell is enclosed in.
        clear: Whether the cell should hold nothing.
        set: Text to write over the cell.
        append: Passage of the prompt to add as a further segment.
        expect: What the cell must currently hold for ``set`` to apply, so that
            overwriting cannot happen against a cell that has since changed.
    """

    row: int
    column: str
    reason: str
    find: str | None = None
    replace: str = ""
    unwrap: str | None = None
    clear: bool = False
    set: str | None = None
    append: str | None = None
    expect: str | None = None

    def apply(self, value: Any) -> str:
        """Return the corrected value of a cell.

        Args:
            value: The cell as read from the spreadsheet.

        Returns:
            The corrected text.

        Raises:
            CorpusError: If the correction no longer describes the cell.
        """
        if self.clear:
            return ""

        text = "" if pd.isna(value) else str(value)

        if self.set is not None:
            if self.expect is not None and text != self.expect:
                raise self._stale(f"cell holds {text[:40]!r}, not the expected value")
            return self.set

        if self.append is not None:
            if self.append in text:
                raise self._stale("the passage is already there")
            return f"{text}{SEGMENT_SEPARATOR}{self.append}" if text.strip() else self.append

        if self.unwrap is not None:
            if not (text.startswith(self.unwrap) and text.endswith(self.unwrap)):
                raise self._stale(f"cell is not enclosed in {self.unwrap!r}")
            return text[len(self.unwrap) : -len(self.unwrap)]

        if self.find is None:
            raise self._stale("no operation declared")

        occurrences = text.count(self.find)
        if occurrences != 1:
            raise self._stale(f"passage occurs {occurrences} times, expected once")
        return text.replace(self.find, self.replace)

    def _stale(self, detail: str) -> CorpusError:
        """Build the error raised when a fix does not describe its cell."""
        return CorpusError(
            f"annotation fix for row {self.row} [{self.column}] no longer applies: {detail}"
        )


def load_fixes(source: str | Path) -> list[AnnotationFix]:
    """Read the corrections from their file.

    Args:
        source: File holding the corrections.

    Returns:
        The corrections, in file order. Empty when the file declares none.

    Raises:
        CorpusError: If the file is missing or malformed.
    """
    source = Path(source)
    if not source.is_file():
        raise CorpusError(f"annotation fixes not found: {source}")

    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    entries = payload.get("fixes") or []
    if not isinstance(entries, list):
        raise CorpusError(f"'fixes' must be a list in {source}")

    fixes = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CorpusError(f"fix {position} is not a mapping in {source}")
        try:
            fixes.append(AnnotationFix(**entry))
        except TypeError as exc:
            raise CorpusError(f"fix {position} in {source}: {exc}") from exc
    return fixes


def apply_fixes(frame: pd.DataFrame, fixes: Iterable[AnnotationFix]) -> pd.DataFrame:
    """Apply every correction to a copy of the worksheet.

    Args:
        frame: The worksheet as read from the spreadsheet.
        fixes: The corrections to apply, in order.

    Returns:
        A copy of the worksheet with the corrections applied.

    Raises:
        CorpusError: If a correction names a row or column that is not there,
            or no longer describes the cell it names.
    """
    corrected = frame.copy()
    for fix in fixes:
        if fix.column not in corrected.columns:
            raise CorpusError(f"annotation fix names an unknown column: {fix.column}")
        if not 0 <= fix.row < len(corrected):
            raise CorpusError(f"annotation fix names row {fix.row}, out of {len(corrected)}")

        label = corrected.index[fix.row]
        corrected.at[label, fix.column] = fix.apply(corrected.at[label, fix.column])
    return corrected


def describe(fixes: Sequence[AnnotationFix]) -> dict[str, Any]:
    """Summarise the corrections, for the run manifest and the report.

    Args:
        fixes: The corrections that were applied.

    Returns:
        A JSON-serialisable summary.
    """
    return {
        "count": len(fixes),
        "records": sorted({fix.row for fix in fixes}),
        "columns": sorted({fix.column for fix in fixes}),
    }
