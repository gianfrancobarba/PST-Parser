"""Integrity check comparing each target against the prompt it was derived from."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pstparser.pst.taxonomy import ROOT_KEY, collect_text

#: Characters ignored when comparing a prompt with its reconstruction.
_IGNORED_CHARACTERS = (" ", "\n", "\t")

IssueKind = Literal["unparseable_target", "low_coverage"]


@dataclass(frozen=True)
class QualityIssue:
    """A single record that failed the integrity check.

    Attributes:
        index: Position of the record in the corpus.
        kind: What went wrong.
        detail: Human-readable explanation.
        source_length: Length of the normalised prompt, when measured.
        reconstructed_length: Length of the normalised reconstruction, when
            measured.
    """

    index: int
    kind: IssueKind
    detail: str
    source_length: int | None = None
    reconstructed_length: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the issue."""
        return {
            "index": self.index,
            "kind": self.kind,
            "detail": self.detail,
            "source_length": self.source_length,
            "reconstructed_length": self.reconstructed_length,
        }


@dataclass(frozen=True)
class QualityReport:
    """Outcome of the integrity check over a corpus.

    Attributes:
        total: Number of records examined.
        min_coverage_ratio: Threshold the check was run with.
        issues: Records that failed, in corpus order.
    """

    total: int
    min_coverage_ratio: float
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every record satisfied the check."""
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "total": self.total,
            "min_coverage_ratio": self.min_coverage_ratio,
            "issue_count": len(self.issues),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def check_corpus(
    prompts: Sequence[str],
    targets: Sequence[str],
    min_coverage_ratio: float,
) -> QualityReport:
    """Verify that each target accounts for enough of its source prompt.

    A target is reported when it cannot be parsed, or when the text recovered
    from its leaves is shorter than ``min_coverage_ratio`` times the prompt.
    Both sides are compared after removing whitespace, so that differences in
    line breaks or indentation do not count as missing content.

    Args:
        prompts: Source prompts, in corpus order.
        targets: Serialised targets, aligned with ``prompts``.
        min_coverage_ratio: Minimum accepted length ratio.

    Returns:
        The report. It never raises on a failing record; callers decide whether
        a non-empty issue list is fatal.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(prompts) != len(targets):
        raise ValueError(
            f"prompts and targets must be aligned, got {len(prompts)} and {len(targets)}"
        )

    issues: list[QualityIssue] = []
    for index, (prompt, target) in enumerate(zip(prompts, targets, strict=True)):
        try:
            parsed = json.loads(target)
            reconstructed = collect_text(parsed[ROOT_KEY])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            issues.append(
                QualityIssue(
                    index=index,
                    kind="unparseable_target",
                    detail=f"target could not be read back: {exc}",
                )
            )
            continue

        source_length = len(_normalise(prompt))
        reconstructed_length = len(_normalise(reconstructed))
        if reconstructed_length < source_length * min_coverage_ratio:
            issues.append(
                QualityIssue(
                    index=index,
                    kind="low_coverage",
                    detail=f"{source_length - reconstructed_length} characters unaccounted for",
                    source_length=source_length,
                    reconstructed_length=reconstructed_length,
                )
            )

    return QualityReport(
        total=len(prompts),
        min_coverage_ratio=min_coverage_ratio,
        issues=issues,
    )


def _normalise(text: str) -> str:
    """Strip the characters that are not compared between prompt and target."""
    for character in _IGNORED_CHARACTERS:
        text = text.replace(character, "")
    return text
