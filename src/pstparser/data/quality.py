"""Integrity check comparing each target against the prompt it was derived from.

The annotation is extractive: every segment is a span of the prompt, copied
verbatim. So the check is not whether the target is roughly as long as its
prompt, but whether each of its segments can actually be found there, and how
much of the prompt the segments account for between them. Comparing lengths
answers neither question — two strings of equal length and unrelated content
pass it, which is how the worst record in the corpus went unreported.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pstparser.pst.alignment import DEFAULT_LEVELS, Alignment, MatchLevel, align_tree

IssueKind = Literal["unparseable_target", "absent_segment", "contested_segment", "low_coverage"]

#: How much of a segment is quoted when reporting it.
_EXCERPT = 60


@dataclass(frozen=True)
class QualityIssue:
    """A single record that failed the integrity check.

    Attributes:
        index: Position of the record in the corpus.
        kind: What went wrong.
        detail: Human-readable explanation.
        coverage: Share of the prompt the located segments account for, when it
            could be measured.
        segment: The segment at fault, when one is.
    """

    index: int
    kind: IssueKind
    detail: str
    coverage: float | None = None
    segment: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the issue."""
        return {
            "index": self.index,
            "kind": self.kind,
            "detail": self.detail,
            "coverage": self.coverage,
            "segment": self.segment,
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

    @property
    def affected(self) -> int:
        """How many records raised at least one issue."""
        return len({issue.index for issue in self.issues})

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "total": self.total,
            "min_coverage_ratio": self.min_coverage_ratio,
            "issue_count": len(self.issues),
            "affected_records": self.affected,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def check_corpus(
    prompts: Sequence[str],
    targets: Sequence[str],
    min_coverage_ratio: float,
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> QualityReport:
    """Verify that each target is made of spans of its own prompt.

    A record is reported when its target cannot be read, when a segment cannot
    be found in the prompt at any level of normalisation, or when the segments
    that were found leave more of the prompt unaccounted for than the threshold
    allows.

    Args:
        prompts: Source prompts, in corpus order.
        targets: Serialised targets, aligned with ``prompts``.
        min_coverage_ratio: Minimum share of the prompt the segments must cover.
        levels: Normalisations to try when locating a segment.

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
        except json.JSONDecodeError as exc:
            issues.append(
                QualityIssue(
                    index=index,
                    kind="unparseable_target",
                    detail=f"target could not be read back: {exc}",
                )
            )
            continue

        alignment = align_tree(prompt, parsed, levels)
        for leaf in alignment.scored:
            if leaf.absent:
                issues.append(
                    QualityIssue(
                        index=index,
                        kind="absent_segment",
                        detail=f"{leaf.path} does not occur in the prompt",
                        segment=_excerpt(leaf.phrase),
                    )
                )
            elif leaf.contested:
                issues.append(
                    QualityIssue(
                        index=index,
                        kind="contested_segment",
                        detail=f"{leaf.path} occurs only where another segment already sits",
                        segment=_excerpt(leaf.phrase),
                    )
                )

        covered = coverage(alignment)
        if covered < min_coverage_ratio:
            issues.append(
                QualityIssue(
                    index=index,
                    kind="low_coverage",
                    detail=f"segments account for {covered:.1%} of the prompt",
                    coverage=covered,
                )
            )

    return QualityReport(
        total=len(prompts),
        min_coverage_ratio=min_coverage_ratio,
        issues=issues,
    )


def coverage(alignment: Alignment) -> float:
    """Share of a prompt that the located segments occupy.

    Whitespace is left out of the count on both sides, because the space between
    two segments belongs to neither and would otherwise be charged as missing.

    Args:
        alignment: The outcome of locating a target in its prompt.

    Returns:
        A value between 0 and 1. An empty prompt is fully covered.
    """
    prompt = alignment.prompt
    total = sum(not character.isspace() for character in prompt)
    if not total:
        return 1.0

    claimed = bytearray(len(prompt))
    for leaf in alignment.leaves:
        if leaf.start is not None and leaf.end is not None:
            claimed[leaf.start : leaf.end] = b"\x01" * (leaf.end - leaf.start)

    accounted = sum(
        flag and not character.isspace() for flag, character in zip(claimed, prompt, strict=True)
    )
    return accounted / total


def _excerpt(text: str) -> str:
    """Quote the start of a segment, so a report stays readable."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= _EXCERPT else f"{collapsed[:_EXCERPT]}..."
