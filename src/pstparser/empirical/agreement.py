"""Agreement between two annotations of the same sources: character kappa, boundary F1.

Both measures canonicalise the annotations with the policy of the renderer
(whitespace between spans belongs to the preceding span, a prefix to the first),
so two annotations that render to the same H agree perfectly. Whitespace
characters are never counted. A mask restricts both measures to chosen
characters: the primary measure uses the judgement units, the
measure on every non-whitespace character is descriptive.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from pstparser.empirical.contracts import SpanAnnotation

AGREEMENT_VERSION: Final = "char-agreement-v1"


@dataclass(frozen=True)
class AgreementItem:
    """Two annotations of one text, and the characters that count (None: all)."""

    task_id: str
    text: str
    first: tuple[SpanAnnotation, ...]
    second: tuple[SpanAnnotation, ...]
    mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        """Refuse a mask of the wrong length."""
        if self.mask is not None and len(self.mask) != len(self.text):
            raise ValueError(f"{self.task_id}: the mask must have one flag per character")


def _segments(text: str, spans: Sequence[SpanAnnotation]) -> list[tuple[int, int, str]]:
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    if not ordered:
        raise ValueError("an annotation needs at least one span")
    segments = []
    for rank, span in enumerate(ordered):
        if not 0 <= span.start < span.end <= len(text):
            raise ValueError("span offsets are outside the text")
        if rank and span.start < ordered[rank - 1].end:
            raise ValueError("overlapping spans")
        start = 0 if rank == 0 else span.start
        end = ordered[rank + 1].start if rank + 1 < len(ordered) else len(text)
        segments.append((start, end, span.path))
    return segments


def character_labels(text: str, spans: Sequence[SpanAnnotation]) -> list[str]:
    """The leaf of every character under the renderer's separator policy."""
    labels = [""] * len(text)
    for start, end, path in _segments(text, spans):
        labels[start:end] = [path] * (end - start)
    return labels


def _counted(item: AgreementItem) -> list[int]:
    return [
        index
        for index, character in enumerate(item.text)
        if not character.isspace() and (item.mask is None or item.mask[index])
    ]


def char_kappa(items: Sequence[AgreementItem]) -> float | None:
    """Cohen's kappa over the counted characters of every item, pooled.

    Returns 1.0 when the two annotations agree on every counted character (even
    when a single leaf leaves no chance agreement to correct), and None when no
    character is counted.
    """
    pairs: list[tuple[str, str]] = []
    for item in items:
        first = character_labels(item.text, item.first)
        second = character_labels(item.text, item.second)
        pairs.extend((first[index], second[index]) for index in _counted(item))
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    if observed == 1:
        return 1.0
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = math.fsum(left[label] * right[label] for label in left) / (total * total)
    return (observed - expected) / (1 - expected)


def _boundaries(item: AgreementItem, spans: Sequence[SpanAnnotation]) -> set[int]:
    counted = set(_counted(item))
    # A boundary is the start of a canonical segment; it counts when its first
    # non-whitespace character is counted.
    found = set()
    for start, end, _ in _segments(item.text, spans)[1:]:
        first = next((i for i in range(start, end) if not item.text[i].isspace()), None)
        if first is not None and first in counted:
            found.add(first)
    return found


def boundary_f1(items: Sequence[AgreementItem]) -> float | None:
    """F1 of the segment boundaries of the second annotation against the first, pooled.

    Returns 1.0 when neither annotation has a counted boundary, None when no
    character is counted at all.
    """
    matched = first_total = second_total = 0
    counted_any = False
    for item in items:
        counted_any = counted_any or bool(_counted(item))
        first = _boundaries(item, item.first)
        second = _boundaries(item, item.second)
        matched += len(first & second)
        first_total += len(first)
        second_total += len(second)
    if not counted_any:
        return None
    if first_total == second_total == 0:
        return 1.0
    return 2 * matched / (first_total + second_total)
