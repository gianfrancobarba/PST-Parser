"""Locating extracted phrases in the prompt they were taken from.

The neural stage of the pipeline produces text alone: a leaf holds the phrases
that were extracted, never where they came from. This module is the second,
deterministic stage. It maps every phrase back onto the source prompt and derives
its position from the span of characters it occupies, which is what makes the
original text recoverable from the tree.

Positions are not assigned directly. Each phrase claims a span of the prompt, no
two spans may overlap, and the position of a leaf is the rank of its span. Stating
the problem this way is what makes it tractable: the non-overlap constraint is the
mutual exclusivity the segmentation is defined by, and the ranking of the spans is
what the recoverability of the prompt rests on.

Choosing which occurrence of a phrase to take, when it occurs more than once, is
an assignment problem that is NP-hard in general, so the rules below are local
and the outcome is not always the best one available. Where they fail, a phrase
is left without a home and reported as contested rather than quietly given a
span that belongs to another: the error is always in the direction of claiming
less than was established, never more.

Nothing here reaches outside the standard library, so locating a phrase costs no
more than reading the prediction file.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

#: How a text is normalised before a phrase is looked up inside it.
MatchLevel = Literal["exact", "collapsed", "stripped"]

#: Levels attempted in order, from the most conservative to the most tolerant.
DEFAULT_LEVELS: Final[tuple[MatchLevel, ...]] = ("exact", "collapsed", "stripped")

#: Runs of whitespace, reduced to a single space at the ``collapsed`` level.
_WHITESPACE_RUN: Final = re.compile(r"\s+")


def strip_whitespace(text: str) -> str:
    """Remove every whitespace character from a string.

    Args:
        text: The string to strip.

    Returns:
        The string without any whitespace.
    """
    return "".join(character for character in text if not character.isspace())


def iter_leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Walk a tree, yielding the path and the text of every string it holds.

    Mapping keys extend the path with a dot, list positions with a bracketed
    index, so that a path identifies one segment unambiguously.

    Args:
        node: A tree, a subtree, a list of segments or a single segment.
        prefix: Path accumulated so far.

    Yields:
        Pairs of path and segment, in traversal order.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_leaves(value, path)
    elif isinstance(node, list):
        for position, item in enumerate(node):
            yield from iter_leaves(item, f"{prefix}[{position}]")
    elif isinstance(node, str):
        yield prefix, node


@dataclass(frozen=True)
class Projection:
    """A normalised view of a text, and where each of its characters came from.

    Attributes:
        text: The normalised text, which phrases are searched in.
        offsets: For each character of ``text``, its index in the original.
    """

    text: str
    offsets: tuple[int, ...]

    def span(self, start: int, length: int) -> tuple[int, int]:
        """Map a match on the projection back onto the original text.

        Args:
            start: Offset of the match within :attr:`text`.
            length: Number of characters the match covers within :attr:`text`.

        Returns:
            The half-open interval the match occupies in the original text.

        Raises:
            ValueError: If the match is empty, which occupies no interval.
        """
        if length <= 0:
            raise ValueError("an empty match occupies no interval")
        return self.offsets[start], self.offsets[start + length - 1] + 1


def project(text: str, level: MatchLevel) -> Projection:
    """Normalise a text while remembering where every character came from.

    Args:
        text: The text to normalise.
        level: How tolerant the normalisation should be.

    Returns:
        The normalised text together with its offsets into ``text``.
    """
    if level == "exact":
        return Projection(text=text, offsets=tuple(range(len(text))))

    characters: list[str] = []
    offsets: list[int] = []

    if level == "stripped":
        for index, character in enumerate(text):
            if not character.isspace():
                characters.append(character)
                offsets.append(index)
        return Projection(text="".join(characters), offsets=tuple(offsets))

    index = 0
    while index < len(text):
        offsets.append(index)
        if text[index].isspace():
            characters.append(" ")
            while index < len(text) and text[index].isspace():
                index += 1
            continue
        characters.append(text[index])
        index += 1
    return Projection(text="".join(characters), offsets=tuple(offsets))


def project_phrase(phrase: str, level: MatchLevel) -> str:
    """Normalise a phrase the way the text it is searched in was normalised.

    Args:
        phrase: The phrase to normalise.
        level: How tolerant the normalisation should be.

    Returns:
        The phrase as it should be looked up at that level.
    """
    if level == "exact":
        return phrase
    if level == "collapsed":
        return _WHITESPACE_RUN.sub(" ", phrase).strip()
    return strip_whitespace(phrase)


def occurrences(projection: Projection, needle: str) -> tuple[tuple[int, int], ...]:
    """Find every interval a phrase occupies in a projected text.

    Overlapping occurrences are all reported, since the count is what tells an
    unambiguous phrase from one the disambiguation rule had to choose for.

    Args:
        projection: The text to search, together with its offsets.
        needle: The phrase to look for, normalised to the same level.

    Returns:
        The intervals in the original text, leftmost first. Empty when the
        phrase is empty or does not occur.
    """
    if not needle:
        return ()

    found: list[tuple[int, int]] = []
    start = projection.text.find(needle)
    while start != -1:
        found.append(projection.span(start, len(needle)))
        start = projection.text.find(needle, start + 1)
    return tuple(found)


@dataclass(frozen=True)
class AlignedLeaf:
    """One extracted phrase, located in the prompt it came from.

    Attributes:
        path: Path of the segment in the tree, such as ``prompt.context.data[0]``.
        phrase: The extracted text, unmodified.
        position: Rank of the phrase among the located ones, or ``None`` when it
            could not be found.
        start: Index of its first character in the prompt.
        end: Index just past its last character in the prompt.
        level: Normalisation at which the phrase was found.
        candidates: Occurrences the phrase had. For a located phrase these are
            the occurrences at the level it was placed at; for one that was not
            placed, the occurrences that were all already claimed.
    """

    path: str
    phrase: str
    position: int | None = None
    start: int | None = None
    end: int | None = None
    level: MatchLevel | None = None
    candidates: int = 0

    @property
    def located(self) -> bool:
        """Whether the phrase was found in the prompt."""
        return self.position is not None

    @property
    def blank(self) -> bool:
        """Whether the phrase holds no character other than whitespace."""
        return not self.phrase.strip()

    @property
    def ambiguous(self) -> bool:
        """Whether more than one occurrence could have been chosen."""
        return self.located and self.candidates > 1

    @property
    def contested(self) -> bool:
        """Whether the phrase occurs, but only where another one already sits.

        Two segments claiming the same span is not a failure of the search: it
        is the annotation placing one piece of text under two nodes, which the
        segmentation forbids. Telling this apart from a phrase that simply is
        not there keeps the two defects from being reported as one.
        """
        return not self.located and self.candidates > 0

    @property
    def absent(self) -> bool:
        """Whether the phrase does not occur in the prompt at all."""
        return not self.located and not self.candidates

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the leaf."""
        return {
            "path": self.path,
            "phrase": self.phrase,
            "position": self.position,
            "start": self.start,
            "end": self.end,
            "level": self.level,
            "candidates": self.candidates,
        }


@dataclass(frozen=True)
class Alignment:
    """Outcome of locating every phrase of a tree in its source prompt.

    Attributes:
        prompt: The text the phrases were located in.
        tree: The tree they were taken from, kept so that the positional tree
            can be rebuilt with the same shape.
        leaves: One entry per segment of the tree, in traversal order.
    """

    prompt: str
    tree: Mapping[str, Any]
    leaves: tuple[AlignedLeaf, ...]

    @property
    def scored(self) -> tuple[AlignedLeaf, ...]:
        """The phrases the rates are computed over: every non-blank one."""
        return tuple(leaf for leaf in self.leaves if not leaf.blank)

    @property
    def located(self) -> int:
        """How many non-blank phrases were found in the prompt."""
        return sum(leaf.located for leaf in self.scored)

    @property
    def ambiguous(self) -> int:
        """How many non-blank phrases had more than one candidate occurrence."""
        return sum(leaf.ambiguous for leaf in self.scored)

    @property
    def contested(self) -> int:
        """How many phrases occur only where another phrase already sits."""
        return sum(leaf.contested for leaf in self.scored)

    def ordered(self) -> tuple[AlignedLeaf, ...]:
        """Return the located phrases in the order they occur in the prompt."""
        ranked: list[tuple[int, AlignedLeaf]] = []
        for leaf in self.leaves:
            if leaf.position is not None:
                ranked.append((leaf.position, leaf))
        return tuple(leaf for _, leaf in sorted(ranked, key=lambda item: item[0]))

    def reconstruction(self) -> str:
        """Rebuild the prompt by concatenating the located phrases in order."""
        return "".join(leaf.phrase for leaf in self.ordered())

    def reconstructs(self, *, strict: bool = False) -> bool:
        """Report whether the rebuilt text agrees with the prompt.

        Whitespace is removed from both sides before comparing, because the
        space that separates two segments belongs to neither of them and would
        otherwise make the comparison fail on every prompt.

        Args:
            strict: Compare the two strings character for character instead,
                whitespace included.

        Returns:
            ``True`` when the prompt is recovered.
        """
        recovered = self.reconstruction()
        if strict:
            return recovered == self.prompt
        return strip_whitespace(recovered) == strip_whitespace(self.prompt)

    def as_tree(self) -> Any:
        """Rebuild the tree with positional leaves.

        Every segment becomes a ``{"position": ..., "phrase": ...}`` object,
        keeping the shape of the tree it came from. A phrase that could not be
        located carries a null position rather than being dropped, so that the
        artefact does not claim more than was established.

        Returns:
            The tree, with each string replaced by a positional object.
        """
        by_path = {leaf.path: leaf for leaf in self.leaves}
        return _rebuild(self.tree, "", by_path)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the alignment."""
        return {
            "phrases": len(self.scored),
            "located": self.located,
            "ambiguous": self.ambiguous,
            "contested": self.contested,
            "reconstructs": self.reconstructs(),
            "leaves": [leaf.as_dict() for leaf in self.leaves],
        }


def align_phrases(
    prompt: str,
    phrases: Sequence[tuple[str, str]],
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> tuple[AlignedLeaf, ...]:
    """Assign each phrase a span of the prompt, no two spans overlapping.

    Four rules make the outcome deterministic and are applied in this order.
    Every level is exhausted before the next one is tried, so a phrase found
    verbatim is never displaced by one found only after normalisation. Within a
    level the longest phrases claim their span first, since a short phrase is
    often a substring of a long one and would otherwise pierce it. Among the
    spans still free, the one standing in the way of the fewest phrases still
    looking for a home is taken; and where several are equally in the way, the
    leftmost.

    That third rule is what keeps a phrase from being stranded. In ``ababa``
    cut into ``ab`` and ``aba``, the longer phrase occurs at both ends, and
    taking the leftmost would sit across the only place the shorter one fits.
    Counting the obstruction first costs nothing when phrases occur once, which
    is the ordinary case, and settles the periodic ones correctly.

    Args:
        prompt: The text to locate the phrases in.
        phrases: Pairs of path and phrase, in traversal order.
        levels: Normalisations to try, from the most conservative.

    Returns:
        One entry per phrase, in the order they were given.
    """
    projections = {level: project(prompt, level) for level in levels}
    results = [AlignedLeaf(path=path, phrase=phrase) for path, phrase in phrases]

    pending = [order for order, (_, phrase) in enumerate(phrases) if phrase.strip()]
    claimed: list[tuple[int, int]] = []
    outbid: dict[int, int] = {}

    for level in levels:
        projection = projections[level]
        at_level = {
            order: occurrences(projection, project_phrase(phrases[order][1], level))
            for order in pending
        }
        for order in sorted(pending, key=lambda index: (-len(phrases[index][1]), index)):
            path, phrase = phrases[order]
            found = at_level[order]
            if found and order not in outbid:
                outbid[order] = len(found)

            free = [span for span in found if not _overlaps(claimed, span)]
            if not free:
                continue

            rival = [
                span
                for other in pending
                if other != order
                for span in at_level[other]
                if not _overlaps(claimed, span)
            ]
            start, end = min(free, key=lambda span: (_obstruction(rival, span), span))
            bisect.insort(claimed, (start, end))
            pending.remove(order)
            results[order] = AlignedLeaf(
                path=path,
                phrase=phrase,
                start=start,
                end=end,
                level=level,
                candidates=len(found),
            )

    for order, count in outbid.items():
        if results[order].start is None:
            results[order] = replace(results[order], candidates=count)

    spans: list[tuple[int, int, int]] = []
    for order, leaf in enumerate(results):
        if leaf.start is not None and leaf.end is not None:
            spans.append((leaf.start, leaf.end, order))
    for position, (_, _, order) in enumerate(sorted(spans)):
        results[order] = replace(results[order], position=position)

    return tuple(results)


def align_tree(
    prompt: str,
    tree: Mapping[str, Any],
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> Alignment:
    """Locate every phrase of a tree in the prompt it was extracted from.

    Args:
        prompt: The text the tree was built from.
        tree: The tree, already parsed.
        levels: Normalisations to try, from the most conservative.

    Returns:
        The alignment, from which positions and the rebuilt prompt follow.
    """
    return Alignment(
        prompt=prompt,
        tree=tree,
        leaves=align_phrases(prompt, list(iter_leaves(tree)), levels),
    )


def _obstruction(rival: Sequence[tuple[int, int]], span: tuple[int, int]) -> int:
    """Count the places a span would take away from the phrases still waiting.

    Args:
        rival: Spans that phrases without a home could still occupy.
        span: The span under consideration.

    Returns:
        How many of those spans it intersects.
    """
    start, end = span
    return sum(other_start < end and start < other_end for other_start, other_end in rival)


def _overlaps(claimed: Sequence[tuple[int, int]], span: tuple[int, int]) -> bool:
    """Report whether a span intersects any of the spans already claimed.

    Args:
        claimed: Disjoint intervals, sorted.
        span: The interval to test.

    Returns:
        ``True`` when the interval is not free.
    """
    start, end = span
    position = bisect.bisect_left(claimed, (start, start))
    if position < len(claimed) and claimed[position][0] < end:
        return True
    return position > 0 and claimed[position - 1][1] > start


def _rebuild(node: Any, prefix: str, by_path: Mapping[str, AlignedLeaf]) -> Any:
    """Copy a tree, replacing each string with its positional object."""
    if isinstance(node, Mapping):
        return {
            key: _rebuild(value, f"{prefix}.{key}" if prefix else str(key), by_path)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            _rebuild(item, f"{prefix}[{position}]", by_path) for position, item in enumerate(node)
        ]
    if isinstance(node, str):
        leaf = by_path[prefix]
        return {"position": leaf.position, "phrase": leaf.phrase}
    return node
