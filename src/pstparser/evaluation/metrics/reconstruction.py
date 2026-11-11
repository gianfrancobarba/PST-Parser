"""How much of the prompt the tree gives back, and how well it was located.

The framework is argued on the claim that a prompt survives its decomposition:
collect the leaves, order them by position, and the original text returns. The
coverage score is only a proxy for that claim, since it reduces a prediction to a
bag of tokens and therefore says which words were placed but never in which
order. These rates test the claim itself.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pstparser.pst.alignment import DEFAULT_LEVELS, Alignment, MatchLevel, align_tree


@dataclass(frozen=True)
class ReconstructionScores:
    """Outcome of locating a set of trees in the prompts they came from.

    Attributes:
        total: Trees examined, whether or not they could be read.
        parsed: Trees that could be read.
        reconstructed: Trees whose ordered leaves give the prompt back.
        phrases: Non-blank phrases examined.
        located: Phrases that were found in their prompt.
        ambiguous: Phrases that had more than one candidate occurrence.
        blank: Phrases holding no character other than whitespace.
        per_example: Whether each tree reconstructs, or ``None`` when it could
            not be read, in input order.
    """

    total: int
    parsed: int
    reconstructed: int
    phrases: int
    located: int
    ambiguous: int
    blank: int
    per_example: list[bool | None] = field(default_factory=list)
    accounted_for: int = 0
    strict_reconstructed: int = 0
    per_example_accounted_for: list[bool | None] = field(default_factory=list)
    per_example_strict: list[bool | None] = field(default_factory=list)

    @property
    def exact_reconstruction_rate(self) -> float | None:
        """Share of trees that give their prompt back, over every tree.

        A tree that cannot be read counts against the rate: an output that does
        not parse has not reconstructed anything.
        """
        return self.reconstructed / self.total if self.total else None

    @property
    def parsed_reconstruction_rate(self) -> float | None:
        """Share of trees that give their prompt back, over those that parse."""
        return self.reconstructed / self.parsed if self.parsed else None

    @property
    def alignment_rate(self) -> float | None:
        """Share of non-blank phrases that were located in their prompt.

        Pooled over every phrase rather than averaged per tree, so that a tree
        holding one phrase does not weigh as much as one holding thirty.
        """
        return self.located / self.phrases if self.phrases else None

    @property
    def ambiguity_rate(self) -> float | None:
        """Share of non-blank phrases that occurred more than once.

        Counted at the normalisation level where the phrase was placed, since
        that is the ambiguity the disambiguation rule actually had to settle.
        """
        return self.ambiguous / self.phrases if self.phrases else None

    @property
    def all_leaves_accounted_for_rate(self) -> float | None:
        """Share with every extracted string placed, over all predictions."""
        return self.accounted_for / self.total if self.total else None

    @property
    def strict_full_reconstruction_rate(self) -> float | None:
        """Share with complete exact reconstruction, over all predictions."""
        return self.strict_reconstructed / self.total if self.total else None

    def as_dict(self, *, version: str = "1") -> dict[str, Any]:
        """Serialise unchanged legacy v1 or explicitly named stronger v2 metrics.

        V2 still includes legacy aliases so consumers can migrate deliberately.
        Neither version is a schema-validity test: score that separately.
        """
        if version not in {"1", "2"}:
            raise ValueError(f"Unsupported reconstruction metric version: {version}")
        result: dict[str, Any] = {
            "total": self.total,
            "parsed": self.parsed,
            "reconstructed": self.reconstructed,
            "phrases": self.phrases,
            "located": self.located,
            "ambiguous": self.ambiguous,
            "blank": self.blank,
            "exact_reconstruction_rate": self.exact_reconstruction_rate,
            "parsed_reconstruction_rate": self.parsed_reconstruction_rate,
            "alignment_rate": self.alignment_rate,
            "ambiguity_rate": self.ambiguity_rate,
        }
        if version == "2":
            result.update(
                {
                    "metric_version": "2",
                    "legacy_metric_version": "1",
                    "aligned_subset_reconstruction_normalized": self.reconstructed,
                    "aligned_subset_reconstruction_normalized_rate": self.exact_reconstruction_rate,
                    "all_leaves_accounted_for": self.accounted_for,
                    "all_leaves_accounted_for_rate": self.all_leaves_accounted_for_rate,
                    "strict_full_reconstruction": self.strict_reconstructed,
                    "strict_full_reconstruction_rate": self.strict_full_reconstruction_rate,
                }
            )
        return result


def align_predictions(
    trees: Sequence[str],
    prompts: Sequence[str],
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> list[Alignment | None]:
    """Locate the phrases of every serialised tree in its prompt.

    Args:
        trees: Serialised trees, either predicted or expected.
        prompts: The source prompts, aligned with ``trees``.
        levels: Normalisations to try, from the most conservative.

    Returns:
        One alignment per input, or ``None`` where the tree could not be read.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    alignments: list[Alignment | None] = []
    for tree, prompt in zip(trees, prompts, strict=True):
        try:
            parsed = json.loads(tree)
        except json.JSONDecodeError:
            alignments.append(None)
            continue

        if not isinstance(parsed, dict):
            alignments.append(None)
            continue

        alignments.append(align_tree(prompt, parsed, levels))
    return alignments


def score_alignments(alignments: Sequence[Alignment | None]) -> ReconstructionScores:
    """Aggregate a set of alignments into rates.

    Args:
        alignments: One alignment per tree, ``None`` where it could not be read.

    Returns:
        The counts and the rates derived from them.
    """
    parsed = 0
    reconstructed = 0
    phrases = 0
    located = 0
    ambiguous = 0
    blank = 0
    per_example: list[bool | None] = []
    per_example_accounted_for: list[bool | None] = []
    per_example_strict: list[bool | None] = []

    for alignment in alignments:
        if alignment is None:
            per_example.append(None)
            per_example_accounted_for.append(None)
            per_example_strict.append(None)
            continue

        parsed += 1
        scored = alignment.scored
        phrases += len(scored)
        located += alignment.located
        ambiguous += alignment.ambiguous
        blank += len(alignment.leaves) - len(scored)

        recovered = alignment.reconstructs()
        reconstructed += recovered
        per_example.append(recovered)
        per_example_accounted_for.append(alignment.all_leaves_accounted_for())
        per_example_strict.append(alignment.strict_full_reconstruction())

    return ReconstructionScores(
        total=len(alignments),
        parsed=parsed,
        reconstructed=reconstructed,
        phrases=phrases,
        located=located,
        ambiguous=ambiguous,
        blank=blank,
        per_example=per_example,
        accounted_for=sum(value is True for value in per_example_accounted_for),
        strict_reconstructed=sum(value is True for value in per_example_strict),
        per_example_accounted_for=per_example_accounted_for,
        per_example_strict=per_example_strict,
    )


def reconstruction_scores(
    trees: Sequence[str],
    prompts: Sequence[str],
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> ReconstructionScores:
    """Locate every tree in its prompt and report how well the prompt returns.

    Args:
        trees: Serialised trees, either predicted or expected.
        prompts: The source prompts, aligned with ``trees``.
        levels: Normalisations to try, from the most conservative.

    Returns:
        The counts and the rates derived from them.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    return score_alignments(align_predictions(trees, prompts, levels))
