"""Which node a token was assigned to, against the node it belonged to.

The matrix is built over the tokens of the prompt, not over whole field values.
Both trees are located in the prompt first, so every token carries the label the
reference gave it and the label the prediction gave it, and the pair is counted.
Comparing field values instead would only ever see the segments that survived
verbatim, and would say nothing about the errors that matter most here: those of
boundary, where a segment is cut one clause too early or too late.

A token that one side left out is counted under its own label, so the matrix
distinguishes assigning a token to the wrong node from not assigning it at all.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from sklearn.metrics import confusion_matrix

from pstparser.evaluation.metrics._text import generalise_key, tokenise_spans
from pstparser.pst.alignment import DEFAULT_LEVELS, Alignment, MatchLevel, align_tree

#: Label given to a prompt token that a side placed in no node at all.
UNASSIGNED: Final = "none"


@dataclass(frozen=True)
class Confusion:
    """Counts of predicted against expected node assignments.

    Attributes:
        matrix: Square matrix whose rows are expected nodes and columns
            predicted nodes.
        labels: Node paths, in the order used by the matrix.
    """

    matrix: Any
    labels: list[str]

    def top_confusions(self, limit: int) -> list[tuple[str, str, int]]:
        """List the most frequent misassignments.

        Args:
            limit: Maximum number of entries returned.

        Returns:
            Triples of expected node, predicted node and count, ordered by
            decreasing count. Correct assignments are excluded.
        """
        if self.matrix.size == 0:
            return []

        off_diagonal = np.array(self.matrix, copy=True)
        np.fill_diagonal(off_diagonal, 0)

        flat_order = np.argsort(off_diagonal, axis=None)[::-1][:limit]
        confusions = []
        for position in flat_order:
            row, column = np.unravel_index(position, off_diagonal.shape)
            count = int(off_diagonal[row, column])
            if count > 0:
                confusions.append((self.labels[row], self.labels[column], count))
        return confusions

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the matrix."""
        return {
            "labels": list(self.labels),
            "matrix": [[int(value) for value in row] for row in self.matrix],
        }


def token_confusion(
    predictions: Sequence[str],
    references: Sequence[str],
    prompts: Sequence[str],
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> Confusion:
    """Count, for every prompt token, where it went against where it belonged.

    Tokens the prediction invented do not appear: the matrix is indexed by the
    tokens of the prompt, so what was added rather than misplaced is the subject
    of the hallucination rate instead.

    Args:
        predictions: The raw model outputs.
        references: The expected targets, aligned with ``predictions``.
        prompts: The source prompts, aligned with ``predictions``.
        levels: Normalisations to try when locating a phrase.

    Returns:
        The matrix together with the node labels it is indexed by. Both are
        empty when there is no token to count.

    Raises:
        ValueError: If the sequences have different lengths.
    """
    expected: list[str] = []
    predicted: list[str] = []
    observed: set[str] = set()

    for prediction, reference, prompt in zip(predictions, references, prompts, strict=True):
        try:
            predicted_tree = json.loads(prediction)
            reference_tree = json.loads(reference)
        except json.JSONDecodeError:
            continue

        predicted_spans = _labelled_spans(align_tree(prompt, predicted_tree, levels))
        reference_spans = _labelled_spans(align_tree(prompt, reference_tree, levels))

        for _, start, end in tokenise_spans(prompt):
            middle = (start + end) // 2
            truth = _label_at(reference_spans, middle)
            guess = _label_at(predicted_spans, middle)
            expected.append(truth)
            predicted.append(guess)
            observed.update((truth, guess))

    if not expected:
        return Confusion(matrix=np.zeros((0, 0), dtype=int), labels=[])

    ordered = sorted(observed - {UNASSIGNED})
    if UNASSIGNED in observed:
        ordered.append(UNASSIGNED)

    return Confusion(
        matrix=confusion_matrix(expected, predicted, labels=ordered),
        labels=ordered,
    )


def _labelled_spans(alignment: Alignment) -> list[tuple[int, int, str]]:
    """List the spans a tree claims in its prompt, each under its node."""
    spans = []
    for leaf in alignment.leaves:
        if leaf.start is not None and leaf.end is not None:
            spans.append((leaf.start, leaf.end, generalise_key(leaf.path)))
    return sorted(spans)


def _label_at(spans: Sequence[tuple[int, int, str]], position: int) -> str:
    """Report the node covering a position, or that no node covers it.

    A token is attributed to the span containing its midpoint. The spans do not
    overlap, so at most one can contain it, and the midpoint keeps the answer
    unambiguous when a segment boundary falls inside a token.
    """
    for start, end, label in spans:
        if start <= position < end:
            return label
        if start > position:
            break
    return UNASSIGNED
