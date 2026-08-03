"""Token agreement between prediction and reference, reported per field.

The tokens a field holds are pooled before anything is compared. Segments are an
implementation detail of how an annotator entered the text: whether one span was
recorded as one segment or as two says nothing about the segmentation being
right, so pairing them by position penalises a correct answer for splitting
differently. What the field was assigned is the question; how it was cut is not.

Precision and recall are computed once per field over the whole population, and
the support is reported beside them, since a perfect score on twenty tokens and
one on three thousand are not the same claim.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pstparser.evaluation.metrics._text import flatten, generalise_key, tokenise
from pstparser.pst.taxonomy import LEAF_PATHS, PATH_SEPARATOR, ROOT_KEY


@dataclass(frozen=True)
class FieldScore:
    """Token agreement for one field, pooled over the scored population.

    Attributes:
        field: Generalised leaf path, such as ``prompt.context.data``.
        precision: Share of the tokens assigned to the field that belonged there.
        recall: Share of the tokens that belonged there which were assigned.
        f1: Their harmonic mean, or ``None`` when neither side holds a token and
            there is nothing to score.
        support: Tokens the reference assigns to the field, summed over examples.
        predicted: Tokens the prediction assigns to it, summed over examples.
    """

    field: str
    precision: float
    recall: float
    f1: float | None
    support: int
    predicted: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the score."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "predicted": self.predicted,
        }


def field_f1_scores(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, FieldScore]:
    """Score every field on the tokens assigned to it.

    A row is emitted for each leaf the taxonomy declares, so a field without
    annotated support is visibly unscored rather than quietly missing, and for
    any further field a prediction invented. Pairs whose prediction or reference
    does not parse are skipped; when nothing is left, no row is emitted at all.

    Args:
        predictions: The raw model outputs.
        references: The expected targets, aligned with ``predictions``.

    Returns:
        A mapping from field path to its score, ordered by field.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    overlap: dict[str, int] = collections.defaultdict(int)
    expected: dict[str, int] = collections.defaultdict(int)
    produced: dict[str, int] = collections.defaultdict(int)
    scored = 0

    for prediction, reference in zip(predictions, references, strict=True):
        try:
            predicted_tree = json.loads(prediction)
            reference_tree = json.loads(reference)
        except json.JSONDecodeError:
            continue

        scored += 1
        predicted_tokens = _tokens_by_field(predicted_tree)
        reference_tokens = _tokens_by_field(reference_tree)

        for path in set(predicted_tokens) | set(reference_tokens):
            found = predicted_tokens.get(path, set())
            wanted = reference_tokens.get(path, set())
            overlap[path] += len(found & wanted)
            produced[path] += len(found)
            expected[path] += len(wanted)

    if not scored:
        return {}

    fields = sorted(set(declared_fields()) | set(expected) | set(produced))
    return {
        field: _score(field, overlap[field], expected[field], produced[field]) for field in fields
    }


def declared_fields() -> tuple[str, ...]:
    """List the field paths of the taxonomy, as they appear once flattened."""
    return tuple(f"{ROOT_KEY}{PATH_SEPARATOR}{path}" for path in LEAF_PATHS)


def _tokens_by_field(tree: Any) -> dict[str, set[str]]:
    """Pool the tokens of every segment of a tree into the field holding it."""
    pooled: dict[str, set[str]] = collections.defaultdict(set)
    for path, text in flatten(tree).items():
        pooled[generalise_key(path)].update(tokenise(text))
    return pooled


def _score(field: str, overlap: int, support: int, predicted: int) -> FieldScore:
    """Turn the pooled counts of one field into its score."""
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / support if support else 0.0

    if not support and not predicted:
        f1: float | None = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return FieldScore(
        field=field,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        predicted=predicted,
    )
