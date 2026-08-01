"""The metrics used to score a set of predictions."""

from pstparser.evaluation.metrics._text import (
    flatten,
    generalise_key,
    normalise_text,
    token_f1,
    tokenise,
)
from pstparser.evaluation.metrics.confusion import Confusion, field_confusion
from pstparser.evaluation.metrics.coverage import coverage_scores
from pstparser.evaluation.metrics.field_f1 import field_f1_scores
from pstparser.evaluation.metrics.hallucination import hallucination_rates
from pstparser.evaluation.metrics.tree_distance import to_bracket_notation, tree_edit_distances
from pstparser.evaluation.metrics.validity import is_valid, json_validity_rate

__all__ = [
    "Confusion",
    "coverage_scores",
    "field_confusion",
    "field_f1_scores",
    "flatten",
    "generalise_key",
    "hallucination_rates",
    "is_valid",
    "json_validity_rate",
    "normalise_text",
    "to_bracket_notation",
    "token_f1",
    "tokenise",
    "tree_edit_distances",
]
