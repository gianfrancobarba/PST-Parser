"""The metrics used to score a set of predictions."""

from pstparser.evaluation.metrics._text import (
    flatten,
    generalise_key,
    normalise_text,
    token_f1,
    tokenise,
    tokenise_spans,
)
from pstparser.evaluation.metrics.confusion import UNASSIGNED, Confusion, token_confusion
from pstparser.evaluation.metrics.coverage import coverage_scores
from pstparser.evaluation.metrics.field_f1 import FieldScore, declared_fields, field_f1_scores
from pstparser.evaluation.metrics.hallucination import hallucination_rates
from pstparser.evaluation.metrics.reconstruction import (
    ReconstructionScores,
    align_predictions,
    reconstruction_scores,
    score_alignments,
)
from pstparser.evaluation.metrics.tree_distance import to_bracket_notation, tree_edit_distances
from pstparser.evaluation.metrics.validity import (
    is_valid,
    json_validity_rate,
    parse_rate,
    parses,
)

__all__ = [
    "UNASSIGNED",
    "Confusion",
    "FieldScore",
    "ReconstructionScores",
    "align_predictions",
    "coverage_scores",
    "declared_fields",
    "field_f1_scores",
    "flatten",
    "generalise_key",
    "hallucination_rates",
    "is_valid",
    "json_validity_rate",
    "normalise_text",
    "parse_rate",
    "parses",
    "reconstruction_scores",
    "score_alignments",
    "to_bracket_notation",
    "token_confusion",
    "token_f1",
    "tokenise",
    "tokenise_spans",
    "tree_edit_distances",
]
