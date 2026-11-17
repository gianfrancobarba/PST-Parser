"""Registered sentence cuts in the gold lint (Q-16, decided 17/11/2026; R-TRA-03).

A reviewer may cut a judgement unit (rule G3 or B6) at a space, for example to
separate the items of a list written on one line, when the row's notes record
the cut as ``cut@<offset>``: the offset of the first character after the cut.
Every other cut of a sentence or of a fixed section stays a blocking defect.
"""

import dataclasses

from test_covariates import BCB, BCB_TEXT

from pstparser.empirical.annotations import canonicalize, lint, propose_bigcodebench
from pstparser.empirical.contracts import SpanAnnotation


def cut_at(index: int, offset: int, notes: str):
    """Split the draft span ``index`` at ``offset``; the second piece goes to constraints."""
    row = propose_bigcodebench(BCB)
    spans = sorted(row.spans, key=lambda span: span.start)
    span = spans[index]
    end = offset
    while BCB_TEXT[end - 1].isspace():
        end -= 1
    pieces = (
        SpanAnnotation(span.path, span.start, end, BCB_TEXT[span.start : end]),
        SpanAnnotation("context.constraints", offset, span.end, BCB_TEXT[offset : span.end]),
    )
    return dataclasses.replace(
        row, spans=(*spans[:index], *pieces, *spans[index + 1 :]), notes=notes
    )


SENTENCE = "It must not modify the input, e.g. by sorting it."
CUT = BCB_TEXT.index("e.g. by sorting")  # inside a G3 judgement unit, after a space


def test_a_registered_cut_inside_a_judgement_unit_is_accepted() -> None:
    row = cut_at(1, CUT, f"G1 cut@{CUT}: the clause after the comma is a constraint.")
    assert lint(BCB, row) == []
    assert len(canonicalize([BCB], [row], require_status="draft")) == 1


def test_an_unregistered_cut_is_still_a_defect() -> None:
    row = cut_at(1, CUT, "")
    problems = lint(BCB, row)
    assert sum("cuts a sentence or a fixed section" in problem for problem in problems) == 2


def test_a_registered_cut_must_be_at_a_space() -> None:
    inside_a_word = CUT + 2
    row = cut_at(1, inside_a_word, f"cut@{inside_a_word}")
    assert any(f"cut@{inside_a_word}" in problem for problem in lint(BCB, row))


def test_the_first_sentence_and_fixed_sections_cannot_be_cut() -> None:
    first = BCB_TEXT.index("sum of a list")
    row = cut_at(0, first, f"cut@{first}")
    assert any(f"cut@{first}" in problem for problem in lint(BCB, row))
    output = BCB_TEXT.index("float: The sum")
    spans = sorted(propose_bigcodebench(BCB).spans, key=lambda span: span.start)
    index = next(i for i, span in enumerate(spans) if span.start < output < span.end)
    row = cut_at(index, output, f"cut@{output}")
    assert any(f"cut@{output}" in problem for problem in lint(BCB, row))
