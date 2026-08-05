"""Normalisation of spreadsheet cells into leaf values."""

from __future__ import annotations

import numpy as np
import pytest

from pstparser.data import SEGMENT_SEPARATOR, clean_field


@pytest.mark.parametrize("empty", [None, float("nan"), np.nan, ""])
def test_empty_cell_holds_no_segment(empty: object) -> None:
    assert clean_field(empty) == []


def test_text_becomes_one_segment() -> None:
    assert clean_field("Summarise the code.") == ["Summarise the code."]


def test_surrounding_whitespace_is_dropped() -> None:
    # The space around a segment sits between the segments, not inside one.
    assert clean_field("  padded  ") == ["padded"]


def test_a_separator_splits_the_cell_into_segments() -> None:
    assert clean_field("class A: pass<sep>class B: pass") == ["class A: pass", "class B: pass"]


def test_segments_are_trimmed_around_the_separator() -> None:
    assert clean_field("first part <sep> second part") == ["first part", "second part"]


def test_several_separators_yield_several_segments() -> None:
    assert clean_field("a<sep>b<sep>c") == ["a", "b", "c"]


def test_an_empty_part_is_not_a_segment() -> None:
    assert clean_field("a<sep><sep>  <sep>b") == ["a", "b"]


def test_a_cell_that_is_only_separators_holds_nothing() -> None:
    assert clean_field("<sep> <sep>") == []


def test_the_separator_never_reaches_the_target() -> None:
    segments = clean_field("left<sep>right")

    assert all(SEGMENT_SEPARATOR not in segment for segment in segments)


def test_bracketed_text_is_kept_as_written() -> None:
    # It reads like a Python list, and a prompt may well contain one. Parsing it
    # would silently replace the extracted span with something the prompt never
    # held, and would lose the brackets and quotes on the way.
    assert clean_field("['alpha', 'beta']") == ["['alpha', 'beta']"]


def test_text_that_only_opens_a_bracket_is_kept_as_written() -> None:
    value = "[INST] do something [/INST]"

    assert clean_field(value) == [value]


def test_a_value_that_is_not_text_is_rendered_as_text() -> None:
    # Rendering it keeps the contract; whether it belongs to the prompt is for
    # the integrity check to decide.
    assert clean_field(42) == ["42"]


def test_the_separator_is_configurable() -> None:
    assert clean_field("a||b", separator="||") == ["a", "b"]
