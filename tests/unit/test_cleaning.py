"""Normalisation of spreadsheet cells into leaf values."""

from __future__ import annotations

import numpy as np
import pytest

from pstparser.data import clean_field


@pytest.mark.parametrize("empty", [None, float("nan"), np.nan, ""])
def test_empty_cell_becomes_empty_list(empty: object) -> None:
    assert clean_field(empty) == []


@pytest.mark.parametrize("empty", [None, float("nan"), ""])
def test_empty_cell_becomes_none_for_scalar_leaf(empty: object) -> None:
    assert clean_field(empty, is_list=False) is None


def test_text_is_wrapped_in_a_single_element_list() -> None:
    assert clean_field("Summarise the code.") == ["Summarise the code."]


def test_surrounding_whitespace_is_preserved() -> None:
    assert clean_field("  padded  ") == ["  padded  "]


def test_bracketed_text_is_parsed_as_a_literal() -> None:
    assert clean_field("['alpha', 'beta']") == ["alpha", "beta"]


def test_bracketed_text_is_parsed_after_stripping() -> None:
    assert clean_field("  ['alpha']  ") == ["alpha"]


def test_unparseable_bracketed_text_falls_back_to_wrapping() -> None:
    value = "[INST] do something [/INST]"

    assert clean_field(value) == [value]


def test_scalar_leaf_keeps_text_unwrapped() -> None:
    assert clean_field("plain", is_list=False) == "plain"


def test_non_text_values_are_returned_unchanged() -> None:
    assert clean_field(42) == 42
