"""Properties every metric must satisfy, whatever the input.

Curated cases pin down known situations; these checks explore the space around
them and catch the inputs nobody thought to write down.
"""

from __future__ import annotations

import json

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pstparser.evaluation.metrics import (
    coverage_scores,
    field_f1_scores,
    hallucination_rates,
    json_validity_rate,
    normalise_text,
    to_bracket_notation,
    token_f1,
    tokenise,
    tree_edit_distances,
)
from pstparser.pst import LEAF_PATHS, ROOT_KEY, assemble, collect_text, serialise_target

PENALTY = 100.0

segments = st.text(min_size=0, max_size=40)
segment_lists = st.lists(segments, min_size=0, max_size=3)
trees = st.fixed_dictionaries({path: segment_lists for path in LEAF_PATHS}).map(assemble)


@given(segments)
def test_token_f1_of_a_string_with_itself_is_one(text: str) -> None:
    assert token_f1(text, text) == 1.0


@given(segments, segments)
def test_token_f1_stays_within_bounds(left: str, right: str) -> None:
    assert 0.0 <= token_f1(left, right) <= 1.0


@given(segments, segments)
def test_token_f1_is_symmetric(left: str, right: str) -> None:
    assert token_f1(left, right) == token_f1(right, left)


@given(segments)
def test_normalisation_is_idempotent(text: str) -> None:
    once = normalise_text(text)

    assert normalise_text(once) == once


@given(segments)
def test_tokens_carry_no_whitespace(text: str) -> None:
    assert all(token.strip() == token and token for token in tokenise(text))


@given(trees)
def test_assembled_tree_always_declares_every_leaf(tree: dict[str, object]) -> None:
    flat = json.dumps(tree)

    assert flat.count('"main_instruction"') == 1
    assert ROOT_KEY in tree


@given(trees)
def test_collect_text_returns_every_segment(tree: dict[str, object]) -> None:
    collected = collect_text(tree[ROOT_KEY])
    expected = "".join(
        segment
        for leaf in (tree[ROOT_KEY],)
        for segment in _segments(leaf)  # type: ignore[arg-type]
    )

    assert collected == expected


@given(trees)
@settings(max_examples=50)
def test_tree_distance_to_itself_is_zero(tree: dict[str, object]) -> None:
    serialised = serialise_target(tree)

    assert tree_edit_distances([serialised], [serialised], PENALTY) == [0.0]


@given(trees, trees)
@settings(max_examples=50)
def test_tree_distance_is_symmetric(left: dict[str, object], right: dict[str, object]) -> None:
    first = serialise_target(left)
    second = serialise_target(right)

    forward = tree_edit_distances([first], [second], PENALTY)
    backward = tree_edit_distances([second], [first], PENALTY)

    assert forward == backward


@given(trees)
@settings(max_examples=50)
def test_tree_distance_is_never_negative(tree: dict[str, object]) -> None:
    reference = serialise_target(assemble({path: [] for path in LEAF_PATHS}))

    assert tree_edit_distances([serialise_target(tree)], [reference], PENALTY)[0] >= 0.0


@given(trees)
def test_bracket_notation_is_balanced(tree: dict[str, object]) -> None:
    rendered = to_bracket_notation(tree)

    assert rendered.count("{") == rendered.count("}")


@given(trees)
def test_identical_trees_score_one_on_every_field(tree: dict[str, object]) -> None:
    serialised = serialise_target(tree)

    scores = field_f1_scores([serialised], [serialised])

    assert all(value == 1.0 for value in scores.values())


@given(trees)
def test_prediction_drawn_from_the_prompt_never_hallucinates(tree: dict[str, object]) -> None:
    serialised = serialise_target(tree)
    prompt = " ".join(_segments(tree[ROOT_KEY]))
    assume(tokenise(prompt))

    assert hallucination_rates([serialised], [prompt]) == [0.0]


def test_segments_abutting_in_the_prompt_are_reported_as_invented() -> None:
    # Predicted text is reconstructed by joining the segments with a space, so
    # two segments that touch in the prompt yield tokens the prompt never had.
    tree = assemble({**{path: [] for path in LEAF_PATHS}, "main_instruction": ["ab", "cd"]})

    assert hallucination_rates([serialise_target(tree)], ["abcd"]) == [1.0]
    assert hallucination_rates([serialise_target(tree)], ["ab cd"]) == [0.0]


@given(trees)
def test_coverage_stays_within_bounds(tree: dict[str, object]) -> None:
    serialised = serialise_target(tree)

    score = coverage_scores([serialised], ["some arbitrary prompt text"])[0]

    assert 0.0 <= score <= 1.0


@given(st.lists(st.text(max_size=20), max_size=5))
def test_validity_rate_stays_within_bounds(predictions: list[str]) -> None:
    assert 0.0 <= json_validity_rate(predictions) <= 1.0


def _segments(node: object) -> list[str]:
    """Collect the text segments of a subtree, in traversal order."""
    if isinstance(node, dict):
        return [segment for value in node.values() for segment in _segments(value)]
    if isinstance(node, list):
        return [segment for item in node for segment in _segments(item)]
    if isinstance(node, str):
        return [node]
    return []
