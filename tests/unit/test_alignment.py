"""Locating extracted phrases in the prompt they were taken from."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pstparser.pst import (
    LEAF_PATHS,
    ROOT_KEY,
    align_phrases,
    align_tree,
    assemble,
    iter_leaves,
    occurrences,
    project,
    project_phrase,
    strip_whitespace,
)


def tree(**leaves: list[str]) -> dict[str, object]:
    """Assemble a tree, leaving every undeclared leaf empty."""
    values: dict[str, object] = {path: [] for path in LEAF_PATHS}
    values.update(leaves)
    return assemble(values)


# --------------------------------------------------------------------------- #
# Traversal
# --------------------------------------------------------------------------- #


def test_iter_leaves_names_every_segment() -> None:
    paths = dict(iter_leaves(tree(main_instruction=["one", "two"])))

    assert paths["prompt.main_instruction[0]"] == "one"
    assert paths["prompt.main_instruction[1]"] == "two"
    assert "prompt.context.data" not in paths


def test_iter_leaves_skips_non_text_nodes() -> None:
    assert list(iter_leaves({"a": None, "b": [1, "kept", 2.5]})) == [("b[1]", "kept")]


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #


def test_exact_projection_is_the_text_itself() -> None:
    projection = project("a b", "exact")

    assert projection.text == "a b"
    assert projection.offsets == (0, 1, 2)


def test_collapsed_projection_reduces_a_run_to_one_space() -> None:
    projection = project("a \n\t b", "collapsed")

    assert projection.text == "a b"
    assert projection.offsets == (0, 1, 5)


def test_stripped_projection_drops_whitespace_and_keeps_offsets() -> None:
    projection = project("a \n b", "stripped")

    assert projection.text == "ab"
    assert projection.offsets == (0, 4)


def test_phrase_is_normalised_the_way_the_text_is() -> None:
    assert project_phrase(" a \n b ", "exact") == " a \n b "
    assert project_phrase(" a \n b ", "collapsed") == "a b"
    assert project_phrase(" a \n b ", "stripped") == "ab"


def test_span_maps_a_match_back_onto_the_original() -> None:
    projection = project("a \n b", "stripped")

    assert projection.span(0, 2) == (0, 5)


def test_an_empty_match_has_no_span() -> None:
    with pytest.raises(ValueError, match="empty match"):
        project("abc", "exact").span(0, 0)


def test_overlapping_occurrences_are_all_reported() -> None:
    found = occurrences(project("aaa", "exact"), "aa")

    assert found == ((0, 2), (1, 3))


def test_an_empty_needle_occurs_nowhere() -> None:
    assert occurrences(project("abc", "exact"), "") == ()


def test_strip_whitespace_removes_every_kind() -> None:
    assert strip_whitespace(" a\tb\nc\xa0d ") == "abcd"


# --------------------------------------------------------------------------- #
# The alignment rules
# --------------------------------------------------------------------------- #


def test_a_phrase_is_located_at_its_span() -> None:
    prompt = "Summarise the code. Answer as JSON."
    aligned = align_tree(prompt, tree(main_instruction=["Summarise the code."]))

    leaf = aligned.scored[0]
    assert (leaf.start, leaf.end) == (0, 19)
    assert leaf.position == 0
    assert leaf.level == "exact"
    assert leaf.candidates == 1


def test_position_follows_the_prompt_not_the_taxonomy() -> None:
    # The data node comes after main_instruction in the taxonomy, and before it
    # in the prompt. Position must follow the prompt.
    prompt = "def add(a, b): pass Summarise it."
    aligned = align_tree(
        prompt,
        tree(main_instruction=["Summarise it."], **{"context.data": ["def add(a, b): pass "]}),
    )

    assert [leaf.phrase for leaf in aligned.ordered()] == [
        "def add(a, b): pass ",
        "Summarise it.",
    ]
    assert aligned.reconstructs()


def test_a_repeated_phrase_claims_two_distinct_spans() -> None:
    aligned = align_tree("run it run it", tree(main_instruction=["run it", " run it"]))

    spans = [(leaf.start, leaf.end) for leaf in aligned.scored]
    assert spans == [(0, 6), (6, 13)]
    assert aligned.ambiguous == 1


def test_the_longest_phrase_claims_its_span_first() -> None:
    # "sh" is a substring of "change.sh": taking it first would pierce the
    # longer span and leave the longer phrase unlocated.
    aligned = align_tree(
        "change.sh and sh",
        tree(main_instruction=["sh"], **{"context.data": ["change.sh"]}),
    )

    assert aligned.located == 2
    by_path = {leaf.path: (leaf.start, leaf.end) for leaf in aligned.scored}
    assert by_path["prompt.context.data[0]"] == (0, 9)
    assert by_path["prompt.main_instruction[0]"] == (14, 16)


def test_an_exact_match_is_preferred_to_a_normalised_one() -> None:
    # The phrase occurs verbatim later in the prompt; a whitespace-tolerant
    # match would have claimed the earlier occurrence instead.
    aligned = align_tree("a  b then a b", tree(main_instruction=["a b"]))

    leaf = aligned.scored[0]
    assert leaf.level == "exact"
    assert (leaf.start, leaf.end) == (10, 13)


def test_a_run_of_whitespace_matching_another_is_recovered_when_collapsed() -> None:
    aligned = align_tree("open\n    <svg>", tree(**{"context.data": ["open    <svg>"]}))

    leaf = aligned.scored[0]
    assert leaf.level == "collapsed"
    assert (leaf.start, leaf.end) == (0, 14)


def test_whitespace_the_model_dropped_is_recovered_when_stripped() -> None:
    # Collapsing cannot help here: the phrase has no separator where the prompt
    # has one, so only removing whitespace altogether brings the two together.
    aligned = align_tree("open <svg>", tree(**{"context.data": ["open<svg>"]}))

    leaf = aligned.scored[0]
    assert leaf.level == "stripped"
    assert leaf.located


def test_a_phrase_absent_from_the_prompt_is_left_unlocated() -> None:
    # A character introduced by hand, as happens when an annotation is edited.
    aligned = align_tree("wait for it...", tree(main_instruction=["wait for it…"]))

    leaf = aligned.scored[0]
    assert leaf.position is None
    assert leaf.start is None
    assert leaf.level is None
    assert leaf.candidates == 0
    assert aligned.located == 0
    assert not aligned.reconstructs()


def test_a_blank_phrase_is_not_scored() -> None:
    aligned = align_tree("some text", tree(main_instruction=["some text", "   "]))

    assert len(aligned.leaves) == 2
    assert len(aligned.scored) == 1
    assert aligned.located == 1


def test_spans_never_overlap() -> None:
    aligned = align_tree("abcabc", tree(main_instruction=["abc", "abc", "bca"]))

    spans = sorted((leaf.start, leaf.end) for leaf in aligned.scored if leaf.located)
    assert all(left[1] <= right[0] for left, right in itertools.pairwise(spans))


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #


def test_reconstruction_recovers_a_fully_segmented_prompt() -> None:
    prompt = "You are a bot. Summarise this. Answer as JSON."
    aligned = align_tree(
        prompt,
        tree(
            main_instruction=[" Summarise this."],
            **{"context.role": ["You are a bot."], "context.format": [" Answer as JSON."]},
        ),
    )

    assert aligned.reconstruction() == prompt
    assert aligned.reconstructs()
    assert aligned.reconstructs(strict=True)


def test_reconstruction_ignores_the_whitespace_between_segments() -> None:
    aligned = align_tree("first.\n\nsecond.", tree(main_instruction=["first.", "second."]))

    assert aligned.reconstructs()
    assert not aligned.reconstructs(strict=True)


def test_a_missing_segment_breaks_the_reconstruction() -> None:
    aligned = align_tree("first. second.", tree(main_instruction=["first."]))

    assert aligned.located == 1
    assert not aligned.reconstructs()


# --------------------------------------------------------------------------- #
# The positional artefact
# --------------------------------------------------------------------------- #


def test_the_positional_tree_keeps_the_shape_of_its_source() -> None:
    aligned = align_tree("do this now", tree(main_instruction=["do this"], **{"context.data": []}))

    rebuilt = aligned.as_tree()[ROOT_KEY]
    assert rebuilt["main_instruction"] == [{"position": 0, "phrase": "do this"}]
    assert rebuilt["context"]["data"] == []
    assert list(rebuilt) == ["main_instruction", "context", "examples", "reasoning"]


def test_an_unlocated_phrase_keeps_a_null_position() -> None:
    aligned = align_tree("nothing here", tree(main_instruction=["absent"]))

    assert aligned.as_tree()[ROOT_KEY]["main_instruction"] == [
        {"position": None, "phrase": "absent"}
    ]


def test_the_alignment_serialises() -> None:
    aligned = align_tree("do this", tree(main_instruction=["do this"]))

    payload = aligned.as_dict()
    assert payload["phrases"] == 1
    assert payload["located"] == 1
    assert payload["ambiguous"] == 0
    assert payload["reconstructs"] is True
    assert payload["leaves"][0]["path"] == "prompt.main_instruction[0]"


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #

slices = st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=6)


@given(slices)
@settings(max_examples=200)
def test_every_part_of_a_partition_is_accounted_for(parts: list[str]) -> None:
    # Each part is a slice of the prompt, so none of them can be missing from
    # it. A part the rules cannot place is reported as contested; what must
    # never happen is a part being called absent, or given someone else's span.
    prompt = "".join(parts)
    aligned = align_tree(prompt, tree(main_instruction=parts))

    assert not any(leaf.absent for leaf in aligned.scored)
    assert aligned.located + aligned.contested == len(aligned.scored)


@given(slices)
@settings(max_examples=200)
def test_a_partition_that_places_fully_reconstructs_the_prompt(parts: list[str]) -> None:
    prompt = "".join(parts)
    aligned = align_tree(prompt, tree(main_instruction=parts))

    if aligned.located == len(aligned.scored):
        assert aligned.reconstructs()


def test_a_periodic_prompt_can_leave_a_phrase_without_a_home() -> None:
    # "aba" occurs at both ends, and either choice sits across one of the two
    # places "ab" could go. Choosing between them is the assignment problem the
    # rules only approximate, so the outcome is reported rather than forced.
    aligned = align_tree("abababa", tree(main_instruction=["ab", "aba", "ba"]))

    assert aligned.contested == 1
    assert aligned.located == 2
    assert not aligned.reconstructs()


@given(slices)
@settings(max_examples=200)
def test_positions_rank_the_located_phrases(parts: list[str]) -> None:
    aligned = align_tree("".join(parts), tree(main_instruction=parts))

    positions = [leaf.position for leaf in aligned.leaves if leaf.located]
    assert sorted(positions) == list(range(len(positions)))  # type: ignore[type-var]


@given(slices)
@settings(max_examples=200)
def test_claimed_spans_are_disjoint(parts: list[str]) -> None:
    aligned = align_tree("".join(parts), tree(main_instruction=parts))

    spans = sorted((leaf.start, leaf.end) for leaf in aligned.leaves if leaf.located)
    assert all(left[1] <= right[0] for left, right in itertools.pairwise(spans))


@given(st.text(max_size=30), st.text(min_size=1, max_size=10))
def test_a_phrase_the_prompt_does_not_hold_is_never_located(prompt: str, phrase: str) -> None:
    aligned = align_tree(prompt, tree(main_instruction=[phrase]))

    if strip_whitespace(phrase) and strip_whitespace(phrase) not in strip_whitespace(prompt):
        assert aligned.located == 0


@given(slices)
@settings(max_examples=100)
def test_aligning_twice_gives_the_same_answer(parts: list[str]) -> None:
    prompt = "".join(parts)
    phrases = [(f"leaf[{order}]", part) for order, part in enumerate(parts)]

    assert align_phrases(prompt, phrases) == align_phrases(prompt, phrases)


def test_projection_offsets_cover_every_kept_character() -> None:
    text = "a \n b\tc"
    for level in ("exact", "collapsed", "stripped"):
        projection = project(text, level)  # type: ignore[arg-type]
        assert len(projection.offsets) == len(projection.text)
        assert list(projection.offsets) == sorted(projection.offsets)
