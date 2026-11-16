"""Character kappa and boundary F1 on the judgement units (M25; 02 §3.5)."""

import pytest

from pstparser.empirical.agreement import (
    AgreementItem,
    boundary_f1,
    char_kappa,
    character_labels,
)
from pstparser.empirical.contracts import SpanAnnotation
from pstparser.empirical.covariates import judgement_mask

CODE = "```\nimport os\ndef task_func(x):\n```"
TEXT = (
    "Sort the list. It must be stable. Return it.\n"
    "The function should output with:\n    list: The sorted list.\n"
    "You should write self-contained code starting with:\n" + CODE
)
RULE = "bcb-markers-v1"


def spans(*cuts: tuple[str, str]) -> tuple[SpanAnnotation, ...]:
    """Spans from (leaf, phrase) pairs, located in TEXT in order."""
    found, cursor = [], 0
    for path, phrase in cuts:
        start = TEXT.index(phrase, cursor)
        found.append(SpanAnnotation(path, start, start + len(phrase), phrase))
        cursor = start + len(phrase)
    return tuple(found)


OUTPUT = "The function should output with:\n    list: The sorted list."
LEAD = "You should write self-contained code starting with:"
BASE = spans(
    ("main_instruction", "Sort the list."),
    ("context.constraints", "It must be stable."),
    ("context.format", "Return it."),
    ("context.format", OUTPUT),
    ("context.constraints", LEAD),
    ("context.data", CODE),
)


def relabel(annotation, index: int, path: str):
    changed = list(annotation)
    span = changed[index]
    changed[index] = SpanAnnotation(path, span.start, span.end, span.phrase)
    return tuple(changed)


def item(second, mask=True) -> AgreementItem:
    return AgreementItem("t", TEXT, BASE, second, judgement_mask(TEXT, RULE) if mask else None)


def test_identical_annotations_give_kappa_and_f1_of_one() -> None:
    assert char_kappa([item(BASE)]) == 1.0
    assert boundary_f1([item(BASE)]) == 1.0
    assert char_kappa([item(BASE, mask=False)]) == 1.0


def test_the_masked_kappa_ignores_a_change_in_a_fixed_section() -> None:
    changed = relabel(BASE, 3, "context.constraints")  # the output formula, relabelled
    assert char_kappa([item(changed)]) == 1.0
    assert boundary_f1([item(changed)]) == 1.0
    # The descriptive measure on every character sees it.
    assert char_kappa([item(changed, mask=False)]) < 1.0


def test_masked_kappa_by_hand_when_a_judgement_sentence_changes_leaf() -> None:
    changed = relabel(BASE, 1, "main_instruction")
    mask = judgement_mask(TEXT, RULE)
    counted = [i for i, c in enumerate(TEXT) if mask[i] and not c.isspace()]
    first = character_labels(TEXT, BASE)
    second = character_labels(TEXT, changed)
    n = len(counted)
    sentence = sum(1 for i in counted if first[i] == "context.constraints")
    observed = (n - sentence) / n
    # Marginals: main / constraints / format in the first, main / format in the second.
    main_1 = sum(1 for i in counted if first[i] == "main_instruction")
    format_1 = sum(1 for i in counted if first[i] == "context.format")
    main_2, format_2 = main_1 + sentence, format_1
    assert main_2 == sum(1 for i in counted if second[i] == "main_instruction")
    expected = (main_1 * main_2 + format_1 * format_2) / (n * n)
    assert char_kappa([item(changed)]) == pytest.approx((observed - expected) / (1 - expected))
    assert boundary_f1([item(changed)]) == 1.0


def test_joining_two_sentences_loses_one_boundary() -> None:
    joined = spans(
        ("main_instruction", "Sort the list. It must be stable."),
        ("context.format", "Return it."),
        ("context.format", OUTPUT),
        ("context.constraints", LEAD),
        ("context.data", CODE),
    )
    # Inside the judgement units the first has boundaries at "It" and "Return",
    # the second only at "Return": F1 = 2 * 1 / (2 + 1).
    assert boundary_f1([item(joined)]) == pytest.approx(2 / 3)


def test_whitespace_is_canonicalised_like_the_renderer() -> None:
    # Moving a separating space from one span to the next changes nothing.
    text = "Do a. Do b."
    first = (
        SpanAnnotation("main_instruction", 0, 5, "Do a."),
        SpanAnnotation("context.format", 6, 11, "Do b."),
    )
    second = (
        SpanAnnotation("main_instruction", 0, 5, "Do a."),
        SpanAnnotation("context.format", 5, 11, " Do b."),
    )
    pair = AgreementItem("w", text, first, second)
    assert char_kappa([pair]) == 1.0
    assert boundary_f1([pair]) == 1.0


def test_items_are_pooled_and_an_empty_mask_gives_none() -> None:
    changed = relabel(BASE, 1, "main_instruction")
    pooled = char_kappa([item(BASE), item(changed)])
    assert pooled is not None and char_kappa([item(changed)]) < pooled < 1.0
    nothing = AgreementItem("t", TEXT, BASE, BASE, tuple([False] * len(TEXT)))
    assert char_kappa([nothing]) is None
    assert boundary_f1([nothing]) is None


def test_a_mask_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="one flag per character"):
        AgreementItem("t", TEXT, BASE, BASE, (True,))
