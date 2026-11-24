"""Field F1 pooled over several predictions, and the parser macro F1, by hand.

The counts are summed per leaf over the valid predictions before precision and
recall are taken (micro per leaf), tokens are the distinct lowercase words of the
leaf in one prediction, an unparsable prediction is left out, and the macro F1 is
the mean over the leaves with gold support. Every expected value below is worked
out from the token sets written next to it.
"""

from __future__ import annotations

import json

import pytest

from pstparser.empirical.auto import macro_f1
from pstparser.evaluation.metrics.field_f1 import field_f1_scores


def tree(main: str, **context: str) -> str:
    node: dict = {"main_instruction": [main]}
    if context:
        node["context"] = {leaf: [text] for leaf, text in context.items()}
    return json.dumps({"prompt": node})


PAIRS = [
    # main: found {sort, the}, wanted {sort, the, list}       -> overlap 2, predicted 2, gold 3
    # data: found {list, a, b, c}, wanted {a, b, c}            -> overlap 3, predicted 4, gold 3
    (tree("Sort the", data="list a b c"), tree("Sort the list", data="a b c")),
    # main: found {return, the, sum, only}, wanted {return, the, sum} -> 3, 4, 3
    # constraints: found {integers}, wanted {only, integers}   -> 1, 1, 2
    (
        tree("Return the sum only", constraints="integers"),
        tree("Return the sum", constraints="only integers"),
    ),
    # Not JSON: the pair is left out entirely.
    ("{not json", tree("Ignored entirely", data="ignored")),
    # main: a repeated word counts once: found {the, cat}, wanted {the, cat, dog} -> 2, 2, 3
    (tree("the cat"), tree("the cat the dog")),
]


def scores() -> dict:
    predictions, references = zip(*PAIRS, strict=True)
    return {
        name: score.as_dict() for name, score in field_f1_scores(predictions, references).items()
    }


def test_main_instruction_is_pooled_over_the_predictions() -> None:
    main = scores()["prompt.main_instruction"]
    # Pooled: overlap 7, predicted 8, gold 9 -> P 7/8, R 7/9, F1 98/119 = 14/17.
    assert (main["support"], main["predicted"]) == (9, 8)
    assert main["precision"] == pytest.approx(7 / 8)
    assert main["recall"] == pytest.approx(7 / 9)
    assert main["f1"] == pytest.approx(14 / 17)
    # The mean of the per-prediction F1 (0.8, 6/7, 0.8) would be different.
    assert main["f1"] != pytest.approx((0.8 + 6 / 7 + 0.8) / 3)


def test_other_leaves_and_unscored_ones() -> None:
    values = scores()
    assert values["prompt.context.data"]["f1"] == pytest.approx(6 / 7)  # P 3/4, R 1
    assert values["prompt.context.data"]["support"] == 3  # the unparsable pair is not counted
    assert values["prompt.context.constraints"]["f1"] == pytest.approx(2 / 3)  # P 1, R 1/2
    assert values["prompt.context.format"]["f1"] is None
    assert values["prompt.context.format"]["support"] == 0


def test_macro_f1_is_the_mean_over_leaves_with_gold_support() -> None:
    value, leaves = macro_f1(scores())
    assert leaves == [
        "prompt.context.constraints",
        "prompt.context.data",
        "prompt.main_instruction",
    ]
    assert value == pytest.approx((14 / 17 + 6 / 7 + 2 / 3) / 3)


def test_a_predicted_leaf_without_gold_support_does_not_enter_the_macro() -> None:
    predictions = [tree("Do it", format="as json")]
    references = [tree("Do it as json")]
    values = {k: v.as_dict() for k, v in field_f1_scores(predictions, references).items()}
    assert values["prompt.context.format"]["f1"] == 0.0  # predicted 2 tokens, no gold
    value, leaves = macro_f1(values)
    assert leaves == ["prompt.main_instruction"]
    # main: found {do, it}, wanted {do, it, as, json} -> P 1, R 1/2, F1 2/3.
    assert value == pytest.approx(2 / 3)


def test_tokens_ignore_case_and_punctuation() -> None:
    # found {sort, the, list}, wanted {sort, the, list}: punctuation and case do not count.
    values = field_f1_scores([tree("Sort, the list.")], [tree("sort the LIST")])
    assert values["prompt.main_instruction"].f1 == pytest.approx(1.0)
