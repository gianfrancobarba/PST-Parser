"""auto-strict-v2: exemption of the omitted starting code block only.

The span of the block comes from the source text (the fixed marker of
BigCodeBench), never from the gold. Every other failure stays as in v1. The C-22
rule reads the share of v1 failures the exemption alone would rescue.
"""

import json

import pytest

from pstparser.empirical.auto import (
    STRICT,
    STRICT_V2,
    auto_annotate,
    c22_primary_policy,
    predicted_spans,
)
from pstparser.empirical.contracts import CanonicalPrompt
from pstparser.empirical.rendering import render_prompt_variants
from pstparser.pst.taxonomy import LEAF_PATHS, assemble

FIRST = "Compute the sum of a list."
MARKER = "You should write self-contained code starting with:"
CODE = "```\nimport os\ndef task_func(x):\n```"
FORMAT = "The function should output with:\n    float: The sum."
TEXT = f"{FIRST}\n{FORMAT}\n{MARKER}\n{CODE}"


def tree(**leaves: list[str]) -> str:
    values = {path: [] for path in LEAF_PATHS}
    for name, phrases in leaves.items():
        values[name.replace("__", ".")] = phrases
    return json.dumps(assemble(values))


FULL = tree(main_instruction=[FIRST], context__format=[FORMAT, MARKER], context__data=[CODE])
NO_CODE = tree(main_instruction=[FIRST], context__format=[FORMAT, MARKER])
NO_CODE_NO_MARKER = tree(main_instruction=[FIRST], context__format=[FORMAT])
NO_CODE_ABSENT = tree(
    main_instruction=[FIRST, "A sentence that is not there."], context__format=[FORMAT, MARKER]
)


def test_a_complete_prediction_is_converted_alike_by_both_policies() -> None:
    v1 = predicted_spans(TEXT, FULL, policy=STRICT)
    v2 = predicted_spans(TEXT, FULL, policy=STRICT_V2)
    assert v1.status == v2.status == "ok" and v1.spans == v2.spans
    assert v2.diagnostics.get("starting_code_exempt", 0) == 0


def test_an_omitted_starting_code_fails_v1_and_becomes_a_source_data_span_in_v2() -> None:
    v1 = predicted_spans(TEXT, NO_CODE, policy=STRICT)
    assert (v1.status, v1.cause) == ("parser_failure", "content_gap")
    assert v1.diagnostics["only_gap_is_starting_code"] == 1
    v2 = predicted_spans(TEXT, NO_CODE, policy=STRICT_V2)
    assert v2.status == "ok" and v2.diagnostics["starting_code_exempt"] == 1
    assert (v2.spans[-1].path, v2.spans[-1].phrase) == ("context.data", CODE)
    plain, _ = render_prompt_variants(CanonicalPrompt("t", TEXT, v2.spans, STRICT_V2))
    assert plain.text == TEXT


@pytest.mark.parametrize(
    ("prediction", "cause"),
    [(NO_CODE_NO_MARKER, "content_gap"), (NO_CODE_ABSENT, "absent_phrase")],
)
def test_any_other_failure_stays_a_failure_in_v2(prediction: str, cause: str) -> None:
    v1 = predicted_spans(TEXT, prediction, policy=STRICT)
    v2 = predicted_spans(TEXT, prediction, policy=STRICT_V2)
    assert (v2.status, v2.cause) == ("parser_failure", cause)
    assert v1.diagnostics.get("only_gap_is_starting_code", 0) == 0


def test_a_text_without_the_starting_code_marker_has_no_exemption() -> None:
    text = f"{FIRST}\n{FORMAT}"
    prediction = tree(main_instruction=[FIRST])
    v2 = predicted_spans(text, prediction, policy=STRICT_V2)
    assert (v2.status, v2.cause) == ("parser_failure", "content_gap")


def _conversion(predictions: list[str], policy: str) -> dict:
    tasks = [f"BigCodeBench/{i}" for i in range(len(predictions))]
    rows = [{"index": i, "prompt": TEXT, "prediction": p} for i, p in enumerate(predictions)]
    sources = [{"task_id": t, "text": TEXT} for t in tasks]
    return auto_annotate(rows, {"tasks": tasks}, sources, policy=policy).s_parse


def test_the_v1_conversion_records_the_share_the_exemption_would_rescue() -> None:
    document = _conversion([FULL, NO_CODE, NO_CODE, NO_CODE_ABSENT], STRICT)
    assert document["failures"] == 3
    assert document["starting_code_only_failures"] == 2
    assert document["starting_code_share_of_failures"] == pytest.approx(2 / 3)
    assert c22_primary_policy(document) == STRICT_V2


def test_the_c22_rule_needs_more_than_half_of_the_failures() -> None:
    half = _conversion([NO_CODE, NO_CODE_ABSENT], STRICT)
    assert half["starting_code_share_of_failures"] == 0.5
    assert c22_primary_policy(half) == STRICT
    none = _conversion([FULL, FULL], STRICT)
    assert none["starting_code_share_of_failures"] is None
    assert c22_primary_policy(none) == STRICT
    with pytest.raises(ValueError, match="auto-strict-v1"):
        c22_primary_policy(_conversion([NO_CODE], STRICT_V2))
