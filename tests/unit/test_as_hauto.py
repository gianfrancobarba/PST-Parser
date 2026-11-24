"""The H-auto variant is the hierarchical rendering of the parser's spans, relabelled.

H-auto and H-gold share the renderer: only the spans differ (parser against gold).
Relabelling must keep the prompt bytes, its hashes and every other field, and
must refuse any variant that is not the hierarchical one.
"""

from dataclasses import fields

import pytest

from pstparser.empirical.auto import as_hauto
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants

TEXT = "Sort the list.\n\ndef task_func(items):\n    pass\n"
CUT = TEXT.index("def")
PROMPT = CanonicalPrompt(
    "BigCodeBench/1",
    TEXT,
    (
        SpanAnnotation("main_instruction", 0, CUT - 2, TEXT[: CUT - 2]),
        SpanAnnotation("context.data", CUT, len(TEXT) - 1, TEXT[CUT:-1]),
    ),
    "gold-v2.0",
)


def test_only_the_condition_changes() -> None:
    _, hierarchical = render_prompt_variants(PROMPT)
    auto = as_hauto(hierarchical)
    assert auto.condition == "H-auto"
    for field in fields(hierarchical):
        if field.name != "condition":
            assert getattr(auto, field.name) == getattr(hierarchical, field.name), field.name


def test_only_the_hierarchical_variant_becomes_h_auto() -> None:
    text_variant, _ = render_prompt_variants(PROMPT)
    with pytest.raises(ValueError, match="hierarchical"):
        as_hauto(text_variant)
    with pytest.raises(ValueError, match="hierarchical"):
        as_hauto(render_envelope_variant(PROMPT))
