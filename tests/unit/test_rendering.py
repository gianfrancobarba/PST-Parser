"""Content preservation and rejection checks for downstream prompt variants."""

import hashlib
import json

import pytest

from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.rendering import render_prompt_variants
from pstparser.evaluation.metrics.reconstruction import reconstruction_scores
from pstparser.pst.taxonomy import LEAF_PATHS


def canonical(text: str, *spans: SpanAnnotation) -> CanonicalPrompt:
    return CanonicalPrompt("example/1", text, spans, "review-1")


def test_renderer_preserves_code_unicode_escaping_and_separator_policy() -> None:
    text = '\r\nReturn JSON.\n\ndef f():\r\n    return "città\\n"\r\n'
    instruction = "Return JSON."
    code_start = text.index("def")
    code = text[code_start:-2]
    prompt = canonical(
        text,
        SpanAnnotation("context.data", code_start, len(text) - 2, code),
        SpanAnnotation("main_instruction", 2, 2 + len(instruction), instruction),
    )
    traditional, hierarchical = render_prompt_variants(prompt)
    assert traditional.text == text
    parsed = json.loads(hierarchical.text)["prompt"]
    assert parsed["main_instruction"] == [{"position": 0, "phrase": text[:code_start]}]
    assert parsed["context"]["data"] == [{"position": 1, "phrase": text[code_start:]}]
    assert parsed["context"]["constraints"] == []
    assert parsed["reasoning"]["paths"] == []
    recovered = parsed["main_instruction"][0]["phrase"] + parsed["context"]["data"][0]["phrase"]
    assert recovered == text
    assert traditional.prompt_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert hierarchical.source_hash == traditional.source_hash
    assert hierarchical.prompt_hash == hashlib.sha256(hierarchical.text.encode("utf-8")).hexdigest()
    assert render_prompt_variants(prompt) == (traditional, hierarchical)
    reordered = canonical(text, *reversed(prompt.annotations))
    assert render_prompt_variants(reordered) == (traditional, hierarchical)


@pytest.mark.parametrize(
    "spans, message",
    [
        ([], "Missing annotations"),
        ([SpanAnnotation("main_instruction", 0, 3, "bad")], "differs"),
        ([SpanAnnotation("main_instruction", 0, 3, "one")], "Unannotated"),
        ([SpanAnnotation("unknown", 0, 7, "one two")], "Unknown"),
        ([SpanAnnotation("main_instruction", -1, 7, "one two")], "offsets"),
        ([SpanAnnotation("main_instruction", 0, 0, "")], "offsets"),
        (
            [
                SpanAnnotation("main_instruction", 0, 7, "one two"),
                SpanAnnotation("context.data", 4, 7, "two"),
            ],
            "Overlapping",
        ),
        (
            [
                SpanAnnotation("main_instruction", 0, 7, "one two"),
                SpanAnnotation("context.data", 0, 7, "one two"),
            ],
            "Overlapping",
        ),
    ],
)
def test_renderer_rejects_invalid_or_incomplete_annotations(
    spans: list[SpanAnnotation], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        render_prompt_variants(canonical("one two", *spans))


def test_annotation_hash_changes_when_semantic_assignment_changes() -> None:
    first = canonical("hello", SpanAnnotation("main_instruction", 0, 5, "hello"))
    second = canonical("hello", SpanAnnotation("context.data", 0, 5, "hello"))
    t1, h1 = render_prompt_variants(first)
    t2, h2 = render_prompt_variants(second)
    assert t1.prompt_hash == t2.prompt_hash
    assert h1.prompt_hash != h2.prompt_hash
    assert h1.annotation_hash != h2.annotation_hash


def test_repeated_phrases_keep_distinct_positions_and_all_separators() -> None:
    _, variant = render_prompt_variants(
        canonical(
            "go \tgo\n",
            SpanAnnotation("main_instruction", 4, 6, "go"),
            SpanAnnotation("main_instruction", 0, 2, "go"),
        )
    )
    assert json.loads(variant.text)["prompt"]["main_instruction"] == [
        {"position": 0, "phrase": "go \t"},
        {"position": 1, "phrase": "go\n"},
    ]


def test_metric_v2_rejects_extras_without_changing_legacy_scores() -> None:
    def encoded(phrases: list[str]) -> str:
        return json.dumps({"main_instruction": phrases})

    scores = reconstruction_scores(
        [encoded(["hello"]), encoded(["hello", "invented"]), "not json"],
        ["hello"] * 3,
    )
    legacy = scores.as_dict()
    assert legacy["exact_reconstruction_rate"] == pytest.approx(2 / 3)
    assert "metric_version" not in legacy
    strong = scores.as_dict(version="2")
    assert strong["metric_version"] == "2"
    assert strong["aligned_subset_reconstruction_normalized_rate"] == pytest.approx(2 / 3)
    assert strong["all_leaves_accounted_for_rate"] == pytest.approx(1 / 3)
    assert strong["strict_full_reconstruction_rate"] == pytest.approx(1 / 3)
    assert scores.per_example_strict == [True, False, None]
    with pytest.raises(ValueError, match="version"):
        scores.as_dict(version="unknown")


def test_renderer_emits_nine_leaves_in_taxonomy_order() -> None:
    _, variant = render_prompt_variants(
        canonical("hello", SpanAnnotation("context.data", 0, 5, "hello"))
    )
    root = json.loads(variant.text)["prompt"]
    paths = []
    for key, value in root.items():
        if isinstance(value, dict):
            paths.extend(f"{key}.{child}" for child in value)
        else:
            paths.append(key)
    assert tuple(paths) == LEAF_PATHS
