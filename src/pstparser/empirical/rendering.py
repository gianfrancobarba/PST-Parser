"""Render traditional and reviewed PST inputs without changing their content."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Final

from pstparser.empirical.contracts import CanonicalPrompt, PromptVariant, SpanAnnotation
from pstparser.pst.taxonomy import LEAF_PATHS, assemble

RENDERER_VERSION: Final = "pst-gold-json-v1"
ENVELOPE_RENDERER_VERSION: Final = "envelope-json-v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def envelope_text(text: str) -> str:
    """Serialise ``{"prompt": text}`` exactly as H is serialised.

    Same ``json.dumps`` options as the hierarchical variant (UTF-8 characters,
    ``indent=2``, no final newline), so code is escaped the same way in E and H.
    """
    return json.dumps({"prompt": text}, ensure_ascii=False, indent=2, allow_nan=False)


def render_envelope_variant(prompt: CanonicalPrompt) -> PromptVariant:
    """Produce E (``envelope-json-v1``): the whole source text in one neutral field.

    E isolates the JSON envelope, its escaping and part of the length (00 T-4).
    It carries no PST label and no annotation: the spans of ``prompt`` are
    ignored, so ``annotation_hash`` and ``annotation_version`` are empty.

    Raises:
        ValueError: If the task identifier or the text is empty.
    """
    if not prompt.task_id.strip():
        raise ValueError("task_id must be non-empty")
    if not prompt.text.strip():
        raise ValueError("A benchmark prompt must contain non-whitespace text")
    text = envelope_text(prompt.text)
    return PromptVariant(
        task_id=prompt.task_id,
        condition="E",
        text=text,
        prompt_hash=_sha256(text),
        source_hash=_sha256(prompt.text),
        annotation_hash="",
        annotation_version="",
        renderer_version=ENVELOPE_RENDERER_VERSION,
    )


def _validated_spans(prompt: CanonicalPrompt) -> list[SpanAnnotation]:
    """Validate annotations and allow only whitespace as unassigned separators."""
    if not prompt.task_id.strip() or not prompt.annotation_version.strip():
        raise ValueError("task_id and annotation_version must be non-empty")
    if not prompt.text or not prompt.text.strip():
        raise ValueError("A benchmark prompt must contain non-whitespace text")
    if not prompt.annotations:
        raise ValueError("Missing annotations")
    for span in prompt.annotations:
        if span.path not in LEAF_PATHS:
            raise ValueError(f"Unknown taxonomy leaf: {span.path}")
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or not 0 <= span.start < span.end <= len(prompt.text)
        ):
            raise ValueError("Span offsets must be valid non-empty character intervals")
        if not span.phrase.strip():
            raise ValueError("Whitespace-only annotations must be attached to adjacent content")
        if span.phrase != prompt.text[span.start : span.end]:
            raise ValueError("Annotated phrase differs from source span")
    ordered = sorted(prompt.annotations, key=lambda span: (span.start, span.end))
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            raise ValueError("Overlapping or duplicated annotation spans")
        if prompt.text[cursor : span.start].strip():
            raise ValueError("Unannotated non-whitespace source content")
        cursor = span.end
    if prompt.text[cursor:].strip():
        raise ValueError("Unannotated non-whitespace source content")
    return ordered


def render_prompt_variants(prompt: CanonicalPrompt) -> tuple[PromptVariant, PromptVariant]:
    """Produce T and H-gold using a frozen, lossless separator policy.

    Unassigned whitespace between spans and at the end goes to the preceding
    span; a prefix goes to the first. Non-whitespace gaps, additions, changed
    text, overlap and empty annotations fail closed. Positions are zero-based
    source ranks, not character offsets. All nine leaves are emitted in taxonomy
    order. JSON uses UTF-8 characters, indent=2 and no final newline. No semantic
    classification, code repair or LLM call is performed by this renderer.

    Returns:
        Traditional and hierarchical variants, in that order.

    Raises:
        ValueError: If the reviewed annotations do not cover the original text.
    """
    ordered = _validated_spans(prompt)
    values: dict[str, Any] = {path: [] for path in LEAF_PATHS}
    reconstructed: list[str] = []
    for rank, span in enumerate(ordered):
        start = 0 if rank == 0 else span.start
        end = ordered[rank + 1].start if rank + 1 < len(ordered) else len(prompt.text)
        phrase = prompt.text[start:end]
        values[span.path].append({"position": rank, "phrase": phrase})
        reconstructed.append(phrase)
    if "".join(reconstructed) != prompt.text:
        raise ValueError("Renderer failed exact source reconstruction")
    hierarchical = json.dumps(assemble(values), ensure_ascii=False, indent=2, allow_nan=False)
    annotation_payload = json.dumps(
        {
            "source_hash": _sha256(prompt.text),
            "annotation_version": prompt.annotation_version,
            "spans": [asdict(span) for span in ordered],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata = {
        "task_id": prompt.task_id,
        "source_hash": _sha256(prompt.text),
        "annotation_hash": _sha256(annotation_payload),
        "annotation_version": prompt.annotation_version,
        "renderer_version": RENDERER_VERSION,
    }
    return (
        PromptVariant(
            condition="T", text=prompt.text, prompt_hash=_sha256(prompt.text), **metadata
        ),
        PromptVariant(
            condition="H-gold", text=hierarchical, prompt_hash=_sha256(hierarchical), **metadata
        ),
    )
