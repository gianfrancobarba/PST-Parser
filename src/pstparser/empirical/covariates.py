"""Frozen prompt covariates, computed from the prompts alone before any generation.

Covariates describe what the hierarchical form changes in each
prompt: its length, how many leaves and segments it uses, how many segments it
moves away from their source order, whether the prompt holds code, and how many
units of the gold needed a human decision. They never read a model output,
a test or a reference solution.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from pstparser.empirical import annotations as gold
from pstparser.empirical.contracts import CanonicalPrompt, PromptVariant, SpanAnnotation
from pstparser.empirical.rendering import render_prompt_variants
from pstparser.pst.taxonomy import LEAF_PATHS

COVARIATES_VERSION: Final = "prompt-covariates-v1"
HAS_CODE_RULE: Final = "has-code-v1"
JUDGEMENT_UNITS_RULE: Final = "judgement-units-v2"
#: Unit types of the drafts: the first sentence (leaf fixed by rule G3.1), the
#: other sentences of the prose (G3) and the sentences of "Note that:" (B6).
UNIT_TYPES: Final = ("first", "prose", "notes")

# has-code-v1: a fenced block, or a line that starts a function definition.
_CODE: Final = re.compile(r"```|^[ \t]*def[ \t]+\w+[ \t]*\(", re.MULTILINE)


def has_code(text: str) -> bool:
    """Rule ``has-code-v1``: the text holds a fenced block or a ``def`` line.

    On BigCodeBench every prompt holds its starting code, so the rule is true
    for all of them; it separates prompts on LLMSecEval.
    """
    return _CODE.search(text) is not None


def _ordered(prompt: CanonicalPrompt) -> list[SpanAnnotation]:
    return sorted(prompt.annotations, key=lambda span: (span.start, span.end))


def structure(prompt: CanonicalPrompt) -> dict[str, int]:
    """Leaves, segments and reordered segments of the hierarchical form.

    ``reorder`` counts the segments whose position in the serialisation of H
    (leaves in taxonomy order, segments by source rank inside a leaf) differs
    from their rank in the source text; it is 0 when H keeps the source order.

    Raises:
        ValueError: If the annotations do not render (same checks as H-gold).
    """
    render_prompt_variants(prompt)
    ordered = _ordered(prompt)
    rank = {id(span): index for index, span in enumerate(ordered)}
    emitted = [span for path in LEAF_PATHS for span in ordered if span.path == path]
    return {
        "n_leaves": len({span.path for span in ordered}),
        "n_segments": len(ordered),
        "reorder": sum(1 for index, span in enumerate(emitted) if rank[id(span)] != index),
    }


def unit_regions(text: str, rule: str) -> dict[str, list[tuple[int, int]]]:
    """Character ranges of the draft units of each type in :data:`UNIT_TYPES`.

    ``first`` is the first sentence (kind ``first``), ``prose`` the ``G3``
    judgement units, ``notes`` the ``B6`` judgement units of "Note that:".

    Raises:
        ValueError: If ``rule`` is unknown or the text does not fit it.
    """
    if rule not in gold.RULES:
        raise ValueError(f"unknown rule {rule!r}; choose one of {sorted(gold.RULES)}")
    _, propose = gold.RULES[rule]
    regions: dict[str, list[tuple[int, int]]] = {name: [] for name in UNIT_TYPES}
    for unit in propose(text):
        if unit.kind == "first":
            regions["first"].append((unit.start, unit.end))
        elif unit.kind == "judgement" and unit.rule == "G3":
            regions["prose"].append((unit.start, unit.end))
        elif unit.kind == "judgement" and unit.rule == "B6":
            regions["notes"].append((unit.start, unit.end))
    return regions


def judgement_regions(text: str, rule: str) -> list[tuple[int, int]]:
    """Character ranges of the units whose leaf is decided by the reviewer.

    Rule ``judgement-units-v2``: the sentences of the prose of
    the description after the first one (``G3``) and the sentences of the
    "Note that:" section (``B6``). The first sentence is left out: rule G3.1
    fixes its leaf. Fixed sections, the starting code, parameter and example
    blocks are excluded. A task may have no judgement unit at all.

    Raises:
        ValueError: If ``rule`` is unknown or the text does not fit it.
    """
    regions = unit_regions(text, rule)
    return sorted(regions["prose"] + regions["notes"])


def _mask(text: str, regions: list[tuple[int, int]]) -> tuple[bool, ...]:
    mask = [False] * len(text)
    for start, end in regions:
        mask[start:end] = [True] * (end - start)
    return tuple(mask)


def unit_type_mask(text: str, rule: str, unit_type: str) -> tuple[bool, ...]:
    """One flag per character: True inside a unit of ``unit_type``."""
    return _mask(text, unit_regions(text, rule)[unit_type])


def judgement_mask(text: str, rule: str) -> tuple[bool, ...]:
    """One flag per character: True inside a judgement region of :func:`judgement_regions`."""
    return _mask(text, judgement_regions(text, rule))


def judgement_units(prompt: CanonicalPrompt, rule: str) -> int:
    """Number of gold spans that overlap a judgement region of the source.

    Computed from the frozen gold: a reviewer who joins two sentences into one
    span leaves one unit. On an untouched draft it is the number of prose
    sentences after the first one plus the sentences of the notes.
    """
    regions = judgement_regions(prompt.text, rule)
    return sum(
        1
        for span in prompt.annotations
        if any(span.start < end and start < span.end for start, end in regions)
    )


def prompt_covariates(
    canonical: CanonicalPrompt,
    variants: Mapping[str, PromptVariant],
    count_tokens: Callable[[str], int],
    rule: str,
) -> dict[str, Any]:
    """Covariates of one task, from its gold and its rendered variants.

    ``count_tokens`` returns the tokens of the whole model input for a variant
    text (system message, wrapper and request, with the generation prompt), so
    lengths are measured on what the model reads. ``variants`` must hold T and
    H-gold rendered from ``canonical``; other conditions are measured too.

    Raises:
        ValueError: If T or H-gold is missing, or a variant belongs to another
            task or another source.
    """
    if not {"T", "H-gold"} <= set(variants):
        raise ValueError(f"{canonical.task_id}: T and H-gold are required")
    expected_t, expected_h = render_prompt_variants(canonical)
    if variants["T"] != expected_t or variants["H-gold"] != expected_h:
        raise ValueError(f"{canonical.task_id}: the variants were not rendered from this gold")
    for condition, variant in variants.items():
        if variant.condition != condition or variant.task_id != canonical.task_id:
            raise ValueError(f"{canonical.task_id}: variant {condition} is mislabelled")
        if variant.source_hash != expected_t.source_hash:
            raise ValueError(f"{canonical.task_id}: variant {condition} has another source")
    conditions = sorted(variants, key=lambda name: (name not in ("T", "H-gold"), name))
    characters = {name: len(variants[name].text) for name in conditions}
    tokens = {name: int(count_tokens(variants[name].text)) for name in conditions}
    if any(value <= 0 for value in tokens.values()):
        raise ValueError(f"{canonical.task_id}: a token count is not positive")
    return {
        "task_id": canonical.task_id,
        "covariates_version": COVARIATES_VERSION,
        "characters": characters,
        "tokens": tokens,
        "ratio_HT": tokens["H-gold"] / tokens["T"],
        "log_ratio_HT": math.log(tokens["H-gold"] / tokens["T"]),
        "log_tokens_T": math.log(tokens["T"]),
        **structure(canonical),
        "has_code": has_code(canonical.text),
        "has_code_rule": HAS_CODE_RULE,
        "n_judgement_units": judgement_units(canonical, rule),
        "judgement_units_rule": JUDGEMENT_UNITS_RULE,
    }


def covariate_table(
    canonicals: Sequence[CanonicalPrompt],
    variants: Sequence[PromptVariant],
    count_tokens: Callable[[str], int],
    rule: str,
) -> list[dict[str, Any]]:
    """Covariates of every canonical prompt, in the order of ``canonicals``.

    Raises:
        ValueError: If a task has no variants, variants name an unknown task,
            or a condition is repeated.
    """
    by_task: dict[str, dict[str, PromptVariant]] = {}
    for variant in variants:
        cell = by_task.setdefault(variant.task_id, {})
        if variant.condition in cell:
            raise ValueError(f"{variant.task_id}: condition {variant.condition} is repeated")
        cell[variant.condition] = variant
    names = [prompt.task_id for prompt in canonicals]
    if len(set(names)) != len(names):
        raise ValueError("a canonical task is repeated")
    if set(by_task) != set(names):
        raise ValueError("variants and canonical prompts must cover the same tasks")
    return [
        prompt_covariates(prompt, by_task[prompt.task_id], count_tokens, rule)
        for prompt in canonicals
    ]
