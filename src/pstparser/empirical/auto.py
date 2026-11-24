"""H-auto: parser predictions turned into source spans, S_parse and the parser gate.

The conversion reads only the source text and the parser's raw output: never
the gold spans. Each located phrase becomes the span of the source characters
it matched, narrowed to its non-whitespace edges; the phrase is taken from the
source, so a match at the ``collapsed`` or ``stripped`` level stays lossless.
Blank leaves carry no content and are ignored.

Two policies:

* ``auto-strict-v1`` (primary, fail-closed): an invalid output, a phrase that is
  absent from the source or only occurs where another one sits (contested), or
  a gap with non-whitespace content fails the task (``parser_failure``);
* ``auto-attach-v1`` (sensitivity): unplaced phrases are dropped and counted; a
  gap with content extends the preceding span (a prefix extends the first);
  only an invalid output or no located phrase fails the task.
* ``auto-strict-v2`` (`protocol/analysis/plan.md`): as ``auto-strict-v1``,
  except that a gap that is exactly the starting code block of a BigCodeBench
  prompt becomes a ``context.data`` span. The block is found in the source text
  by its fixed marker (rule B4 of the drafts), never in the gold. Available but
  never the default: it becomes primary only by the C-22 rule of
  ``protocol/analysis/plan.md`` §2 (:func:`c22_primary_policy`) read on the v1
  conversion of the pilot tasks.

The tasks whose conversion succeeds form S_parse, written with its hash before
any H-auto generation. H3 is estimated on S_parse; the end-to-end estimate
counts every parser failure as a failure of the H-auto arm.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

from pstparser.empirical.analysis import OutcomeRecord, _exact_binomial_interval
from pstparser.empirical.contracts import CanonicalPrompt, PromptVariant, SpanAnnotation
from pstparser.empirical.rendering import render_prompt_variants
from pstparser.evaluation.metrics.validity import is_valid
from pstparser.pst.alignment import DEFAULT_LEVELS, align_tree

STRICT: Final = "auto-strict-v1"
ATTACH: Final = "auto-attach-v1"
STRICT_V2: Final = "auto-strict-v2"
POLICIES: Final = (STRICT, ATTACH, STRICT_V2)
#: Strict policies: a gap with content fails the task (v2 exempts the starting code).
STRICT_POLICIES: Final = (STRICT, STRICT_V2)
#: Share of the v1 failures above which auto-strict-v2 becomes primary.
C22_THRESHOLD: Final = 0.5
S_PARSE_FORMAT: Final = "s-parse-v1"
GATE_FORMAT: Final = "parser-gate-v1"
#: Causes of a parser failure, in the order they are checked.
CAUSES: Final = (
    "invalid_output",
    "absent_phrase",
    "contested_phrase",
    "content_gap",
    "no_located_phrase",
    "render_error",
)
#: Outcome injected for the H-auto arm of a failed task in the end-to-end estimate.
FAILURE_OUTCOME: Final = {"functional": "fail", "security": "I"}

Status = Literal["ok", "parser_failure"]
_INDEX: Final = re.compile(r"\[\d+\]$")


@dataclass(frozen=True)
class AutoResult:
    """The spans of one task, or the reason the conversion failed."""

    status: Status
    spans: tuple[SpanAnnotation, ...]
    cause: str | None
    diagnostics: dict[str, int] = field(default_factory=dict)


def leaf_path(path: str) -> str:
    """``prompt.context.data[0]`` → ``context.data``."""
    stripped = _INDEX.sub("", path)
    if not stripped.startswith("prompt."):
        raise ValueError(f"not a leaf of the prompt tree: {path}")
    return stripped.removeprefix("prompt.")


def _narrow(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _span(text: str, path: str, start: int, end: int) -> SpanAnnotation:
    return SpanAnnotation(path, start, end, text[start:end])


def starting_code_block(text: str) -> tuple[int, int] | None:
    """Bounds of the starting code block of a BigCodeBench prompt, from the source alone.

    The block is the unit of rule B4 of the drafts (the fenced code after the
    fixed marker); a text without that shape has none.
    """
    from pstparser.empirical.annotations import bigcodebench_units

    try:
        units = bigcodebench_units(text)
    except ValueError:
        return None
    blocks = [(unit.start, unit.end) for unit in units if unit.rule == "B4"]
    return blocks[0] if len(blocks) == 1 else None


def predicted_spans(text: str, prediction: str, *, policy: str) -> AutoResult:
    """Spans of ``text`` from the parser's raw ``prediction`` under ``policy``.

    Raises:
        ValueError: If the policy is unknown.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    if not is_valid(prediction):
        return AutoResult("parser_failure", (), "invalid_output", {"invalid_output": 1})
    alignment = align_tree(text, json.loads(prediction), DEFAULT_LEVELS)
    scored = alignment.scored
    absent = sum(leaf.absent for leaf in scored)
    contested = sum(leaf.contested for leaf in scored)
    located = sorted((leaf for leaf in scored if leaf.located), key=lambda leaf: leaf.start or 0)
    spans = []
    for leaf in located:
        assert leaf.start is not None and leaf.end is not None
        start, end = _narrow(text, leaf.start, leaf.end)
        spans.append(_span(text, leaf_path(leaf.path), start, end))
    gaps = []
    cursor = 0
    for index, span in enumerate([*spans, None]):
        stop = len(text) if span is None else span.start
        start, end = _narrow(text, cursor, stop)
        if start < end:
            gaps.append((index, start, end))
        if span is not None:
            cursor = span.end
    diagnostics = {
        "phrases": len(scored),
        "blank_phrases": len(alignment.leaves) - len(scored),
        "located": len(located),
        "absent": absent,
        "contested": contested,
        "non_exact": sum(leaf.level != "exact" for leaf in located),
        "content_gaps": len(gaps),
    }
    block = starting_code_block(text) if policy in STRICT_POLICIES else None
    code_gaps = [gap for gap in gaps if block is not None and (gap[1], gap[2]) == block]
    if policy == STRICT:
        # How many v1 failures the v2 exemption alone would rescue (C-22 rule).
        diagnostics["only_gap_is_starting_code"] = int(
            not absent and not contested and bool(spans) and gaps == code_gaps and bool(gaps)
        )
    if policy == STRICT_V2 and code_gaps:
        assert block is not None
        index = code_gaps[0][0]
        spans.insert(index, _span(text, "context.data", *block))
        gaps = [gap for gap in gaps if gap not in code_gaps]
        diagnostics["starting_code_exempt"] = 1
    if policy in STRICT_POLICIES:
        for cause, count in (
            ("absent_phrase", absent),
            ("contested_phrase", contested),
            ("content_gap", len(gaps)),
            ("no_located_phrase", int(not spans)),
        ):
            if count:
                return AutoResult("parser_failure", (), cause, diagnostics)
    else:
        if not spans:
            return AutoResult("parser_failure", (), "no_located_phrase", diagnostics)
        diagnostics["dropped_phrases"] = absent + contested
        diagnostics["attached_gaps"] = len(gaps)
        for index, start, end in gaps:
            # A gap before span ``index`` belongs to the span before it; the prefix to the first.
            if index == 0:
                first = spans[0]
                spans[0] = _span(text, first.path, start, first.end)
            else:
                previous = spans[index - 1]
                spans[index - 1] = _span(text, previous.path, previous.start, end)
    try:
        render_prompt_variants(CanonicalPrompt("check", text, tuple(spans), policy))
    except ValueError:
        return AutoResult("parser_failure", (), "render_error", diagnostics)
    return AutoResult("ok", tuple(spans), None, diagnostics)


def tasks_digest(tasks: Sequence[str]) -> str:
    """SHA-256 of the task list (UTF-8 JSON, compact, in the given order)."""
    payload = json.dumps(list(tasks), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clopper_pearson(failures: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval of a binomial proportion."""
    if n <= 0 or not 0 <= failures <= n:
        raise ValueError("need 0 <= failures <= n and n > 0")
    return _exact_binomial_interval(failures, n, confidence)


@dataclass(frozen=True)
class Conversion:
    """H-auto canonical prompts, failures and the S_parse document of one run."""

    canonical: tuple[CanonicalPrompt, ...]
    failures: tuple[dict[str, Any], ...]
    s_parse: dict[str, Any]


def auto_annotate(
    predictions: Sequence[Mapping[str, Any]],
    id_map: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    inputs: Mapping[str, str | None] | None = None,
) -> Conversion:
    """Convert the parser's predictions of ``sources`` under ``policy``.

    Only ``index``, ``prompt`` and ``prediction`` of a prediction are read;
    its ``target`` (the gold tree, if any) is ignored. Every source task needs
    exactly one prediction, whose prompt is the source text.

    Raises:
        ValueError: If predictions, index map and sources do not correspond.
    """
    tasks = id_map["tasks"]
    texts = {row["task_id"]: row["text"] for row in sources}
    if len(texts) != len(sources) or sorted(texts) != sorted(tasks):
        raise ValueError("the index map and the sources must hold the same tasks once")
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in predictions:
        if row["index"] in by_index:
            raise ValueError(f"index {row['index']} repeated")
        by_index[row["index"]] = row
    if sorted(by_index) != list(range(len(tasks))):
        raise ValueError("one prediction per index of the map is required")
    canonical: list[CanonicalPrompt] = []
    failures: list[dict[str, Any]] = []
    successes: list[str] = []
    causes: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for index, task in enumerate(tasks):
        row = by_index[index]
        text = texts[task]
        if row["prompt"] != text:
            raise ValueError(f"{task}: the parser did not read the source text")
        result = predicted_spans(text, row["prediction"], policy=policy)
        totals.update(result.diagnostics)
        if result.status == "ok":
            canonical.append(CanonicalPrompt(task, text, result.spans, policy))
            successes.append(task)
        else:
            assert result.cause is not None
            causes[result.cause] += 1
            failures.append(
                {
                    "task_id": task,
                    "policy": policy,
                    "cause": result.cause,
                    "diagnostics": result.diagnostics,
                }
            )
    low, high = clopper_pearson(len(failures), len(tasks))
    document = {
        "format": S_PARSE_FORMAT,
        "policy": policy,
        "tasks": successes,
        "tasks_sha256": tasks_digest(successes),
        "n": len(tasks),
        "converted": len(successes),
        "failures": len(failures),
        "failure_rate": len(failures) / len(tasks),
        "failure_rate_ci95_clopper_pearson": [low, high],
        "causes": {cause: causes[cause] for cause in CAUSES if causes[cause]},
        "diagnostics": dict(sorted(totals.items())),
        "inputs": dict(inputs or {}),
    }
    if policy == STRICT:
        rescued = totals["only_gap_is_starting_code"]
        document["starting_code_only_failures"] = rescued
        document["starting_code_share_of_failures"] = rescued / len(failures) if failures else None
    return Conversion(tuple(canonical), tuple(failures), document)


def c22_primary_policy(s_parse: Mapping[str, Any]) -> str:
    """The primary strict policy by the C-22 rule, read on a v1 conversion.

    ``auto-strict-v2`` if the omitted starting code alone causes more than half
    of the strict failures; otherwise ``auto-strict-v1``. With no failure, v1.

    Raises:
        ValueError: If the document is not an ``auto-strict-v1`` conversion.
    """
    if s_parse.get("format") != S_PARSE_FORMAT or s_parse.get("policy") != STRICT:
        raise ValueError(f"the C-22 rule reads an {STRICT} conversion")
    share = s_parse.get("starting_code_share_of_failures")
    return STRICT_V2 if share is not None and share > C22_THRESHOLD else STRICT


def load_s_parse(document: Mapping[str, Any]) -> list[str]:
    """The tasks of an S_parse document, after checking its hash.

    Raises:
        ValueError: If the format or the hash does not match.
    """
    if document.get("format") != S_PARSE_FORMAT:
        raise ValueError(f"not an {S_PARSE_FORMAT} document")
    tasks = list(document["tasks"])
    if tasks_digest(tasks) != document["tasks_sha256"]:
        raise ValueError("the S_parse task list does not match its hash")
    return tasks


def as_hauto(variant: PromptVariant) -> PromptVariant:
    """The hierarchical variant of an H-auto canonical prompt, labelled H-auto."""
    if variant.condition != "H-gold":
        raise ValueError("only the hierarchical variant becomes H-auto")
    return replace(variant, condition="H-auto")


def hauto_ledgers(
    records: Sequence[OutcomeRecord],
    s_parse: Sequence[str],
    failures: Sequence[Mapping[str, Any]],
) -> tuple[list[OutcomeRecord], list[OutcomeRecord]]:
    """The ledgers of the two H3 estimates, T against H-auto.

    Primary: the S_parse tasks only. End-to-end (secondary): every task, with
    the H-auto arm of each failed task injected as ``fail`` (functional) or
    ``I`` (security) with reason ``parser_failure:<cause>``, once per replicate
    observed for T. Records of other conditions are left out; their
    ``analysis_id`` gets the suffix ``-s-parse`` or ``-end-to-end``.

    Raises:
        ValueError: If a task lacks T, an S_parse task lacks H-auto, a failed
            task has an H-auto record, or the two lists overlap.
    """
    failed = {row["task_id"]: row["cause"] for row in failures}
    parsed = set(s_parse)
    if parsed & set(failed) or len(parsed) != len(s_parse):
        raise ValueError("S_parse and the failures must be disjoint and without repetitions")
    endpoints = {record.endpoint for record in records}
    identifiers = {record.analysis_id for record in records}
    if len(endpoints) != 1 or len(identifiers) != 1:
        raise ValueError("the ledger must hold one endpoint and one analysis_id")
    (endpoint,) = endpoints
    (identifier,) = identifiers
    t_records = [r for r in records if r.condition == "T"]
    h_records = [r for r in records if r.condition == "H-auto"]
    if {r.task_id for r in t_records} != parsed | set(failed):
        raise ValueError("T must be recorded for every S_parse and every failed task")
    if {r.task_id for r in h_records} != parsed:
        raise ValueError("H-auto must be recorded for exactly the S_parse tasks")
    primary = [
        replace(r, analysis_id=f"{identifier}-s-parse")
        for r in (*t_records, *h_records)
        if r.task_id in parsed
    ]
    injected = [
        OutcomeRecord(
            analysis_id=identifier,
            endpoint=endpoint,
            task_id=r.task_id,
            condition="H-auto",
            replicate_id=r.replicate_id,
            outcome=FAILURE_OUTCOME[endpoint],  # type: ignore[arg-type]
            reason=f"parser_failure:{failed[r.task_id]}",
        )
        for r in t_records
        if r.task_id in failed
    ]
    end_to_end = [
        replace(r, analysis_id=f"{identifier}-end-to-end")
        for r in (*t_records, *h_records, *injected)
    ]
    return primary, end_to_end


def s_parse_ledger(records: Sequence[OutcomeRecord], s_parse: Sequence[str]) -> list[OutcomeRecord]:
    """Every condition of the S_parse tasks, for RQ3c and the decomposition of Δ_auto.

    The ``analysis_id`` gets the suffix ``-s-parse-all``; the analyses pick their
    pair with ``analyze --ledger-conditions``.

    Raises:
        ValueError: If the ledger holds more than one ``analysis_id`` or an
            S_parse task lacks H-auto.
    """
    identifiers = {record.analysis_id for record in records}
    if len(identifiers) != 1:
        raise ValueError("the ledger must hold one analysis_id")
    (identifier,) = identifiers
    parsed = set(s_parse)
    with_hauto = {r.task_id for r in records if r.condition == "H-auto"}
    if not parsed <= with_hauto:
        raise ValueError("H-auto must be recorded for every S_parse task")
    return [
        replace(r, analysis_id=f"{identifier}-s-parse-all") for r in records if r.task_id in parsed
    ]


def macro_f1(field_f1: Mapping[str, Mapping[str, Any]]) -> tuple[float | None, list[str]]:
    """Mean F1 over the leaves with gold support, as ``score`` reports them."""
    leaves = sorted(name for name, value in field_f1.items() if value["support"] > 0)
    values = [field_f1[name]["f1"] or 0.0 for name in leaves]
    return (math.fsum(values) / len(values) if values else None), leaves


MACRO_F1_DEFINITION: Final = (
    "Mean over the leaves with gold support of the leaf F1 reported by score: per prediction "
    "and leaf, the distinct tokens assigned to the leaf; overlap, predicted and gold token "
    "counts summed over the predictions (micro per leaf), then averaged over the leaves "
    "(macro). Predictions that are not valid JSON are excluded from the F1 and count only in "
    "the strict failure rate."
)


def parser_gate(
    s_parse: Mapping[str, Any],
    score: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    id_map: Mapping[str, Any],
    id_map_sha256: str | None,
    sample_tasks: Sequence[str],
    c22: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The parser gate: strict failure rate and leaf macro F1 against frozen thresholds.

    ``s_parse`` must come from the primary strict policy on S_gold: ``auto-strict-v1``,
    or ``auto-strict-v2`` only when the C-22 rule read on ``c22`` (the v1
    conversion of the pilot tasks) makes it primary (`protocol/analysis/plan.md`);
    a v1 conversion is refused when that rule makes v2 primary. ``score`` is the
    ``score`` report of the same predictions against the gold targets. Both are
    bound to those predictions: the score must record the digest
    the conversion read, over the indices of ``id_map``, whose digest the
    conversion recorded too; and the tasks of ``id_map`` must be exactly
    ``sample_tasks`` (S_gold).

    Raises:
        ValueError: If a threshold is missing or invalid, the conversion is not
            the primary strict one by the C-22 rule, or score, conversion, index
            map and sample do not correspond.
    """
    for key in ("max_strict_failure", "min_macro_f1"):
        value = thresholds.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"the frozen thresholds need {key} in [0, 1]")
    policy = s_parse.get("policy")
    if policy not in STRICT_POLICIES:
        raise ValueError(f"the gate reads a strict conversion ({', '.join(STRICT_POLICIES)})")
    primary = None if c22 is None else c22_primary_policy(c22)
    if policy == STRICT_V2 and primary != STRICT_V2:
        raise ValueError("auto-strict-v2 is gated only when the C-22 rule makes it primary")
    if policy == STRICT and primary == STRICT_V2:
        raise ValueError("the C-22 rule makes auto-strict-v2 primary: gate its conversion")
    load_s_parse(s_parse)
    if score["total"] != s_parse["n"]:
        raise ValueError("the score report and the conversion cover different task counts")
    inputs = s_parse.get("inputs") or {}
    provenance = score.get("provenance") or {}
    predictions_sha256 = inputs.get("predictions_sha256")
    if not predictions_sha256 or provenance.get("predictions_sha256") != predictions_sha256:
        raise ValueError("the score report was computed on other predictions than the conversion")
    if id_map_sha256 is None or inputs.get("id_map_sha256") != id_map_sha256:
        raise ValueError("the index map is not the one the conversion read")
    tasks = list(id_map["tasks"])
    if provenance.get("indices") != list(range(len(tasks))):
        raise ValueError("the score report does not cover the indices of the index map in order")
    if "tasks" in provenance and provenance["tasks"] != tasks:
        raise ValueError("the score report records other tasks than the index map")
    if len(set(tasks)) != len(tasks) or sorted(tasks) != sorted(sample_tasks):
        raise ValueError("the parsed tasks are not exactly the tasks of the frozen sample")
    f1, leaves = macro_f1(score["field_f1"])
    reasons = []
    if s_parse["failure_rate"] > thresholds["max_strict_failure"]:
        reasons.append("strict failure rate above the threshold")
    if f1 is None or f1 < thresholds["min_macro_f1"]:
        reasons.append("leaf macro F1 below the threshold")
    return {
        "format": GATE_FORMAT,
        "policy": policy,
        "c22_primary_policy": primary,
        "thresholds": {
            "max_strict_failure": thresholds["max_strict_failure"],
            "min_macro_f1": thresholds["min_macro_f1"],
        },
        "n": s_parse["n"],
        "strict_failures": s_parse["failures"],
        "strict_failure_rate": s_parse["failure_rate"],
        "strict_failure_rate_ci95_clopper_pearson": s_parse["failure_rate_ci95_clopper_pearson"],
        "causes": s_parse["causes"],
        "s_parse_size": s_parse["converted"],
        "s_parse_tasks_sha256": s_parse["tasks_sha256"],
        "macro_f1": f1,
        "macro_f1_leaves": leaves,
        "macro_f1_definition": MACRO_F1_DEFINITION,
        "predictions_sha256": predictions_sha256,
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "h3": "descriptive" if reasons else "confirmatory",
    }
