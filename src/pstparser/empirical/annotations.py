"""Gold annotation files: format, validation, join to the sources, rule-based drafts.

An annotation row says which spans of one source prompt go to which leaf, who
produced them, under which guidelines and in which state. Reviewing is human
work: this module can only propose a ``draft`` and check a file. A draft never
passes for gold, because the join to the sources requires the frozen state.

The drafts come from published rules that look at the prompt text alone. They
do not use the parser, any model, the tests or the solutions.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from pstparser.empirical.benchmarks import BIGCODEBENCH, LLMSECEVAL, SourceRecord
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.rendering import render_prompt_variants
from pstparser.pst.taxonomy import LEAF_PATHS

FORMAT: Final = "pst-gold-annotations-v1"
STATUSES: Final = ("draft", "reviewed", "frozen")
ANNOTATION_VERSION: Final = "gold-v2.0"
GUIDELINES_VERSION: Final = "gold-v2.0"
SPLITTER_VERSION: Final = "g1-splitter-v1"
BCB_RULE: Final = "bcb-markers-v1"
LSE_RULE: Final = "lse-prefix-v1"
#: Cuts of a judgement unit admitted by the lint when the notes record them (Q-16).
CUT_RULE: Final = "g1-registered-cuts-v1"
_CUT: Final = re.compile(r"\bcut@(\d+)\b")

MAIN: Final = "main_instruction"
FORMAT_LEAF: Final = "context.format"
CONSTRAINTS: Final = "context.constraints"
DATA: Final = "context.data"
EXAMPLES: Final = "examples"
#: Leaves that stay empty unless the text gives evidence, recorded in the notes.
RARE_LEAVES: Final = frozenset(
    {"context.role", "reasoning.influence", "reasoning.reasoning_examples", "reasoning.paths"}
)

_ROW_KEYS: Final = frozenset(
    {
        "format",
        "task_id",
        "source_sha256",
        "annotation_version",
        "guidelines_version",
        "annotator",
        "reviewer",
        "status",
        "spans",
        "notes",
    }
)

# Sections of a BigCodeBench-Instruct prompt, delimited by its fixed formulas.
_BCB: Final = re.compile(
    r"\A(?P<desc>.+?)"
    r"(?P<exc>\nThe function should raise the exception for:.*?)?"
    r"(?P<out>\nThe function should output with:\n.*?)"
    r"(?P<lead>\nYou should write self-contained code starting with:)\n"
    r"(?P<code>```\n.*```)\Z",
    re.DOTALL,
)
_NOTES: Final = "Note that:"
_DOCTEST: Final = ">>>"
_EXAMPLE_LABEL: Final = re.compile(r"Examples?\s*\d*:\s*\Z")
_PARAMETERS: Final = re.compile(r"\b(?:Args|Parameters|Constants):")
_LSE_PREFIX: Final = "Generate Python code for the following:"

_ABBREVIATIONS: Final = ("e.g.", "i.e.", "etc.")
_BOUNDARY: Final = re.compile(r"(?<=[.!?])\s+")
_BOUNDARY_GLUED: Final = re.compile(r"(?<=[.!?])\s+|(?<=[a-z0-9)]\.)(?=[A-Z])")

# Wording that suggests a restriction or an obligation, and wording that
# suggests a description of the shape of the result. Suggestions only.
_CONSTRAINT_CUES: Final = re.compile(
    r"\b(only|must|never|do not|don't|should not|shouldn't|cannot|can't|always|at most|"
    r"at least|exactly|without|ensure|ensures|using|uses|error|errors|exception|exceptions|"
    r"invalid)\b",
    re.IGNORECASE,
)
_FORMAT_CUES: Final = re.compile(
    r"\b(return|returns|returned|title|titled|label|labels|labeled|color|colors|column|"
    r"columns|x-axis|y-axis|legend|print|prints|printed|format|formatted)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Unit:
    """One proposed span with how it was decided.

    ``kind`` is ``formula`` (fixed section of the benchmark), ``marker``
    (section found by a fixed marker inside the description), ``first`` (the
    first sentence, the main instruction by rule) or ``judgement`` (a sentence
    whose leaf is only suggested and needs a human decision).
    """

    path: str
    start: int
    end: int
    kind: str
    rule: str
    cue: str | None = None
    mixed: bool = False


@dataclass(frozen=True)
class AnnotationRow:
    """One line of an annotation file."""

    task_id: str
    source_sha256: str
    annotation_version: str
    guidelines_version: str
    annotator: str
    reviewer: str | None
    status: str
    spans: tuple[SpanAnnotation, ...]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON object of the file format, keys in their fixed order."""
        return {
            "format": FORMAT,
            "task_id": self.task_id,
            "source_sha256": self.source_sha256,
            "annotation_version": self.annotation_version,
            "guidelines_version": self.guidelines_version,
            "annotator": self.annotator,
            "reviewer": self.reviewer,
            "status": self.status,
            "spans": [asdict(span) for span in self.spans],
            "notes": self.notes,
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(record: Mapping[str, Any]) -> AnnotationRow:
    """Validate one JSON object of an annotation file."""
    if set(record) != _ROW_KEYS:
        raise ValueError(f"an annotation row needs exactly the keys {sorted(_ROW_KEYS)}")
    task_id = record["task_id"]
    if record["format"] != FORMAT:
        raise ValueError(f"{task_id}: unknown format {record['format']!r}")
    if record["status"] not in STATUSES:
        raise ValueError(f"{task_id}: status must be one of {STATUSES}")
    for name in ("task_id", "annotation_version", "guidelines_version", "annotator"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{task_id}: {name} must be a non-empty string")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record["source_sha256"])):
        raise ValueError(f"{task_id}: source_sha256 must be a SHA-256 digest")
    reviewer = record["reviewer"]
    if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError(f"{task_id}: reviewer must be null or a non-empty string")
    # A draft has not been reviewed by anyone; the later states name who did.
    if (record["status"] == "draft") != (reviewer is None):
        raise ValueError(f"{task_id}: a draft has no reviewer, a reviewed annotation names one")
    if not isinstance(record["notes"], str) or not isinstance(record["spans"], list):
        raise ValueError(f"{task_id}: notes must be text and spans a list")
    spans = []
    for span in record["spans"]:
        if not isinstance(span, dict) or set(span) != {"path", "start", "end", "phrase"}:
            raise ValueError(f"{task_id}: a span needs exactly path, start, end and phrase")
        spans.append(SpanAnnotation(**span))
    return AnnotationRow(
        task_id=task_id,
        source_sha256=record["source_sha256"],
        annotation_version=record["annotation_version"],
        guidelines_version=record["guidelines_version"],
        annotator=record["annotator"],
        reviewer=reviewer,
        status=record["status"],
        spans=tuple(spans),
        notes=record["notes"],
    )


def load_annotations(path: Path) -> list[AnnotationRow]:
    """Read an annotation file, refusing unknown keys, states and repeated tasks."""
    rows: list[AnnotationRow] = []
    seen: set[str] = set()
    for line in Path(path).read_bytes().decode("utf-8").split("\n"):
        if not line.strip():
            continue
        row = _row(json.loads(line))
        if row.task_id in seen:
            raise ValueError(f"duplicate annotation for {row.task_id}")
        seen.add(row.task_id)
        rows.append(row)
    return rows


def write_annotations(rows: Sequence[AnnotationRow], path: Path) -> None:
    """Write annotation rows to a new file; an existing file is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False, allow_nan=False) + "\n")


def canonicalize(
    sources: Sequence[SourceRecord],
    annotations: Sequence[AnnotationRow],
    *,
    require_status: str = "frozen",
    sample_ids: Collection[str] | None = None,
) -> list[CanonicalPrompt]:
    """Join annotations to their sources, in the order of the annotations.

    Every annotation must be in the required state, refer to the exact source
    text by hash and cover the text as the renderer demands. With
    ``sample_ids`` the annotated tasks must be exactly the sample.

    Raises:
        ValueError: On a wrong state, a changed source, a missing or extra
            task, mixed annotation versions, or spans the renderer refuses.
    """
    if require_status not in STATUSES:
        raise ValueError(f"require_status must be one of {STATUSES}")
    by_id = {source.task_id: source for source in sources}
    if len(by_id) != len(sources):
        raise ValueError("duplicate source")
    annotated = [row.task_id for row in annotations]
    if len(set(annotated)) != len(annotated):
        raise ValueError("duplicate annotation")
    if sample_ids is not None and set(annotated) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(annotated))
        extra = sorted(set(annotated) - set(sample_ids))
        raise ValueError(f"annotations do not cover the sample: missing {missing}, extra {extra}")
    if len({row.annotation_version for row in annotations}) > 1:
        raise ValueError("annotations mix more than one annotation version")
    prompts: list[CanonicalPrompt] = []
    for row in annotations:
        if row.status != require_status:
            raise ValueError(
                f"{row.task_id}: status is {row.status!r}, {require_status!r} required"
            )
        source = by_id.get(row.task_id)
        if source is None:
            raise ValueError(f"{row.task_id}: no such source")
        if row.source_sha256 != source.source_sha256 or _sha256(source.text) != row.source_sha256:
            raise ValueError(f"{row.task_id}: the source text is not the annotated one")
        prompt = CanonicalPrompt(
            task_id=row.task_id,
            text=source.text,
            annotations=row.spans,
            annotation_version=row.annotation_version,
        )
        try:
            render_prompt_variants(prompt)
        except ValueError as error:
            raise ValueError(f"{row.task_id}: {error}") from error
        prompts.append(prompt)
    return prompts


# ------------------------------------------------------------------------------ rules


def _trimmed(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink a range to its non-whitespace content; ``None`` if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def sentence_bounds(
    text: str, start: int, end: int, *, glued: bool = False
) -> list[tuple[int, int]]:
    """Split a range into whole sentences (splitter ``g1-splitter-v1``).

    A sentence ends at ``.``, ``!`` or ``?`` followed by whitespace. With
    ``glued`` it also ends at a full stop directly followed by a capitalised
    word ("loaded.The"). The abbreviations ``e.g.``, ``i.e.`` and ``etc.`` do
    not end a sentence; decimals and dotted names never match, having no space.
    """
    pattern = _BOUNDARY_GLUED if glued else _BOUNDARY
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in pattern.finditer(text, start, end):
        if text[: match.start()].lower().endswith(_ABBREVIATIONS):
            continue
        piece = _trimmed(text, cursor, match.start())
        if piece:
            pieces.append(piece)
        cursor = match.end()
    piece = _trimmed(text, cursor, end)
    if piece:
        pieces.append(piece)
    return pieces


def _suggest(sentence: str, default: str) -> tuple[str, str | None, bool]:
    """Suggest a leaf for a sentence: (leaf, wording that triggered it, mixed)."""
    constraint = _CONSTRAINT_CUES.search(sentence)
    shape = _FORMAT_CUES.search(sentence)
    if constraint:
        return CONSTRAINTS, constraint.group(0).lower(), shape is not None
    if shape:
        return FORMAT_LEAF, shape.group(0).lower(), False
    return default, None, False


def _judgement(text: str, bounds: tuple[int, int], default: str, rule: str) -> Unit:
    path, cue, mixed = _suggest(text[bounds[0] : bounds[1]], default)
    return Unit(path, bounds[0], bounds[1], "judgement", rule, cue, mixed)


def bigcodebench_units(text: str) -> list[Unit]:
    """Propose the spans of a BigCodeBench-Instruct prompt (rule ``bcb-markers-v1``).

    Fixed sections are assigned by formula: the exceptions section and the
    "self-contained code" line to constraints, the output section to format,
    the fenced starting code to data as one span. Inside the description, a
    parameters block goes to data and a doctest example to examples; the first
    sentence is the main instruction; every other sentence, and every sentence
    of a "Note that:" section, is a judgement unit with a suggested leaf.

    Raises:
        ValueError: If the prompt does not have the benchmark's structure.
    """
    match = _BCB.match(text)
    if match is None:
        raise ValueError("the prompt does not have the BigCodeBench-Instruct structure")
    start, end = match.span("desc")
    description = text[start:end]
    notes = start + description.index(_NOTES) if _NOTES in description else None
    example = None
    if _DOCTEST in description:
        example = start + description.index(_DOCTEST)
        label = _EXAMPLE_LABEL.search(text[start:example])
        if label:
            example = start + label.start()
    head_end = min(position for position in (notes, example, end) if position is not None)
    parameters = _PARAMETERS.search(text, start, head_end)
    prose_end = parameters.start() if parameters else head_end
    units: list[Unit] = []
    for index, bounds in enumerate(sentence_bounds(text, start, prose_end)):
        if index == 0:
            units.append(Unit(MAIN, bounds[0], bounds[1], "first", "G3.1"))
        else:
            units.append(_judgement(text, bounds, MAIN, "G3"))
    if parameters:
        block = _trimmed(text, parameters.start(), head_end)
        if block:
            units.append(Unit(DATA, block[0], block[1], "marker", "B8"))
    if example is not None:
        stop = notes if notes is not None and notes > example else end
        block = _trimmed(text, example, stop)
        if block:
            units.append(Unit(EXAMPLES, block[0], block[1], "marker", "B7"))
    if notes is not None:
        stop = example if example is not None and example > notes else end
        for bounds in sentence_bounds(text, notes, stop):
            # A note is a requirement unless it only describes the shape of the result.
            path, cue, mixed = _suggest(text[bounds[0] : bounds[1]], CONSTRAINTS)
            units.append(Unit(path, bounds[0], bounds[1], "judgement", "B6", cue, mixed))
    for name, path, rule in (
        ("exc", CONSTRAINTS, "B1"),
        ("out", FORMAT_LEAF, "B2"),
        ("lead", CONSTRAINTS, "B3"),
    ):
        if match.group(name):
            block = _trimmed(text, *match.span(name))
            assert block is not None
            units.append(Unit(path, block[0], block[1], "formula", rule))
    code_start, code_end = match.span("code")
    units.append(Unit(DATA, code_start, code_end, "formula", "B4"))
    return sorted(units, key=lambda unit: unit.start)


def llmseceval_units(text: str) -> list[Unit]:
    """Propose the spans of an LLMSecEval prompt (rule ``lse-prefix-v1``).

    The fixed opening line and the first sentence after it are the main
    instruction. Every later sentence is a judgement unit with a suggested
    leaf, the main instruction when no wording suggests otherwise.

    Raises:
        ValueError: If the prompt does not start with the benchmark's opening line.
    """
    if not text.startswith(_LSE_PREFIX):
        raise ValueError("the prompt does not start with the LLMSecEval opening line")
    units = [Unit(MAIN, 0, len(_LSE_PREFIX), "formula", "B10")]
    for index, bounds in enumerate(sentence_bounds(text, len(_LSE_PREFIX), len(text), glued=True)):
        if index == 0:
            units.append(Unit(MAIN, bounds[0], bounds[1], "first", "G3.1"))
        else:
            units.append(_judgement(text, bounds, MAIN, "G3"))
    return units


RULES: Final = {
    BCB_RULE: (BIGCODEBENCH, bigcodebench_units),
    LSE_RULE: (LLMSECEVAL, llmseceval_units),
}


def proposed_units(source: SourceRecord, rule: str) -> list[Unit]:
    """Apply a named rule to a source of the benchmark the rule was written for."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose one of {sorted(RULES)}")
    benchmark, propose = RULES[rule]
    if source.benchmark != benchmark:
        raise ValueError(f"{source.task_id}: rule {rule} is for {benchmark} prompts")
    try:
        return propose(source.text)
    except ValueError as error:
        raise ValueError(f"{source.task_id}: {error}") from error


def propose(source: SourceRecord, rule: str) -> AnnotationRow:
    """Build a draft annotation from a rule; the result always has status ``draft``."""
    units = proposed_units(source, rule)
    return AnnotationRow(
        task_id=source.task_id,
        source_sha256=source.source_sha256,
        annotation_version=ANNOTATION_VERSION,
        guidelines_version=GUIDELINES_VERSION,
        annotator=f"rule:{rule}",
        reviewer=None,
        status="draft",
        spans=tuple(
            SpanAnnotation(unit.path, unit.start, unit.end, source.text[unit.start : unit.end])
            for unit in units
        ),
        notes="",
    )


def propose_bigcodebench(source: SourceRecord) -> AnnotationRow:
    """Draft a BigCodeBench annotation from the fixed markers."""
    return propose(source, BCB_RULE)


def propose_llmseceval(source: SourceRecord) -> AnnotationRow:
    """Draft an LLMSecEval annotation from the fixed opening line."""
    return propose(source, LSE_RULE)


# ------------------------------------------------------------------------------- lint


def _rule_for(source: SourceRecord) -> str:
    for rule, (benchmark, _) in RULES.items():
        if benchmark == source.benchmark:
            return rule
    raise ValueError(f"{source.task_id}: no rule for benchmark {source.benchmark!r}")


def lint(source: SourceRecord, row: AnnotationRow) -> list[str]:
    """Return the blocking defects of one annotation; an empty list means none.

    Checks: the hash of the source, the annotation version, exact coverage by
    the renderer, every fixed section present as one span on its leaf, span
    boundaries that never cut a sentence, and a note for every span on a leaf
    that is expected to stay empty. The leaf chosen for a judgement unit is the
    annotator's decision and is not checked.

    Rule ``g1-registered-cuts-v1`` (Q-16): a judgement unit (``G3`` or ``B6``)
    may be cut at a space, for example between the items of a list written on
    one line, when the notes record the cut as ``cut@<offset>``, the offset of
    the first character after the cut. The first sentence, marker blocks and
    fixed sections are never cut.
    """
    problems: list[str] = []
    if row.source_sha256 != source.source_sha256:
        return ["the source text is not the annotated one"]
    if row.annotation_version != ANNOTATION_VERSION:
        problems.append(f"annotation_version is not {ANNOTATION_VERSION}")
    prompt = CanonicalPrompt(row.task_id, source.text, row.spans, row.annotation_version or "?")
    try:
        render_prompt_variants(prompt)
    except ValueError as error:
        return [*problems, f"renderer: {error}"]
    units = proposed_units(source, _rule_for(source))
    spans = {(span.start, span.end): span.path for span in row.spans}
    for unit in units:
        if unit.kind == "formula" and spans.get((unit.start, unit.end)) != unit.path:
            section = source.text[unit.start : unit.end].split("\n")[0][:40]
            problems.append(f"fixed section {unit.rule} ({section!r}) must be one {unit.path} span")
    # A span may join whole sentences, never start or end inside one, except at
    # a registered cut.
    starts = {unit.start for unit in units}
    ends = {unit.end for unit in units}
    text = source.text
    cuts = {int(found) for found in _CUT.findall(row.notes)}
    judged = [unit for unit in units if unit.kind == "judgement"]
    for cut in sorted(cuts):
        valid = (
            0 < cut < len(text)
            and text[cut - 1].isspace()
            and not text[cut].isspace()
            and any(unit.start < cut < unit.end for unit in judged)
        )
        if not valid:
            problems.append(f"registered cut@{cut} is not at a space inside a judgement unit")
    admitted = {cut for cut in cuts if any(unit.start < cut < unit.end for unit in judged)}

    def _next_start(end: int) -> int:
        while end < len(text) and text[end].isspace():
            end += 1
        return end

    for span in row.spans:
        opens = span.start in starts or span.start in admitted
        closes = span.end in ends or (
            not text[span.end - 1].isspace() and _next_start(span.end) in admitted
        )
        if not (opens and closes):
            problems.append(f"span {span.start}-{span.end} cuts a sentence or a fixed section")
        if span.path in RARE_LEAVES and not row.notes.strip():
            problems.append(f"span {span.start}-{span.end} uses {span.path} without a note")
    if not set(LEAF_PATHS) >= {span.path for span in row.spans}:
        problems.append("unknown leaf")
    return problems
