"""Gold annotation files: drafts from rules, validation, join to the sources, lint."""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pstparser.empirical.annotations import (
    AnnotationRow,
    canonicalize,
    lint,
    load_annotations,
    propose,
    propose_bigcodebench,
    propose_llmseceval,
    proposed_units,
    sentence_bounds,
    write_annotations,
)
from pstparser.empirical.benchmarks import SourceRecord, write_sources
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import SpanAnnotation
from pstparser.empirical.rendering import render_prompt_variants

CODE = "```\nimport os\nLIMIT = 3\ndef task_func(x):\n```"
BCB_TEXT = (
    "Compute the sum of a list. It must not modify the input, e.g. by sorting it. "
    "The result is scaled by 2.5 afterwards.\n"
    "Note that: Notes: The title is 'Sum'. Empty lists never fail.\n"
    "The function should raise the exception for: ValueError: If the list is empty.\n"
    "The function should output with:\n    float: The sum. None if nothing was added.\n"
    "You should write self-contained code starting with:\n" + CODE
)
BCB_EXAMPLE_TEXT = (
    "Add two numbers. Args: - a (int): the first one.\n"
    "Example 2: >>> task_func(1)\n2\n"
    "The function should output with:\n    int: The total.\n"
    "You should write self-contained code starting with:\n" + CODE
)
LSE_TEXT = (
    "Generate Python code for the following: \n The code reads a file.The file must exist. "
    "It returns the content.  Then it sends everything to <url>."
)


def source(task: str, text: str, benchmark: str = "bigcodebench") -> SourceRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceRecord(
        task_id=task,
        text=text,
        source_sha256=digest,
        original_sha256=digest,
        benchmark=benchmark,
        dataset_revision="1" * 40,
        dataset_file_sha256="d" * 64,
        original_id=task,
        family_id=None,
        text_field="instruct_prompt",
        transforms=(),
    )


BCB = source("BigCodeBench/1", BCB_TEXT)
BCB_EXAMPLE = source("BigCodeBench/2", BCB_EXAMPLE_TEXT)
LSE = source("LSE-001", LSE_TEXT, "llmseceval")


def described(record: SourceRecord, rule: str) -> list[tuple[str, str, str, str, str | None]]:
    return [
        (unit.path, record.text[unit.start : unit.end], unit.kind, unit.rule, unit.cue)
        for unit in proposed_units(record, rule)
    ]


# -------------------------------------------------------------------------------- rules


def test_bigcodebench_draft_follows_the_fixed_sections_and_suggests_the_rest() -> None:
    assert described(BCB, "bcb-markers-v1") == [
        ("main_instruction", "Compute the sum of a list.", "first", "G3.1", None),
        (
            "context.constraints",
            "It must not modify the input, e.g. by sorting it.",
            "judgement",
            "G3",
            "must",
        ),
        ("main_instruction", "The result is scaled by 2.5 afterwards.", "judgement", "G3", None),
        ("context.format", "Note that: Notes: The title is 'Sum'.", "judgement", "B6", "title"),
        ("context.constraints", "Empty lists never fail.", "judgement", "B6", "never"),
        (
            "context.constraints",
            "The function should raise the exception for: ValueError: If the list is empty.",
            "formula",
            "B1",
            None,
        ),
        (
            "context.format",
            "The function should output with:\n    float: The sum. None if nothing was added.",
            "formula",
            "B2",
            None,
        ),
        (
            "context.constraints",
            "You should write self-contained code starting with:",
            "formula",
            "B3",
            None,
        ),
        ("context.data", CODE, "formula", "B4", None),
    ]


def test_parameter_block_and_doctest_example_are_found_by_their_markers() -> None:
    assert described(BCB_EXAMPLE, "bcb-markers-v1")[:3] == [
        ("main_instruction", "Add two numbers.", "first", "G3.1", None),
        ("context.data", "Args: - a (int): the first one.", "marker", "B8", None),
        ("examples", "Example 2: >>> task_func(1)\n2", "marker", "B7", None),
    ]


def test_starting_code_is_one_data_span_equal_to_the_fenced_block() -> None:
    row = propose_bigcodebench(BCB)

    data = [span for span in row.spans if span.path == "context.data"]
    assert len(data) == 1
    assert data[0].phrase == CODE
    assert BCB_TEXT[data[0].start : data[0].end] == CODE
    assert data[0].end == len(BCB_TEXT)


def test_llmseceval_draft_keeps_the_opening_line_and_splits_glued_sentences() -> None:
    assert described(LSE, "lse-prefix-v1") == [
        ("main_instruction", "Generate Python code for the following:", "formula", "B10", None),
        ("main_instruction", "The code reads a file.", "first", "G3.1", None),
        ("context.constraints", "The file must exist.", "judgement", "G3", "must"),
        ("context.format", "It returns the content.", "judgement", "G3", "returns"),
        ("main_instruction", "Then it sends everything to <url>.", "judgement", "G3", None),
    ]


def test_sentence_with_both_wordings_is_flagged_as_mixed() -> None:
    text = (
        "Plot the data. The title must be 'A'.\n"
        "The function should output with:\n    Axes: The plot.\n"
        "You should write self-contained code starting with:\n" + CODE
    )

    unit = proposed_units(source("BigCodeBench/3", text), "bcb-markers-v1")[1]

    assert (unit.path, unit.cue, unit.mixed) == ("context.constraints", "must", True)


def test_sentence_splitter_boundaries() -> None:
    text = "Use os.path.join here, i.e. always. Keep 2.5 apart! Done? Yes.No"

    plain = [text[a:b] for a, b in sentence_bounds(text, 0, len(text))]
    glued = [text[a:b] for a, b in sentence_bounds(text, 0, len(text), glued=True)]

    assert plain == ["Use os.path.join here, i.e. always.", "Keep 2.5 apart!", "Done?", "Yes.No"]
    assert glued == [
        "Use os.path.join here, i.e. always.",
        "Keep 2.5 apart!",
        "Done?",
        "Yes.",
        "No",
    ]
    assert sentence_bounds("  \n ", 0, 4) == []


def test_drafts_are_drafts_and_carry_the_rule_and_the_source_hash() -> None:
    for row, rule in (
        (propose_bigcodebench(BCB), "bcb-markers-v1"),
        (propose_llmseceval(LSE), "lse-prefix-v1"),
    ):
        assert row.status == "draft"
        assert row.reviewer is None
        assert row.annotator == f"rule:{rule}"
        assert row.annotation_version == row.guidelines_version == "gold-v2.0"
    assert propose_bigcodebench(BCB).source_sha256 == BCB.source_sha256


def test_rules_refuse_a_prompt_of_another_shape_or_benchmark() -> None:
    with pytest.raises(ValueError, match=r"BigCodeBench/9.*structure"):
        propose_bigcodebench(source("BigCodeBench/9", "Just do it."))
    with pytest.raises(ValueError, match=r"LSE-9.*opening line"):
        propose_llmseceval(source("LSE-9", "Write code.", "llmseceval"))
    with pytest.raises(ValueError, match="is for llmseceval"):
        propose(BCB, "lse-prefix-v1")
    with pytest.raises(ValueError, match="unknown rule"):
        propose(BCB, "parser")


def test_drafts_pass_the_renderer_and_rebuild_the_source() -> None:
    sources = [BCB, BCB_EXAMPLE]
    rows = [propose_bigcodebench(record) for record in sources]

    prompts = canonicalize(sources, rows, require_status="draft")

    for record, prompt in zip(sources, prompts, strict=True):
        traditional, hierarchical = render_prompt_variants(prompt)
        assert traditional.text == record.text
        tree = json.loads(hierarchical.text)["prompt"]
        phrases = [
            item
            for value in tree.values()
            for leaf in (value.values() if isinstance(value, dict) else [value])
            for item in leaf
        ]
        rebuilt = "".join(item["phrase"] for item in sorted(phrases, key=lambda i: i["position"]))
        assert rebuilt == record.text
    code = json.loads(render_prompt_variants(prompts[0])[1].text)["prompt"]["context"]["data"]
    assert code == [{"position": 8, "phrase": CODE}]


# --------------------------------------------------------------------------------- join


def reviewed(row: AnnotationRow, status: str = "frozen", **changes: Any) -> AnnotationRow:
    return dataclasses.replace(row, status=status, reviewer="R1", annotator="A1", **changes)


def test_a_draft_does_not_pass_for_frozen_gold() -> None:
    draft = propose_bigcodebench(BCB)

    with pytest.raises(ValueError, match=r"BigCodeBench/1: status is 'draft', 'frozen' required"):
        canonicalize([BCB], [draft])
    with pytest.raises(ValueError, match="'reviewed', 'frozen' required"):
        canonicalize([BCB], [reviewed(draft, "reviewed")])
    assert canonicalize([BCB], [reviewed(draft)])[0].text == BCB_TEXT


def test_a_source_changed_by_one_character_is_refused() -> None:
    frozen = reviewed(propose_bigcodebench(BCB))
    changed = source("BigCodeBench/1", BCB_TEXT.replace("sum of a list", "sum of a lisT"))

    with pytest.raises(ValueError, match="not the annotated one"):
        canonicalize([changed], [frozen])


def test_join_needs_exactly_the_sample() -> None:
    first = reviewed(propose_bigcodebench(BCB))
    second = reviewed(propose_bigcodebench(BCB_EXAMPLE))
    sources = [BCB, BCB_EXAMPLE]

    assert (
        len(canonicalize(sources, [first, second], sample_ids=[BCB.task_id, BCB_EXAMPLE.task_id]))
        == 2
    )
    with pytest.raises(ValueError, match=r"missing \['BigCodeBench/2'\]"):
        canonicalize(sources, [first], sample_ids=[BCB.task_id, BCB_EXAMPLE.task_id])
    with pytest.raises(ValueError, match=r"extra \['BigCodeBench/2'\]"):
        canonicalize(sources, [first, second], sample_ids=[BCB.task_id])
    with pytest.raises(ValueError, match="duplicate annotation"):
        canonicalize(sources, [first, first])
    with pytest.raises(ValueError, match="no such source"):
        canonicalize([BCB], [first, second])
    with pytest.raises(ValueError, match="more than one annotation version"):
        canonicalize(
            sources,
            [first, reviewed(propose_bigcodebench(BCB_EXAMPLE), annotation_version="gold-v2.1")],
        )


def without(row: AnnotationRow, index: int) -> AnnotationRow:
    return dataclasses.replace(row, spans=row.spans[:index] + row.spans[index + 1 :])


def test_gap_with_content_and_overlap_are_refused() -> None:
    frozen = reviewed(propose_bigcodebench(BCB))
    first = frozen.spans[0]
    overlapping = dataclasses.replace(
        frozen,
        spans=(
            SpanAnnotation(first.path, first.start, first.end + 4, BCB_TEXT[: first.end + 4]),
            *frozen.spans[1:],
        ),
    )

    with pytest.raises(ValueError, match=r"BigCodeBench/1: Unannotated non-whitespace"):
        canonicalize([BCB], [without(frozen, 2)])
    with pytest.raises(ValueError, match=r"BigCodeBench/1: Overlapping"):
        canonicalize([BCB], [overlapping])


# --------------------------------------------------------------------------------- file


def test_annotation_file_round_trip_and_format(tmp_path: Path) -> None:
    rows = [propose_bigcodebench(BCB), reviewed(propose_bigcodebench(BCB_EXAMPLE), "reviewed")]
    path = tmp_path / "gold" / "annotations.jsonl"

    write_annotations(rows, path)

    assert load_annotations(path) == rows
    first = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
    assert list(first) == [
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
    ]
    assert first["format"] == "pst-gold-annotations-v1" and first["status"] == "draft"
    assert set(first["spans"][0]) == {"path", "start", "end", "phrase"}
    with pytest.raises(FileExistsError):
        write_annotations(rows, path)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"status": "done"}, "status must be one of"),
        ({"status": "reviewed"}, "a draft has no reviewer"),
        ({"reviewer": "R1"}, "a draft has no reviewer"),
        ({"format": "pst-gold-annotations-v0"}, "unknown format"),
        ({"source_sha256": "abc"}, "SHA-256"),
        ({"annotator": " "}, "annotator must be"),
        ({"extra": 1}, "exactly the keys"),
        ({"spans": [{"path": "main_instruction", "start": 0, "end": 3}]}, "exactly path, start"),
    ],
)
def test_malformed_rows_are_refused(change: dict[str, Any], message: str, tmp_path: Path) -> None:
    record = {**propose_bigcodebench(BCB).as_dict(), **change}
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_annotations(path)


def test_status_is_mandatory_and_repeated_tasks_are_refused(tmp_path: Path) -> None:
    record = propose_bigcodebench(BCB).as_dict()
    missing = {key: value for key, value in record.items() if key != "status"}
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the keys"):
        load_annotations(path)
    twice = tmp_path / "twice.jsonl"
    twice.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate annotation"):
        load_annotations(twice)


# --------------------------------------------------------------------------------- lint


def test_rule_drafts_have_no_blocking_defect() -> None:
    assert lint(BCB, propose_bigcodebench(BCB)) == []
    assert lint(BCB_EXAMPLE, propose_bigcodebench(BCB_EXAMPLE)) == []
    assert lint(LSE, propose_llmseceval(LSE)) == []


def relabel(row: AnnotationRow, index: int, path: str) -> AnnotationRow:
    spans = list(row.spans)
    spans[index] = dataclasses.replace(spans[index], path=path)
    return dataclasses.replace(row, spans=tuple(spans))


def test_changing_the_leaf_of_a_judgement_sentence_is_allowed() -> None:
    row = relabel(propose_bigcodebench(BCB), 2, "context.format")

    assert lint(BCB, row) == []


def test_lint_reports_a_fixed_section_on_another_leaf() -> None:
    row = relabel(propose_bigcodebench(BCB), 6, "context.constraints")

    assert lint(BCB, row) == [
        "fixed section B2 ('The function should output with:') must be one context.format span"
    ]


def test_lint_reports_a_split_starting_code_block() -> None:
    row = propose_bigcodebench(BCB)
    code = row.spans[-1]
    cut = code.start + CODE.index("LIMIT")
    halves = (
        SpanAnnotation("context.data", code.start, cut - 1, BCB_TEXT[code.start : cut - 1]),
        SpanAnnotation("context.data", cut, code.end, BCB_TEXT[cut : code.end]),
    )
    split = dataclasses.replace(row, spans=(*row.spans[:-1], *halves))

    problems = lint(BCB, split)

    assert problems[0].startswith("fixed section B4") and "one context.data span" in problems[0]
    assert sum("cuts a sentence or a fixed section" in problem for problem in problems) == 2


def test_lint_reports_a_span_that_cuts_a_sentence_but_accepts_joined_sentences() -> None:
    row = propose_bigcodebench(BCB)
    second, third = row.spans[1], row.spans[2]
    joined = SpanAnnotation(
        "context.constraints", second.start, third.end, BCB_TEXT[second.start : third.end]
    )
    merged = dataclasses.replace(row, spans=(row.spans[0], joined, *row.spans[3:]))
    middle = second.start + 10
    cut = dataclasses.replace(
        row,
        spans=(
            row.spans[0],
            SpanAnnotation(
                "context.constraints", second.start, middle, BCB_TEXT[second.start : middle]
            ),
            SpanAnnotation("main_instruction", middle, second.end, BCB_TEXT[middle : second.end]),
            *row.spans[2:],
        ),
    )

    assert lint(BCB, merged) == []
    assert lint(BCB, cut) == [
        f"span {second.start}-{middle} cuts a sentence or a fixed section",
        f"span {middle}-{second.end} cuts a sentence or a fixed section",
    ]


def test_lint_reports_a_rare_leaf_without_a_note_and_other_basic_defects() -> None:
    role = relabel(propose_bigcodebench(BCB), 2, "context.role")
    start, end = role.spans[2].start, role.spans[2].end

    assert lint(BCB, role) == [f"span {start}-{end} uses context.role without a note"]
    assert lint(BCB, dataclasses.replace(role, notes="G4: the text assigns a persona.")) == []
    assert lint(
        BCB, dataclasses.replace(propose_bigcodebench(BCB), annotation_version="gold-1.0")
    ) == ["annotation_version is not gold-v2.0"]
    assert lint(BCB, without(propose_bigcodebench(BCB), 2)) == [
        "renderer: Unannotated non-whitespace source content"
    ]
    assert lint(BCB_EXAMPLE, propose_bigcodebench(BCB)) == [
        "the source text is not the annotated one"
    ]


# ----------------------------------------------------------------------------- commands


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    sources = tmp_path / "in" / "sources.jsonl"
    write_sources([BCB, BCB_EXAMPLE], sources)
    sample = tmp_path / "in" / "sample.json"
    sample.write_text(
        json.dumps({"kind": "pilot-sample", "tasks": [{"task_id": BCB_EXAMPLE.task_id}]}),
        encoding="utf-8",
    )
    return {"sources": sources, "sample": sample}


def test_propose_command_writes_drafts_for_the_sample_only(
    files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "drafts" / "pilot.annotations.jsonl"
    arguments = [
        "propose-spans",
        str(files["sources"]),
        "--rule",
        "bcb-markers-v1",
        "--sample",
        str(files["sample"]),
        "--out",
        str(out),
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    rows = load_annotations(out)
    assert [row.task_id for row in rows] == [BCB_EXAMPLE.task_id]
    assert rows == [propose_bigcodebench(BCB_EXAMPLE)]
    manifest = json.loads(
        (out.parent / "empirical-propose-spans_manifest.json").read_text(encoding="utf-8")
    )
    assert (manifest["rule"], manifest["status"], manifest["rows"]) == (
        "bcb-markers-v1",
        "draft",
        1,
    )
    assert manifest["splitter"] == "g1-splitter-v1"
    assert manifest["annotations_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


def test_canonicalize_command_refuses_drafts_and_feeds_the_render_command(
    files: dict[str, Path], tmp_path: Path
) -> None:
    drafts = tmp_path / "gold" / "drafts.jsonl"
    write_annotations([propose_bigcodebench(BCB_EXAMPLE)], drafts)
    frozen = tmp_path / "gold" / "frozen.jsonl"
    write_annotations([reviewed(propose_bigcodebench(BCB_EXAMPLE))], frozen)
    out = tmp_path / "canonical" / "pilot.canonical.jsonl"

    def run(annotation_file: Path) -> Any:
        return CliRunner().invoke(
            app,
            [
                "canonicalize",
                str(files["sources"]),
                str(annotation_file),
                "--sample",
                str(files["sample"]),
                "--out",
                str(out),
            ],
        )

    refused = run(drafts)
    assert refused.exit_code != 0
    assert "status is 'draft', 'frozen' required" in str(refused.exception)
    assert not out.exists()
    accepted = run(frozen)
    assert accepted.exit_code == 0, accepted.output
    variants = tmp_path / "canonical" / "variants.jsonl"
    rendered = CliRunner().invoke(app, ["render", str(out), str(variants)])
    assert rendered.exit_code == 0, rendered.output
    lines = [json.loads(line) for line in variants.read_text(encoding="utf-8").split("\n") if line]
    assert [(line["condition"], line["source_hash"]) for line in lines] == [
        ("T", BCB_EXAMPLE.source_sha256),
        ("H-gold", BCB_EXAMPLE.source_sha256),
    ]
    assert lines[0]["text"] == BCB_EXAMPLE_TEXT


def test_gold_lint_command_lists_defects_and_fails(files: dict[str, Path], tmp_path: Path) -> None:
    good = tmp_path / "gold" / "good.jsonl"
    write_annotations([propose_bigcodebench(BCB), propose_bigcodebench(BCB_EXAMPLE)], good)
    bad = tmp_path / "gold" / "bad.jsonl"
    write_annotations([relabel(propose_bigcodebench(BCB), 6, "context.constraints")], bad)

    clean = CliRunner().invoke(app, ["gold-lint", str(files["sources"]), str(good)])
    dirty = CliRunner().invoke(app, ["gold-lint", str(files["sources"]), str(bad)])

    assert clean.exit_code == 0
    assert "2 annotations, states ['draft'], 0 defects" in clean.output
    assert dirty.exit_code == 1
    assert "BigCodeBench/1: fixed section B2" in dirty.output
    assert "1 defects" in dirty.output
