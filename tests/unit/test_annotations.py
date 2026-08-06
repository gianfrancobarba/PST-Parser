"""Reading a corpus annotated as text."""

from __future__ import annotations

from pathlib import Path

import pytest

from pstparser.data import (
    PARADIGMS,
    CorpusError,
    read_annotations,
    write_annotation_skeleton,
)
from pstparser.pst import LEAF_PATHS

ASSET = Path("tests/assets/tiny_annotations.yaml")


def write(tmp_path: Path, body: str) -> Path:
    """Write a source file and return its path."""
    path = tmp_path / "annotations.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def one(tmp_path: Path, leaves: str) -> Path:
    """Write a source holding one record with the given leaves."""
    return write(
        tmp_path,
        "records:\n  - prompt: Rename the variable.\n    leaves:\n" + leaves,
    )


def test_records_are_read_in_file_order() -> None:
    records = read_annotations(ASSET)

    assert len(records) == 3
    assert records[0].prompt.startswith("You are a senior Go reviewer.")
    assert records[1].prompt.startswith("class A: pass")
    assert records[2].prompt.startswith("You are a log triage assistant.")


def test_a_prompt_is_read_verbatim() -> None:
    records = read_annotations(ASSET)

    # Internal blank lines survive, and a literal block leaves no trailing one.
    assert "\n\nfor range time.Tick(time.Second) { work() }\n\n" in records[0].prompt
    assert not records[0].prompt.endswith("\n")


def test_a_leaf_written_as_a_list_becomes_several_segments() -> None:
    records = read_annotations(ASSET)

    assert records[1].leaves["context.data"] == ["class A: pass", "class B: pass"]


def test_a_leaf_written_as_text_becomes_one_segment() -> None:
    records = read_annotations(ASSET)

    assert records[0].leaves["context.role"] == ["You are a senior Go reviewer."]


def test_every_declared_leaf_is_present() -> None:
    records = read_annotations(ASSET)

    assert set(records[0].leaves) == set(LEAF_PATHS)
    # A leaf the record does not mention holds nothing.
    assert records[0].leaves["examples"] == []


def test_the_paradigm_is_optional() -> None:
    records = read_annotations(ASSET)

    assert records[0].paradigm == "zero_shot_cot"
    assert records[1].paradigm is None


@pytest.mark.parametrize("paradigm", PARADIGMS)
def test_every_paradigm_is_accepted(tmp_path: Path, paradigm: str) -> None:
    path = write(tmp_path, f"records:\n  - prompt: Do it.\n    paradigm: {paradigm}\n")

    assert read_annotations(path)[0].paradigm == paradigm


@pytest.mark.parametrize("written", ["''", "[]", "null"])
def test_an_empty_leaf_holds_nothing(tmp_path: Path, written: str) -> None:
    path = one(tmp_path, f"      main_instruction: {written}\n")

    assert read_annotations(path)[0].leaves["main_instruction"] == []


def test_whitespace_around_a_segment_is_dropped(tmp_path: Path) -> None:
    path = one(tmp_path, '      main_instruction: ["  Rename the variable.  ", "   "]\n')

    assert read_annotations(path)[0].leaves["main_instruction"] == ["Rename the variable."]


def test_a_segment_carrying_the_separator_is_refused(tmp_path: Path) -> None:
    # It is what pasting a spreadsheet cell in here produces. Interpreting it
    # would import the very limitation this format exists to leave behind.
    path = one(tmp_path, "      context.data: first <sep> second\n")

    with pytest.raises(CorpusError, match="segment 0 carries '<sep>'"):
        read_annotations(path)


def test_a_column_heading_is_not_a_leaf_path(tmp_path: Path) -> None:
    # The delivered worksheet spells the restrictive node CONSTRAINS. A file
    # written here names leaves by their path, so the misspelling cannot travel.
    path = one(tmp_path, "      CONSTRAINS: Do not change the behaviour.\n")

    with pytest.raises(CorpusError, match="unknown leaf 'CONSTRAINS'"):
        read_annotations(path)


def test_a_leaf_holding_something_other_than_text_is_refused(tmp_path: Path) -> None:
    # Unlike a spreadsheet cell, whose type is not ours to choose, this is
    # written deliberately, so it is a mistake rather than a value to render.
    path = one(tmp_path, "      main_instruction: 42\n")

    with pytest.raises(CorpusError, match="must be text or a list of text, not int"):
        read_annotations(path)


def test_a_segment_that_is_not_text_is_refused(tmp_path: Path) -> None:
    path = one(tmp_path, "      main_instruction: [Rename it., 42]\n")

    with pytest.raises(CorpusError, match="segment 1 is not text"):
        read_annotations(path)


def test_a_repeated_leaf_is_refused(tmp_path: Path) -> None:
    # The parser keeps the last of them without a word, which on a file of
    # annotations is a leaf silently discarded.
    path = one(
        tmp_path,
        '      context.data: "class A: pass"\n      context.data: "class B: pass"\n',
    )

    with pytest.raises(CorpusError, match="declared twice"):
        read_annotations(path)


def test_an_unknown_record_key_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "records:\n  - promt: Rename the variable.\n")

    with pytest.raises(CorpusError, match="unknown keys: promt"):
        read_annotations(path)


@pytest.mark.parametrize("body", ["records:\n  - leaves: {}\n", "records:\n  - prompt: '  '\n"])
def test_a_record_without_a_prompt_is_refused(tmp_path: Path, body: str) -> None:
    with pytest.raises(CorpusError, match="'prompt' must be a non-empty string"):
        read_annotations(write(tmp_path, body))


def test_an_unknown_paradigm_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "records:\n  - prompt: Do it.\n    paradigm: cot\n")

    with pytest.raises(CorpusError, match="unknown paradigm 'cot'"):
        read_annotations(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="corpus not found"):
        read_annotations(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path, "records:\n  - prompt: [unclosed\n")

    with pytest.raises(CorpusError, match="is not valid YAML"):
        read_annotations(path)


def test_records_must_be_a_list(tmp_path: Path) -> None:
    path = write(tmp_path, "records:\n  prompt: Rename the variable.\n")

    with pytest.raises(CorpusError, match="'records' must be a list"):
        read_annotations(path)


def test_a_record_must_be_a_mapping(tmp_path: Path) -> None:
    path = write(tmp_path, "records:\n  - Rename the variable.\n")

    with pytest.raises(CorpusError, match="record 0 is not a mapping"):
        read_annotations(path)


def test_a_source_may_contribute_nothing(tmp_path: Path) -> None:
    assert read_annotations(write(tmp_path, "records: []\n")) == []


def test_a_skeleton_is_written_in_the_format_that_is_read_back(tmp_path: Path) -> None:
    # Awkward on purpose: a leading space, a trailing newline, tabs and a
    # carriage return are all things a literal block cannot always carry, and a
    # prompt that does not survive the round trip would fail the integrity check
    # later as though it had been annotated wrongly.
    written = [
        ("zero_shot_cot", "One line only."),
        ("few_shot_cot", "  leading space\nsecond line\n\nfourth"),
        ("tree_of_thoughts", "trailing newline\n"),
        ("zero_shot_cot", "tabs\there and a colon: value\r\n- and a dash"),
    ]

    path = write_annotation_skeleton(written, tmp_path / "nested" / "annotation.yaml")
    records = read_annotations(path)

    assert [(record.paradigm, record.prompt) for record in records] == written
    assert all(set(record.leaves) == set(LEAF_PATHS) for record in records)
    assert all(not any(record.leaves.values()) for record in records)
