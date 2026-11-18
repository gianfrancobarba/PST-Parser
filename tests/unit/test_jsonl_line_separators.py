"""JSON Lines readers split on the line feed only (R-PRO-01).

``str.splitlines`` also breaks on U+2028, U+2029 and U+0085, which
``json.dumps(..., ensure_ascii=False)`` writes unescaped inside strings: a
prompt holding one of them must load back intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pstparser.data import Split, load_split, save_split
from pstparser.data.alignments import AlignmentRecord, load_alignments, write_alignments
from pstparser.data.predictions import PredictionRecord, load_predictions, write_predictions
from pstparser.data.prepare import PreparedRecord, load_records, write_records

SEPARATORS = ["\u2028", "\u2029", "\x85"]


@pytest.mark.parametrize("separator", SEPARATORS)
def test_predictions_keep_a_prompt_with_a_unicode_line_separator(
    tmp_path: Path, separator: str
) -> None:
    prompt = f"first{separator}second"
    path = write_predictions(
        [PredictionRecord(0, prompt, "{}", "{}"), PredictionRecord(1, "plain", "{}", "{}")],
        tmp_path / "predictions.jsonl",
    )

    loaded = load_predictions(path)

    assert [record.prompt for record in loaded] == [prompt, "plain"]


@pytest.mark.parametrize("separator", SEPARATORS)
def test_alignments_keep_a_prompt_with_a_unicode_line_separator(
    tmp_path: Path, separator: str
) -> None:
    prompt = f"first{separator}second"
    path = write_alignments(
        [AlignmentRecord(0, prompt, None, None, None, None)], tmp_path / "alignments.jsonl"
    )

    assert [record.prompt for record in load_alignments(path)] == [prompt]


@pytest.mark.parametrize("separator", SEPARATORS)
def test_prepared_records_keep_a_prompt_with_a_unicode_line_separator(
    tmp_path: Path, separator: str
) -> None:
    prompt = f"first{separator}second"
    path = write_records([PreparedRecord(0, prompt, "{}")], tmp_path / "records.jsonl")

    assert [record.prompt for record in load_records(path)] == [prompt]


@pytest.mark.parametrize("separator", SEPARATORS)
def test_a_split_bound_to_records_with_a_unicode_line_separator_loads(
    tmp_path: Path, separator: str
) -> None:
    records = write_records(
        [PreparedRecord(index, f"prompt{separator}{index}", "{}") for index in range(3)],
        tmp_path / "records.jsonl",
    )
    save_split(Split(train=[0], val=[1], test=[2]), tmp_path / "split", records_path=records)

    split = load_split(tmp_path / "split", records_path=records)

    assert (split.train, split.val, split.test) == ([0], [1], [2])
