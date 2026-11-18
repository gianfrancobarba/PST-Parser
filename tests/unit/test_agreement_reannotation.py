"""The re-annotation from scratch is a human second file, at a recorded interval (P-07)."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from pstparser.empirical import cli
from pstparser.empirical.annotations import BCB_RULE, propose, write_annotations
from pstparser.empirical.benchmarks import SourceRecord, write_sources

TEXT = (
    "Compute the sum of a list. It must not modify the input.\n"
    "The function should output with:\n    float: The sum.\n"
    "You should write self-contained code starting with:\n"
    "```\nimport os\ndef task_func(x):\n```"
)
DIGEST = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
BCB = SourceRecord(
    "BigCodeBench/1",
    TEXT,
    DIGEST,
    DIGEST,
    "bigcodebench",
    "1" * 40,
    "d" * 64,
    "BigCodeBench/1",
    None,
    "instruct_prompt",
    (),
)


def _run(tmp_path: Path, annotator: str, first_date: str, second_date: str):
    sources = tmp_path / "sources.jsonl"
    write_sources([BCB], sources)
    first = [propose(BCB, BCB_RULE)]
    second = [replace(first[0], annotator=annotator)]
    write_annotations(first, tmp_path / "a.jsonl")
    write_annotations(second, tmp_path / "b.jsonl")
    out = tmp_path / "agree" / "agreement.json"
    arguments = [
        "agreement",
        str(sources),
        str(tmp_path / "a.jsonl"),
        str(tmp_path / "b.jsonl"),
        "--rule",
        BCB_RULE,
        "--first-date",
        first_date,
        "--second-date",
        second_date,
        "--out",
        str(out),
    ]
    return CliRunner().invoke(cli.app, arguments), out


def test_a_second_file_written_by_a_rule_is_refused(tmp_path: Path) -> None:
    result, out = _run(tmp_path, "rule:bcb-markers-v1", "2026-11-21", "2026-11-27")
    assert isinstance(result.exception, ValueError)
    assert "rule" in str(result.exception)
    assert not out.exists()


def test_the_interval_and_the_dates_are_in_the_report(tmp_path: Path) -> None:
    result, out = _run(tmp_path, "studente", "2026-11-21", "2026-11-27")
    assert result.exit_code == 0, result.exception
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["first_date"] == "2026-11-21"
    assert report["second_date"] == "2026-11-27"
    assert report["interval_days"] == 6


def test_a_second_date_before_the_first_is_refused(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "studente", "2026-11-27", "2026-11-21")
    assert isinstance(result.exception, ValueError)
    assert "date" in str(result.exception)
