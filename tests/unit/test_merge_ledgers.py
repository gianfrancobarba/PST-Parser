"""Union of the outcome ledgers of the check runs of one session, for hauto-ledgers (P8, C-26).

check-bigcodebench runs once on generations-core.jsonl (T, H-gold, E on S_gold)
and once on generations-extra.jsonl (H-auto on S_parse). hauto-ledgers wants one
ledger with T and H-auto under one analysis_id: merge-ledgers writes it, and
refuses mixed endpoints and cells judged twice.
"""

import json

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import OutcomeRecord, read_outcomes, write_outcomes
from pstparser.empirical.auto import hauto_ledgers
from pstparser.empirical.cli import app
from pstparser.empirical.ledgers import merge_ledgers

TASKS = [f"BigCodeBench/{i}" for i in range(4)]
S_PARSE = TASKS[:3]


def record(analysis: str, task: str, condition: str, outcome: str = "pass") -> OutcomeRecord:
    reason = "pass" if outcome == "pass" else "assertion_failure"
    return OutcomeRecord(analysis, "functional", task, condition, 0, outcome, reason)  # type: ignore[arg-type]


def core() -> list[OutcomeRecord]:
    return [
        record("core", t, c, "pass" if i % 2 else "fail")
        for i, t in enumerate(TASKS)
        for c in ("T", "H-gold", "E")
    ]


def extra() -> list[OutcomeRecord]:
    return [record("extra", t, "H-auto") for t in S_PARSE]


def test_the_union_feeds_hauto_ledgers() -> None:
    merged = merge_ledgers([core(), extra()], "bcb-hauto")
    assert {r.analysis_id for r in merged} == {"bcb-hauto"}
    assert len(merged) == len(core()) + len(extra())
    failures = [{"task_id": TASKS[3], "cause": "content_gap"}]
    primary, end_to_end = hauto_ledgers(merged, S_PARSE, failures)
    assert {r.task_id for r in primary} == set(S_PARSE)
    assert {(r.task_id, r.condition) for r in end_to_end} >= {(TASKS[3], "H-auto")}


def test_a_cell_judged_in_two_ledgers_is_refused() -> None:
    with pytest.raises(ValueError, match="more than once"):
        merge_ledgers([core(), extra(), extra()[:1]], "bcb-hauto")


def test_mixed_endpoints_are_refused() -> None:
    security = OutcomeRecord("sec", "security", TASKS[0], "T", 0, "S", "no_findings")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="endpoint"):
        merge_ledgers([core(), [security]], "bcb-hauto")


def test_an_empty_union_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        merge_ledgers([[], []], "bcb-hauto")


def test_the_command_writes_the_union_and_its_manifest(tmp_path) -> None:
    write_outcomes(core(), tmp_path / "core.jsonl")
    write_outcomes(extra(), tmp_path / "extra.jsonl")
    out = tmp_path / "merged" / "outcomes.jsonl"
    arguments = [
        "merge-ledgers",
        str(tmp_path / "core.jsonl"),
        str(tmp_path / "extra.jsonl"),
        "--analysis-id",
        "bcb-hauto",
        "--out",
        str(out),
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.exception
    assert read_outcomes(out) == merge_ledgers([core(), extra()], "bcb-hauto")
    manifest = json.loads((out.parent / "empirical-merge-ledgers_manifest.json").read_text("utf-8"))
    assert manifest["analysis_id"] == "bcb-hauto"
    assert len(manifest["ledgers_sha256"]) == 2
    assert CliRunner().invoke(app, arguments).exit_code != 0  # never overwritten
