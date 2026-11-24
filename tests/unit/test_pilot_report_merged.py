"""The pilot report of one journal read from the check runs of the merged files.

With two journals in a session, check-bigcodebench runs on the merged files
(generations-core.jsonl for the common conditions, generations-extra.jsonl for
H-auto), so the outcomes of one journal are spread over two ledgers that also
hold the other journal. ``restrict=True`` keeps only the cells and requests of
the journal; the exact coverage check still applies after the restriction.
"""

import json
from typing import Any

import pytest
from test_pilot_report_five import CONFIGURATION, evaluation, generation

from pstparser.empirical.auto import as_hauto
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.pilot_report import pilot_report
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants

CORE = ["T", "H-gold", "E"]
S_PARSE = [f"BigCodeBench/{i}" for i in range(3)]
OTHERS = [f"BigCodeBench/{i}" for i in range(3, 6)]
PASSING = {"T": {0, 3}, "H-gold": {0, 1, 4}, "E": {2}, "H-auto": {1}}


def variants(tasks: list[str], hauto: bool) -> list:
    rows = []
    for task in tasks:
        text = f"Solve {task}. Keep it short."
        cut = text.index("Keep")
        prompt = CanonicalPrompt(
            task,
            text,
            (
                SpanAnnotation("main_instruction", 0, cut - 1, text[: cut - 1]),
                SpanAnnotation("context.constraints", cut, len(text), text[cut:]),
            ),
            "v",
        )
        t, h = render_prompt_variants(prompt)
        rows += [t, h, render_envelope_variant(prompt)]
        if hauto:
            rows.append(as_hauto(h))
    return rows


def journal(tmp_path, name: str, tasks: list[str], conditions: list[str]):
    directory = tmp_path / name
    plan = request_plan(variants(tasks, "H-auto" in conditions), [0], 7319, conditions)
    run_journal(plan, CONFIGURATION, directory, generation)
    manifest = json.loads((directory / "empirical-generate_manifest.json").read_text("utf-8"))
    rows = [
        json.loads(x) for x in (directory / "generations.jsonl").read_text("utf-8").split("\n") if x
    ]
    return manifest, rows


def session(tmp_path) -> dict[str, Any]:
    a = journal(tmp_path, "a", S_PARSE, [*CORE, "H-auto"])
    b = journal(tmp_path, "b", OTHERS, CORE)
    core_rows = [r for r in a[1] + b[1] if r["condition"] in CORE]
    extra_rows = [r for r in a[1] if r["condition"] == "H-auto"]
    return {
        "a": a,
        "b": b,
        "core": evaluation(core_rows, PASSING),
        "extra": evaluation(extra_rows, PASSING),
    }


def test_each_journal_is_read_from_the_merged_check_runs(tmp_path) -> None:
    s = session(tmp_path)
    outcomes = s["core"][0] + s["extra"][0]
    details = s["core"][1] + s["extra"][1]
    for name in ("a", "b"):
        manifest, rows = s[name]
        direct = pilot_report(manifest, rows, *evaluation(rows, PASSING))
        merged = pilot_report(manifest, rows, outcomes, details, restrict=True)
        assert merged == direct


def test_without_restriction_the_merged_runs_are_refused(tmp_path) -> None:
    s = session(tmp_path)
    manifest, rows = s["a"]
    with pytest.raises(ValueError, match="cover the recorded journal exactly"):
        pilot_report(manifest, rows, s["core"][0] + s["extra"][0], s["core"][1] + s["extra"][1])


def test_a_cell_missing_from_the_merged_runs_still_stops_the_report(tmp_path) -> None:
    s = session(tmp_path)
    manifest, rows = s["a"]
    outcomes = [
        r for r in s["core"][0] + s["extra"][0] if (r.task_id, r.condition) != (S_PARSE[0], "E")
    ]
    with pytest.raises(ValueError, match="outcomes do not cover"):
        pilot_report(manifest, rows, outcomes, s["core"][1] + s["extra"][1], restrict=True)


def test_a_cell_judged_twice_stops_the_report(tmp_path) -> None:
    s = session(tmp_path)
    manifest, rows = s["a"]
    outcomes = s["core"][0] + s["extra"][0] + s["extra"][0][:1]
    with pytest.raises(ValueError, match="outcomes do not cover"):
        pilot_report(manifest, rows, outcomes, s["core"][1] + s["extra"][1], restrict=True)


def test_the_command_reads_the_core_and_extra_check_runs_with_merged(tmp_path) -> None:
    from typer.testing import CliRunner

    from pstparser.empirical.analysis import write_outcomes
    from pstparser.empirical.cli import app

    s = session(tmp_path)
    files: list[str] = []
    for name in ("core", "extra"):
        outcomes, details = s[name]
        write_outcomes(outcomes, tmp_path / f"{name}-outcomes.jsonl")
        (tmp_path / f"{name}-details.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in details), encoding="utf-8"
        )
        files += ["--outcomes", str(tmp_path / f"{name}-outcomes.jsonl")]
        files += ["--details", str(tmp_path / f"{name}-details.jsonl")]
    out = tmp_path / "report" / "pilot-report.json"
    base = ["pilot-report", str(tmp_path / "a"), "--out", str(out), *files]
    refused = CliRunner().invoke(app, base)
    assert refused.exit_code != 0  # several check runs only with --merged
    result = CliRunner().invoke(app, [*base, "--merged"])
    assert result.exit_code == 0, result.exception
    manifest, rows = s["a"]
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == json.loads(
        json.dumps(pilot_report(manifest, rows, *evaluation(rows, PASSING)))
    )
    record = json.loads((out.parent / "empirical-pilot-report_manifest.json").read_text("utf-8"))
    assert record["merged"] is True
    assert len(record["outcomes_sha256"]) == len(record["details_sha256"]) == 2


def test_the_restriction_needs_an_evaluation(tmp_path) -> None:
    s = session(tmp_path)
    manifest, rows = s["a"]
    with pytest.raises(ValueError, match="restrict"):
        pilot_report(manifest, rows, restrict=True)
