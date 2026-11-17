"""Non-directional pilot report (M16): technical criteria G1-G14 and b + c, never an effect."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import OutcomeRecord, write_outcomes
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.pilot_report import FORBIDDEN_KEY_FRAGMENTS, pilot_report
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants

CONDITIONS = ("T", "H-gold", "E")
TASKS = [f"BigCodeBench/{i}" for i in range(10)]
CONFIGURATION = {
    "model": {"max_seq_length": 4096},
    "decoding": {"max_new_tokens": 1280, "repetition_penalty": 1.0},
    "purpose": "pilot",
    "runtime": {"revision_verified": True, "chat_template_matches_official": True},
    "environment": {"accelerator": {"devices": ["NVIDIA A100-SXM4-40GB"]}},
}
TRUNCATED = {("BigCodeBench/9", "H-gold"), ("BigCodeBench/8", "E")}
EMPTY = {("BigCodeBench/7", "T")}
CATEGORY = {
    ("BigCodeBench/5", "H-gold"): "json_response",
    ("BigCodeBench/6", "H-gold"): "prompt_as_data",
    ("BigCodeBench/6", "E"): "prompt_as_data",
}
NOT_SINGLE_BLOCK = {("BigCodeBench/2", "T"), ("BigCodeBench/3", "H-gold"), ("BigCodeBench/4", "E")}
FLAKY = ("BigCodeBench/1", "T")
# Two worlds with the same technical data, opposite directions and the same b + c.
WORLDS = {
    "A": {"T": {0, 1, 2}, "H-gold": {0, 1, 2, 3, 4}, "E": {0}},
    "B": {"T": {0, 1, 2, 3, 4}, "H-gold": {0, 1, 2}, "E": {0, 1, 2, 3}},
}


def variants() -> list:
    rows = []
    for task in TASKS:
        text = f"Solve {task}."
        prompt = CanonicalPrompt(
            task, text, (SpanAnnotation("main_instruction", 0, len(text), text),), "v"
        )
        rows += [*render_prompt_variants(prompt), render_envelope_variant(prompt)]
    return rows


def fake_generation(request: dict) -> dict[str, Any]:
    index = int(request["task_id"].split("/")[1])
    key = (request["task_id"], request["condition"])
    inputs = {"T": 100, "H-gold": 300 if index == 0 else 250, "E": 125}[request["condition"]]
    return {
        "raw_output": "```python\nx = 1\n```",
        "input_tokens": inputs,
        "output_tokens": 100 + 10 * index + (5 if request["condition"] == "H-gold" else 0),
        "finish_reason": "length" if key in TRUNCATED else "eos",
        "elapsed_seconds": 10.0 if key == ("BigCodeBench/3", "E") else 1.0,
        "peak_vram_bytes": 1000 + index,
    }


def journal_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "journal"
    plan = request_plan(variants(), [0], 7319, conditions=CONDITIONS)
    run_journal(plan, CONFIGURATION, directory, fake_generation)
    return directory


def read_directory(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "empirical-generate_manifest.json").read_text("utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "generations.jsonl").read_text("utf-8").split("\n")
        if line
    ]
    return manifest, rows


def functional(rows: list[dict], world: str) -> tuple[list[OutcomeRecord], list[dict]]:
    outcomes, details = [], []
    for row in rows:
        key = (row["task_id"], row["condition"])
        index = int(row["task_id"].split("/")[1])
        passed = index in WORLDS[world][row["condition"]]
        assert not passed or (key not in TRUNCATED | EMPTY and key not in CATEGORY)
        truncated, empty = key in TRUNCATED, key in EMPTY
        executed = not truncated and not empty
        reason = (
            "truncated"
            if truncated
            else "no_code"
            if empty
            else CATEGORY.get(key, "pass" if passed else "assertion_failure")
        )
        status = "pass" if passed else "fail"
        runs = [status, "fail", status] if key == FLAKY else [status, status]
        outcomes.append(
            OutcomeRecord(
                "pilot-bcb",
                "functional",
                row["task_id"],
                row["condition"],
                0,
                "pass" if passed else "fail",
                reason,
            )
        )
        details.append(
            {
                "request_id": row["request_id"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "replicate_id": 0,
                "extraction_policy": "bigcodebench-sanitize",
                "extraction_statuses": {
                    "python-single-block-v1": "no_code" if key in NOT_SINGLE_BLOCK else "ok",
                    "python-fenced-blocks-v2": "ok",
                    "python-longest-block-v2": "ok",
                },
                "sanitized_lines": 0 if empty else 3,
                "executed": executed,
                "outcome": "pass" if passed else "fail",
                "reason": reason,
                "run_statuses": runs if executed else [],
                "flaky": key == FLAKY,
                "finish_reason": row["finish_reason"],
            }
        )
    return outcomes, details


@pytest.fixture
def pilot(tmp_path):
    manifest, rows = read_directory(journal_directory(tmp_path))
    return manifest, rows


def report_for(pilot, world="A") -> dict:
    manifest, rows = pilot
    outcomes, details = functional(rows, world)
    return pilot_report(manifest, rows, outcomes, details)


def keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [k for key, item in value.items() for k in [key, *keys(item)]]
    if isinstance(value, list):
        return [k for item in value for k in keys(item)]
    return []


def test_no_forbidden_key_anywhere(pilot) -> None:
    report = report_for(pilot)
    assert FORBIDDEN_KEY_FRAGMENTS == (
        "delta",
        "diff",
        "baseline_only",
        "treatment_only",
        "pass",
        "success",
        "effect",
    )
    for key in keys(report):
        assert not any(fragment in key.lower() for fragment in FORBIDDEN_KEY_FRAGMENTS), key


def test_success_by_condition_cannot_be_reconstructed(pilot) -> None:
    # Same technical data, opposite direction, same b + c: the reports must coincide.
    first, second = report_for(pilot, "A"), report_for(pilot, "B")
    assert first == second
    text = json.dumps(first)
    for world in WORLDS.values():
        for condition, passed in world.items():
            assert f'"{condition}": {len(passed)}' not in text


def test_discordant_pairs_are_b_plus_c_of_t_and_h_gold(pilot) -> None:
    report = report_for(pilot)
    assert report["discordant_pairs"] == {
        "conditions": ["T", "H-gold"],
        "complete_pairs": 10,
        "discordant_total": 2,
        "discordant_rate": 0.2,
    }


def test_journal_completeness_resume_rendering_and_budget(pilot) -> None:
    report = report_for(pilot)
    assert report["report_version"] == "pilot-report-v2"
    assert report["endpoint"] == "functional"
    assert report["conditions"] == ["T", "H-gold", "E"]
    assert report["G1_journal_completeness"] == {
        "planned": 30,
        "recorded": 30,
        "unrecorded": 0,
        "duplicates": 0,
        "unknown": 0,
        "go": True,
    }
    assert report["G2_resume"]["fingerprint_mismatches"] == 0
    assert report["G2_resume"]["input_mismatches"] == 0
    assert report["G3_rendering"] == {"t_requests": 10, "t_unequal_to_source": 0, "go": True}
    assert report["G4_token_budget"] == {
        "max_seq_length": 4096,
        "max_new_tokens": 1280,
        "max_input_tokens": 300,
        "max_input_plus_budget": 1580,
        "go": True,
    }


def test_truncation_is_per_condition_with_the_budget_rule(pilot) -> None:
    g5 = report_for(pilot)["G5_truncation"]
    assert g5["by_condition"] == {
        "T": {"requests": 10, "present": False, "go": True},
        "H-gold": {"requests": 10, "present": True, "go": False},
        "E": {"requests": 10, "present": True, "go": False},
    }
    assert g5["length"] == 2 and g5["rate"] == pytest.approx(2 / 30)
    assert g5["threshold"] == 0.02
    assert g5["output_tokens_p99"] == 195 and g5["output_tokens_max"] == 195
    assert g5["max_new_tokens_if_raised"] == 244  # ceil(1.25 * 195)


def test_extraction_and_prompt_as_data(pilot) -> None:
    report = report_for(pilot)
    assert report["G6_extraction"] == {
        "primary_policy": "bigcodebench-sanitize",
        "rows": 30,
        "failures": 1,
        "rate": pytest.approx(1 / 30),
        "threshold": 0.15,
        "go": True,
        "by_condition": {
            "T": {"rows": 10, "present": True},
            "H-gold": {"rows": 10, "present": False},
            "E": {"rows": 10, "present": False},
        },
        "diagnostic": {"python-single-block-v1": {"failures": 3, "rate": 0.1}},
    }
    assert report["G6b_prompt_as_data"] == {
        "by_condition": {
            "T": {"rows": 10, "json_response_present": False, "prompt_as_data_present": False},
            "H-gold": {"rows": 10, "json_response_present": True, "prompt_as_data_present": True},
            "E": {"rows": 10, "json_response_present": False, "prompt_as_data_present": True},
        },
        "total": {"rows": 30, "json_response": 1, "prompt_as_data": 2},
    }


def test_flakiness_provenance_resources_and_overhead(pilot) -> None:
    report = report_for(pilot)
    assert report["G8_flakiness"] == {
        "executed": 27,
        "discordant_runs": 1,
        "agreement": pytest.approx(26 / 27),
        "flaky": 1,
        "threshold": 0.99,
        "go": False,
    }
    assert report["G10_provenance"] == {
        "purpose": "pilot",
        "revision_verified": True,
        "chat_template_matches_official": True,
        "repetition_penalty": 1.0,
        "source_digest": pilot[0]["source"]["digest"],
        "go": True,
    }
    assert report["G11_resources"] == {
        "elapsed_seconds_total": 39.0,
        "elapsed_seconds_median": 1.0,
        "elapsed_seconds_p95": 1.0,
        "peak_vram_bytes": 1009,
        "devices": ["NVIDIA A100-SXM4-40GB"],
    }
    assert report["G13_input_overhead"] == {
        "H-gold": {"pairs": 10, "median": 2.5, "p10": 2.5, "p90": 2.5, "min": 2.5, "max": 3.0},
        "E": {"pairs": 10, "median": 1.25, "p10": 1.25, "p90": 1.25, "min": 1.25, "max": 1.25},
    }
    assert set(report["not_computed_here"]) == {"G7", "G12", "G14"}


def test_journal_faults_are_counted_not_hidden(pilot) -> None:
    manifest, rows = pilot
    altered = [*rows[1:], rows[2], {**rows[3], "request_id": "f" * 64}]
    altered[4] = {**altered[4], "fingerprint": "0" * 64}
    altered[5] = {**altered[5], "seed": 99}
    report = pilot_report(manifest, altered)
    assert report["G1_journal_completeness"] == {
        "planned": 30,
        "recorded": 29,
        "unrecorded": 1,
        "duplicates": 1,
        "unknown": 1,
        "go": False,
    }
    assert report["G2_resume"]["fingerprint_mismatches"] == 1
    assert report["G2_resume"]["input_mismatches"] == 1
    assert report["endpoint"] is None and report["discordant_pairs"] is None
    assert report["G6_extraction"] is None and report["G8_flakiness"] is None


def test_evaluation_files_must_cover_the_recorded_journal(pilot) -> None:
    manifest, rows = pilot
    outcomes, details = functional(rows, "A")
    with pytest.raises(ValueError, match="outcomes do not cover"):
        pilot_report(manifest, rows, outcomes[1:], details)
    with pytest.raises(ValueError, match="details do not cover"):
        pilot_report(manifest, rows, outcomes, details[1:])
    with pytest.raises(ValueError, match="together"):
        pilot_report(manifest, rows, outcomes, None)
    mixed = [*outcomes[:-1], replace(outcomes[-1], endpoint="security")]
    with pytest.raises(ValueError, match="one endpoint"):
        pilot_report(manifest, rows, mixed, details)


def test_t_differing_from_its_source_is_counted(pilot) -> None:
    manifest, rows = pilot
    t_row = next(r for r in rows if r["condition"] == "T")
    altered = [{**r, "source_hash": "0" * 64} if r is t_row else r for r in rows]
    report = pilot_report(manifest, altered)
    assert report["G3_rendering"] == {"t_requests": 10, "t_unequal_to_source": 1, "go": False}


def security_fixture(tmp_path: Path) -> tuple[dict, list[dict], list[OutcomeRecord], list[dict]]:
    directory = tmp_path / "security"
    prompts = []
    for task in ("LSE-000", "LSE-001"):
        text = f"Write {task}."
        prompts += render_prompt_variants(
            CanonicalPrompt(
                task, text, (SpanAnnotation("main_instruction", 0, len(text), text),), "v"
            )
        )
    plan = request_plan(prompts, [1, 2, 3, 4, 5], 1)
    configuration = {
        **CONFIGURATION,
        "decoding": {"max_new_tokens": 1280, "repetition_penalty": 1.0},
    }
    run_journal(plan, configuration, directory, fake_generation_security)
    manifest, rows = read_directory(directory)
    outcomes, details = [], []
    for number, row in enumerate(rows):
        usable = number % 5 != 0
        state = "missing" if number == 1 else ("V" if number % 2 else "N") if usable else "U"
        outcomes.append(
            OutcomeRecord(
                "pilot-sec",
                "security",
                row["task_id"],
                row["condition"],
                row["replicate_id"],
                state,
                "codeql_unavailable" if state == "missing" else "",
            )
        )
        details.append(
            {
                "request_id": row["request_id"],
                "condition": row["condition"],
                "measure": "general",
                "state": state,
                "usable": usable,
                "checks_complete": state != "missing",
                "extraction_status": "ok" if usable else "no_code",
                "extraction_statuses": {"python-single-block-v1": "ok" if usable else "no_code"},
                "finish_reason": row["finish_reason"],
                "analysed": state != "missing",
            }
        )
    return manifest, rows, outcomes, details


def fake_generation_security(request: dict) -> dict[str, Any]:
    return {"raw_output": "x", "input_tokens": 10, "output_tokens": 5, "finish_reason": "eos"}


def test_security_report_measures_codeql_operation_only(tmp_path) -> None:
    manifest, rows, outcomes, details = security_fixture(tmp_path)
    report = pilot_report(manifest, rows, outcomes, details)
    assert report["endpoint"] == "security"
    assert report["discordant_pairs"] is None
    assert report["G8_flakiness"] is None and report["G6b_prompt_as_data"] is None
    # 20 programs, 4 unusable (every fifth), 16 usable of which 1 not analysed.
    assert report["G9_codeql"] == {
        "measure": "general",
        "usable": 16,
        "analysed": 15,
        "rate": 15 / 16,
        "threshold": 0.95,
        "go": False,
    }
    assert report["G6_extraction"]["primary_policy"] == "python-fenced-blocks-v2"
    assert report["G6_extraction"]["failures"] == 4
    assert report["G11_resources"]["peak_vram_bytes"] is None
    for key in keys(report):
        assert not any(fragment in key.lower() for fragment in FORBIDDEN_KEY_FRAGMENTS), key


def test_pilot_report_command(tmp_path) -> None:
    directory = journal_directory(tmp_path)
    _, rows = read_directory(directory)
    outcomes, details = functional(rows, "A")
    write_outcomes(outcomes, tmp_path / "outcomes.jsonl")
    (tmp_path / "details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in details), encoding="utf-8"
    )
    out = tmp_path / "report" / "pilot-report.json"
    arguments = [
        "pilot-report",
        str(directory),
        "--outcomes",
        str(tmp_path / "outcomes.jsonl"),
        "--details",
        str(tmp_path / "details.jsonl"),
        "--out",
        str(out),
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.exception
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["discordant_pairs"]["discordant_total"] == 2
    manifest = json.loads((out.parent / "empirical-pilot-report_manifest.json").read_text("utf-8"))
    assert manifest["stage"] == "empirical-pilot-report"
    assert set(manifest) >= {"journal_sha256", "outcomes_sha256", "details_sha256", "report_sha256"}
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0  # an existing report is never overwritten
