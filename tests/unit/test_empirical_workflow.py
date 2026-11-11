import json

from typer.testing import CliRunner

from pstparser.empirical.cli import app
from pstparser.empirical.sandbox import ExecutionResult
from pstparser.empirical.workflow import FunctionalOracle, calibrate_oracles, evaluate_functional


def test_calibration_and_outcomes_preserve_invalid_and_missing() -> None:
    oracle = FunctionalOracle("t", "v1", "harness", "correct", "incorrect")

    def execute(code, harness):
        if code == "correct":
            return ExecutionResult("passed", 0, "ok")
        if code == "incorrect":
            return ExecutionResult("failed", 1, "assertion")
        return ExecutionResult("infrastructure_error", None, "daemon unavailable")

    calibrated = calibrate_oracles([oracle], execute)
    generations = [
        {
            "task_id": "t",
            "condition": "T",
            "replicate_id": 0,
            "raw_output": "bad syntax ???",
            "finish_reason": "eos",
        },
        {
            "task_id": "t",
            "condition": "H-gold",
            "replicate_id": 0,
            "raw_output": "x=1",
            "finish_reason": "eos",
        },
    ]
    outcomes, details = evaluate_functional(generations, [oracle], "run", execute, calibrated)
    assert [row.outcome for row in outcomes] == ["fail", "missing"]
    assert len(details) == 2


def test_always_passing_harness_fails_calibration() -> None:
    oracle = FunctionalOracle("t", "v1", "pass", "x=1", "x=2")
    result = calibrate_oracles([oracle], lambda *_: ExecutionResult("passed", 0, ""))
    assert result["t"]["calibrated"] is False


def test_render_command_uses_source_and_refuses_overwrite(tmp_path) -> None:
    source = tmp_path / "canonical.jsonl"
    destination = tmp_path / "variants.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "t",
                "text": "hello",
                "annotation_version": "v1",
                "annotations": [
                    {"path": "main_instruction", "start": 0, "end": 5, "phrase": "hello"}
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["render", str(source), str(destination)])
    assert result.exit_code == 0, result.output
    assert len(destination.read_text().splitlines()) == 2
    assert runner.invoke(app, ["render", str(source), str(destination)]).exit_code != 0
