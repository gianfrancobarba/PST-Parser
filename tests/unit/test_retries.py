"""Bounded, logged retries of infrastructure failures."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical import sandbox
from pstparser.empirical.cli import app
from pstparser.empirical.retries import MAX_ATTEMPTS, exhausted, with_retries
from pstparser.empirical.sandbox import ExecutionResult

IMAGE = "python@sha256:" + "a" * 64


def scripted(*results: str):
    """An execution that returns the given statuses in turn and records each attempt."""
    calls: list[int] = []
    sequence = list(results)

    def execute(attempt: int) -> str:
        calls.append(attempt)
        return sequence[min(attempt, len(sequence)) - 1]

    return execute, calls


def infrastructure(result: str) -> str | None:
    return "daemon unavailable" if result == "infra" else None


def test_two_failures_then_success_give_three_logged_attempts() -> None:
    execute, calls = scripted("infra", "infra", "passed")
    result, attempts = with_retries(execute, infrastructure)
    assert result == "passed"
    assert calls == [1, 2, 3]
    assert [(a["attempt"], a["status"], a["detail"]) for a in attempts] == [
        (1, "infrastructure_error", "daemon unavailable"),
        (2, "infrastructure_error", "daemon unavailable"),
        (3, "ok", ""),
    ]
    assert not exhausted(attempts)


def test_three_failures_exhaust_the_attempts() -> None:
    execute, calls = scripted("infra", "infra", "infra", "passed")
    result, attempts = with_retries(execute, infrastructure)
    assert MAX_ATTEMPTS == 3
    assert (result, calls) == ("infra", [1, 2, 3])
    assert exhausted(attempts)


@pytest.mark.parametrize("final", ["failed", "timeout", "passed"])
def test_an_outcome_of_the_program_is_never_retried(final: str) -> None:
    execute, calls = scripted(final, "passed")
    result, attempts = with_retries(execute, infrastructure)
    assert (result, calls, len(attempts)) == (final, [1], 1)


def test_exceptions_are_not_retried() -> None:
    def execute(attempt: int) -> str:
        raise RuntimeError("broken input")

    with pytest.raises(RuntimeError, match="broken input"):
        with_retries(execute, infrastructure)
    with pytest.raises(ValueError):
        with_retries(lambda _: "x", infrastructure, max_attempts=0)


# ----------------------------------------------------------------------- the check command


def oracle_files(tmp_path: Path) -> tuple[Path, Path]:
    oracles = tmp_path / "oracles.jsonl"
    oracles.write_text(
        json.dumps(
            {
                "task_id": "t",
                "version": "v1",
                "harness": "HARNESS",
                "positive_control": "GOOD",
                "negative_control": "BAD",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generations = tmp_path / "generations.jsonl"
    rows = [
        {"task_id": "t", "condition": c, "replicate_id": 0, "raw_output": f"```python\n{code}\n```"}
        | {"finish_reason": "eos"}
        for c, code in (("T", "FLAKY"), ("H-gold", "DOWN"))
    ]
    generations.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return generations, oracles


def test_the_check_command_retries_and_logs_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, int] = {}

    def run(code: str, harness: str, config: sandbox.SandboxConfig) -> ExecutionResult:
        seen[code] = seen.get(code, 0) + 1
        if code == "GOOD":
            return ExecutionResult("passed", 0, "")
        if code == "BAD":
            return ExecutionResult("failed", 1, "assertion")
        if code == "FLAKY" and seen[code] < 3:
            return ExecutionResult("infrastructure_error", None, "cannot start container")
        if code == "FLAKY":
            return ExecutionResult("passed", 0, "")
        return ExecutionResult("infrastructure_error", None, "cannot start container")

    monkeypatch.setattr(sandbox, "run_python_check", run)
    generations, oracles = oracle_files(tmp_path)
    directory = tmp_path / "check"
    result = CliRunner().invoke(
        app, ["check", str(generations), str(oracles), str(directory), IMAGE, "run-1"]
    )
    assert result.exit_code == 0, result.exception
    outcomes = [
        json.loads(line) for line in (directory / "outcomes.jsonl").read_text().split("\n") if line
    ]
    assert [(o["condition"], o["outcome"], o["reason"]) for o in outcomes] == [
        ("T", "pass", "passed"),
        ("H-gold", "missing", "infrastructure_error"),
    ]
    attempts = [
        json.loads(line) for line in (directory / "attempts.jsonl").read_text().split("\n") if line
    ]
    # Controls once each; FLAKY three times (two failures, then a pass); DOWN three failures.
    assert [a["attempt"] for a in attempts] == [1, 1, 1, 2, 3, 1, 2, 3]
    assert seen == {"GOOD": 1, "BAD": 1, "FLAKY": 3, "DOWN": 3}
    manifest = json.loads((directory / "empirical-check_manifest.json").read_text("utf-8"))
    assert manifest["retry_count"] == 4
    assert manifest["retry_policy"] == "infrastructure-retries-v1"
    assert manifest["max_attempts"] == 3
