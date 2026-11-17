"""C-15 (R-ORA-01..03): harness faults are faults, and cost only their own programs.

Counterexamples of the independent review, inverted: rows absent with exit 0
were never retried, one exhausted run made every program missing, a missing
row in one run prevailed over a valid outcome of the other, and a discordance
above 1% only printed a warning.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pstparser.empirical import bigcodebench as bcb


def sr(request: str, status: str, reason: str | None = None) -> bcb.SampleResult:
    return bcb.SampleResult(request, status, {"reason": reason} if reason else {})


def dataset(tmp: Path, tasks: list[str]) -> bcb.HarnessConfig:
    path = tmp / "data" / "d.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            row = {name: "" for name in bcb.DATASET_FIELDS}
            row.update(task_id=task, code_prompt="def task_func():\n", entry_point="task_func")
            handle.write(json.dumps(row) + "\n")
    nltk = tmp / "nltk"
    nltk.mkdir()
    return bcb.HarnessConfig(
        image="x@sha256:" + "0" * 64,
        dataset_jsonl=path,
        dataset_sha256="0",
        nltk_dir=nltk,
        nltk_sha256="0",
        parallel=1,
    )


def generations(tasks: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": f"{t.split('/')[1]}-{c}",
            "task_id": t,
            "condition": c,
            "replicate_id": 0,
            "raw_output": "```python\ndef task_func():\n    return 1\n```",
            "finish_reason": "stop",
        }
        for t in tasks
        for c in ("T", "H")
    ]


def sanitizer(config, generations, entry_points, work_dir):
    return {g["request_id"]: "def task_func():\n    return 1" for g in generations}


Behaviour = Callable[[str, list[str]], tuple[int, list[str], str, list[str]]]


def runner(behaviour: Behaviour) -> tuple[Callable[..., bcb.HarnessRun], list[str]]:
    """A fake harness: behaviour(name, tasks) -> (exit, absent, status, failed references)."""
    calls: list[str] = []

    def execute(config, target, task_ids=None, **_):
        calls.append(target.name)
        lines = (target / bcb.SAMPLES_FILE).read_text(encoding="utf-8").split("\n")
        rows = [json.loads(line) for line in lines if line]
        exit_code, drop, status, reference = behaviour(target.name, sorted(task_ids or []))
        evals: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            entry = {
                "task_id": row["task_id"],
                "status": status,
                "details": {},
                "solution": "def task_func():\n" + "\n    pass\n" + row["solution"],
            }
            evals.setdefault(row["task_id"], []).append(entry)
        for task in drop:
            evals.pop(task, None)
        (target / bcb.RESULTS_FILE).write_text(json.dumps({"eval": evals}), encoding="utf-8")
        (target / bcb.LOG_FILE).write_text("log", encoding="utf-8")
        (target / bcb.PASS_AT_K_FILE).write_text(
            json.dumps({"gt_pass_rate": 1.0 if not reference else 0.5, "failed_tasks": reference}),
            encoding="utf-8",
        )
        return bcb.HarnessRun(target, ("docker",), exit_code, 1.0)

    return execute, calls


def evaluate(tmp: Path, tasks: list[str], behaviour: Behaviour, **options: Any) -> dict[str, Any]:
    execute, calls = runner(behaviour)
    result = bcb.evaluate_generations(
        dataset(tmp, tasks),
        generations(tasks),
        tmp / "out",
        analysis_id="a",
        order_seed=1,
        runner=execute,
        sanitizer=sanitizer,
        **options,
    )
    result["calls"] = calls
    return result


def outcomes(result: dict[str, Any]) -> dict[tuple[str, str], tuple[str, str]]:
    return {(r.task_id, r.condition): (r.outcome, r.reason) for r in result["outcomes"]}


TASKS = ["BigCodeBench/1", "BigCodeBench/2"]


# --- R-ORA-03: a missing row in one run does not prevail over a valid outcome.


def test_pass_against_missing_needs_a_third_run_and_the_majority_of_valid_outcomes() -> None:
    first, second = [sr("r", "pass")], [sr("r", "missing", "row_absent")]
    with pytest.raises(ValueError, match="third run is required"):
        bcb.resolve_majority(first, second)
    (settled,) = bcb.resolve_majority(first, second, [sr("r", "pass")])
    assert (settled.status, settled.flaky) == ("pass", True)
    (settled,) = bcb.resolve_majority(first, second, [sr("r", "fail")])
    assert (settled.status, dict(settled.details)) == ("missing", {"reason": "no_majority"})


def test_missing_twice_stays_missing_without_a_third_run() -> None:
    (settled,) = bcb.resolve_majority([sr("r", "missing", "row_absent")], [sr("r", "missing")])
    assert settled.status == "missing"


# --- R-ORA-01: absent rows with exit 0 and failed references are faults, retried.


def test_absent_rows_with_exit_zero_are_retried(tmp_path: Path) -> None:
    def behaviour(name: str, tasks: list[str]):
        return 0, ["BigCodeBench/2"] if name == "run-a" else [], "pass", []

    result = evaluate(tmp_path, TASKS, behaviour)
    assert result["calls"] == ["run-a", "run-a-attempt-2", "run-b"]
    assert {value for value in outcomes(result).values()} == {("pass", "pass")}


def test_a_failed_reference_is_a_fault_of_the_run(tmp_path: Path) -> None:
    def behaviour(name: str, tasks: list[str]):
        return 0, [], "pass", ["BigCodeBench/1"] if name == "run-a" else []

    result = evaluate(tmp_path, TASKS, behaviour)
    assert result["calls"] == ["run-a", "run-a-attempt-2", "run-b"]
    assert result["attempts"][0]["detail"].startswith("reference failed")


# --- R-ORA-02: an exhausted run is repeated task by task.


def test_an_exhausted_run_is_repeated_by_task_and_only_the_faulty_batch_is_missing(
    tmp_path: Path,
) -> None:
    def behaviour(name: str, tasks: list[str]):
        # A program of task 2 crashes the container in every run, whole or by task.
        crashed = len(tasks) > 1 or tasks == ["BigCodeBench/2"]
        return (1 if crashed else 0), [], "pass", []

    result = evaluate(tmp_path, TASKS, behaviour)
    settled = outcomes(result)
    assert settled[("BigCodeBench/1", "T")] == ("pass", "pass")
    assert settled[("BigCodeBench/2", "T")] == ("missing", "harness_unavailable")
    assert "run-a-task-1" in result["calls"] and "run-a-task-2" in result["calls"]


# --- R-ORA-01: more than 1% of discordant programs stops the evaluation.


def test_a_discordance_above_one_percent_stops_unless_accepted(tmp_path: Path) -> None:
    tasks = [f"BigCodeBench/{n}" for n in range(1, 6)]

    def behaviour(name: str, tasks: list[str]):
        return 0, [], "fail" if name == "run-b" else "pass", []

    with pytest.raises(ValueError, match="1%"):
        evaluate(tmp_path / "a", tasks, behaviour)
    accepted = evaluate(tmp_path / "b", tasks, behaviour, accept_discordance=True)
    assert accepted["summary"]["discordance_accepted"] is True
