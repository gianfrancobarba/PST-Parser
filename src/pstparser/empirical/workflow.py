"""Connect frozen generation records to calibrated functional checks and analysis.

Benchmark adapters and CWE rubrics must be approved separately. A supplied
Python harness is not automatically HumanEval+ and is never labelled as such.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pstparser.empirical.analysis import Outcome, OutcomeRecord
from pstparser.empirical.extraction import extract_python
from pstparser.empirical.sandbox import ExecutionResult


@dataclass(frozen=True)
class FunctionalOracle:
    """Trusted test harness and known positive/negative controls for one task."""

    task_id: str
    version: str
    harness: str
    positive_control: str
    negative_control: str


def calibrate_oracles(
    oracles: Sequence[FunctionalOracle],
    execute: Callable[[str, str], ExecutionResult],
) -> dict[str, dict[str, Any]]:
    """Check both controls; do not accept a harness that passes every program."""
    results: dict[str, dict[str, Any]] = {}
    for oracle in oracles:
        if oracle.task_id in results or not oracle.task_id or not oracle.version:
            raise ValueError("oracles require unique task IDs and a version")
        if not all((oracle.harness, oracle.positive_control, oracle.negative_control)):
            raise ValueError("harness and both controls are required")
        positive = execute(oracle.positive_control, oracle.harness)
        negative = execute(oracle.negative_control, oracle.harness)
        results[oracle.task_id] = {
            "oracle_version": oracle.version,
            "positive_status": positive.status,
            "negative_status": negative.status,
            "calibrated": positive.status == "passed" and negative.status == "failed",
        }
    return results


def evaluate_functional(
    generations: Sequence[Mapping[str, Any]],
    oracles: Sequence[FunctionalOracle],
    analysis_id: str,
    execute: Callable[[str, str], ExecutionResult],
    calibration: Mapping[str, Mapping[str, Any]],
) -> tuple[list[OutcomeRecord], list[dict[str, Any]]]:
    """Apply one extractor/harness policy symmetrically, retaining every outcome.

    Infrastructure errors become explicit missing records. Timeout, malformed
    code and an output cut at its token budget are observed failures. Automatic
    retries are not performed here: a caller must apply the frozen retry policy
    and preserve its attempt log before calling this final-outcome adapter.
    """
    by_task = {o.task_id: o for o in oracles}
    if len(by_task) != len(oracles):
        raise ValueError("duplicate oracle task")
    keys: set[tuple[str, str, int]] = set()
    outcomes: list[OutcomeRecord] = []
    details: list[dict[str, Any]] = []
    for generation in generations:
        task = generation["task_id"]
        condition = generation["condition"]
        replicate = generation["replicate_id"]
        key = (task, condition, replicate)
        if key in keys:
            raise ValueError("duplicate generation")
        keys.add(key)
        oracle = by_task[task]
        status = calibration.get(task, {})
        if not status.get("calibrated") or status.get("oracle_version") != oracle.version:
            raise ValueError(f"oracle not calibrated: {task}")
        extracted = extract_python(generation["raw_output"])
        result = None
        if generation["finish_reason"] == "length":
            reason = "output_token_limit"
        elif extracted.status != "ok":
            reason = extracted.status
        else:
            assert extracted.code is not None
            result = execute(extracted.code, oracle.harness)
            reason = result.status
        outcome: Outcome = (
            "missing"
            if reason == "infrastructure_error"
            else "pass"
            if reason == "passed"
            else "fail"
        )
        outcomes.append(
            OutcomeRecord(
                analysis_id=analysis_id,
                endpoint="functional",
                task_id=task,
                condition=condition,
                replicate_id=replicate,
                outcome=outcome,
                reason=reason,
            )
        )
        details.append(
            {
                "task_id": task,
                "condition": condition,
                "replicate_id": replicate,
                "code": extracted.code,
                "extraction_status": extracted.status,
                "oracle_version": oracle.version,
                "reason": reason,
                "execution_detail": result.detail if result is not None else None,
                "exit_code": result.exit_code if result is not None else None,
            }
        )
    return outcomes, details
