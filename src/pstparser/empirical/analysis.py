"""Strict paired analysis of a frozen, single-model downstream experiment.

Missing infrastructure outcomes must be explicit records. An absent row is a
broken ledger, not an invitation to silently shrink the denominator. Security
states are assigned by an external oracle; this module does not infer safety.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Endpoint = Literal["functional", "security"]
Outcome = Literal["pass", "fail", "V", "N", "I", "U", "missing"]


@dataclass(frozen=True)
class TaskSpec:
    """A planned task and its optional, predeclared source family."""

    task_id: str
    family_id: str | None = None


@dataclass(frozen=True)
class AnalysisDesign:
    """Frozen analysis scope; analysis_id identifies dataset/model/configuration.

    Conditions are ordered (baseline, treatment). All tasks must have family IDs
    or none may have them. IDs identify provenance; callers must archive the
    manifest they refer to. This module cannot verify a model's actual weights.
    """

    analysis_id: str
    endpoint: Endpoint
    tasks: tuple[TaskSpec, ...]
    replicate_ids: tuple[int, ...] = (0,)
    conditions: tuple[str, str] = ("T", "H-gold")


@dataclass(frozen=True)
class OutcomeRecord:
    """One final oracle outcome, after the preregistered retry policy.

    Findings are retained independently of security state, including I/U.
    Missing records require a reason; they never mean an observed failure.
    """

    analysis_id: str
    endpoint: Endpoint
    task_id: str
    condition: str
    replicate_id: int
    outcome: Outcome
    reason: str = ""
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskEstimate:
    """Equal-replicate arm means and treatment-minus-baseline difference."""

    task_id: str
    family_id: str | None
    baseline: float
    treatment: float
    difference: float


@dataclass(frozen=True)
class PairedReport:
    """JSON-compatible estimates, denominators, diagnostics and sensitivity.

    Effects and intervals use the proportion scale, not percentage points.
    The conservative interval is a Bonferroni marginal exact-binomial bound,
    restricted to independent tasks with one binary observation per arm.
    """

    analysis_id: str
    endpoint: Endpoint
    conditions: tuple[str, str]
    planned_tasks: int
    included_tasks: int
    planned_records_per_arm: int
    included_records_per_arm: int
    excluded_tasks: dict[str, list[str]]
    outcome_counts: dict[str, dict[str, int]]
    task_estimates: tuple[TaskEstimate, ...]
    delta: float | None
    bootstrap_ci: tuple[float, float] | None
    bootstrap_unit: str
    clusters: int
    bootstrap_samples: int
    bootstrap_seed: int
    confidence: float
    mcnemar_table: dict[str, int] | None
    mcnemar_p: float | None
    conservative_ci: tuple[float, float] | None
    sensitivity: dict[str, float | None]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable report, including explicit null estimates."""
        return asdict(self)


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _join(
    design: AnalysisDesign, records: Sequence[OutcomeRecord]
) -> dict[tuple[str, str, int], OutcomeRecord]:
    _identifier(design.analysis_id, "analysis_id")
    if design.endpoint not in ("functional", "security"):
        raise ValueError("Unknown endpoint")
    if len(design.conditions) != 2 or len(set(design.conditions)) != 2:
        raise ValueError("Exactly two distinct conditions are required")
    for condition in design.conditions:
        _identifier(condition, "condition")
    if not design.tasks or len({t.task_id for t in design.tasks}) != len(design.tasks):
        raise ValueError("Tasks must be non-empty and unique")
    for task in design.tasks:
        _identifier(task.task_id, "task_id")
        if task.family_id is not None:
            _identifier(task.family_id, "family_id")
    if len({t.family_id is None for t in design.tasks}) != 1:
        raise ValueError("Family IDs must be supplied for all tasks or none")
    if (
        not design.replicate_ids
        or len(set(design.replicate_ids)) != len(design.replicate_ids)
        or any(type(r) is not int or r < 0 for r in design.replicate_ids)
    ):
        raise ValueError("Replicate IDs must be distinct nonnegative integers")
    expected = {
        (task.task_id, condition, replicate)
        for task in design.tasks
        for condition in design.conditions
        for replicate in design.replicate_ids
    }
    allowed = (
        {"pass", "fail", "missing"}
        if design.endpoint == "functional"
        else {"V", "N", "I", "U", "missing"}
    )
    joined: dict[tuple[str, str, int], OutcomeRecord] = {}
    for record in records:
        if record.analysis_id != design.analysis_id or record.endpoint != design.endpoint:
            raise ValueError("Mixed analysis scope or endpoint")
        if type(record.replicate_id) is not int:
            raise ValueError("Record replicate_id must be an integer")
        key = (record.task_id, record.condition, record.replicate_id)
        if key not in expected:
            raise ValueError(f"Unexpected record key: {key}")
        if key in joined:
            raise ValueError(f"Duplicate record key: {key}")
        if record.outcome not in allowed:
            raise ValueError(f"Invalid {design.endpoint} outcome: {record.outcome}")
        if not isinstance(record.reason, str):
            raise ValueError("reason must be a string")
        if record.outcome == "missing" and not record.reason.strip():
            raise ValueError("Missing infrastructure outcomes require a reason")
        if not isinstance(record.findings, (tuple, list)) or any(
            not isinstance(finding, str) for finding in record.findings
        ):
            raise ValueError("findings must be a sequence of strings")
        joined[key] = record
    absent = expected - joined.keys()
    if absent:
        raise ValueError(f"Absent planned records (use explicit missing): {sorted(absent)}")
    return joined


def _value(record: OutcomeRecord, *, unknown: int = 0, missing: int = 0) -> int:
    if record.outcome == "missing":
        return missing
    if record.outcome == "U":
        return unknown
    return int(record.outcome in ("pass", "N"))


def exact_mcnemar(baseline_only: int, treatment_only: int) -> float:
    """Compute the two-sided exact binomial McNemar p-value.

    Counts must describe independent discordant task pairs. No continuity
    correction or asymptotic chi-square approximation is applied.
    """
    if any(type(n) is not int or n < 0 for n in (baseline_only, treatment_only)):
        raise ValueError("Discordant counts must be nonnegative integers")
    n = baseline_only + treatment_only
    tail = sum(math.comb(n, k) for k in range(min(baseline_only, treatment_only) + 1))
    return float(min(1.0, 2 * tail / (2**n)))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Adjust one predeclared family of p-values using Holm's step-down rule.

    Supply only valid inferential tests from the same declared family. This
    helper does not turn exploratory comparisons into confirmatory tests.
    """
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("p-values must be finite and in [0, 1]")
    ordered = sorted(p_values, key=lambda name: p_values[name])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_values[name]))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def _binomial_tail(n: int, p: float, start: int, stop: int) -> float:
    if p == 0:
        return float(start == 0)
    if p == 1:
        return float(stop == n)
    return math.fsum(
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * math.log(p)
            + (n - k) * math.log1p(-p)
        )
        for k in range(start, stop + 1)
    )


def _exact_binomial_interval(successes: int, n: int, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    bounds = []
    for lower in (True, False):
        if lower and successes == 0:
            bounds.append(0.0)
            continue
        if not lower and successes == n:
            bounds.append(1.0)
            continue
        left, right = 0.0, 1.0
        for _ in range(60):
            mid = (left + right) / 2
            probability = (
                _binomial_tail(n, mid, successes, n)
                if lower
                else _binomial_tail(n, mid, 0, successes)
            )
            if (lower and probability < alpha) or (not lower and probability > alpha):
                left = mid
            else:
                right = mid
        bounds.append((left + right) / 2)
    return bounds[0], bounds[1]


def _bootstrap(
    estimates: Sequence[TaskEstimate], *, samples: int, seed: int, confidence: float
) -> tuple[tuple[float, float], int, bool]:
    groups: dict[str, list[float]] = {}
    for estimate in estimates:
        key = estimate.family_id if estimate.family_id is not None else estimate.task_id
        groups.setdefault(key, []).append(estimate.difference)
    sums = np.array([sum(values) for values in groups.values()], dtype=float)
    sizes = np.array([len(values) for values in groups.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        draws[index] = sums[chosen].sum() / sizes[chosen].sum()
    alpha = (1 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha], method="linear")
    return (float(low), float(high)), len(groups), bool(np.all(draws == draws[0]))


def analyze_paired(
    design: AnalysisDesign,
    records: Sequence[OutcomeRecord],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> PairedReport:
    """Join the entire planned ledger and estimate the paired effect.

    Primary analysis excludes a whole task if any planned arm/replicate is
    explicitly missing. U is zero for the primary security endpoint. Sensitivity
    includes U=1 and U bounds on complete tasks, missing bounds on all planned
    tasks (holding U=0), and joint U/missing bounds. No p-value is emitted for
    clustered or repeated-sample designs. Bootstrap preserves complete blocks.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be an integer >= 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    joined = _join(design, records)
    baseline, treatment = design.conditions
    excluded: dict[str, list[str]] = {}
    estimates: list[TaskEstimate] = []
    for task in design.tasks:
        missing = [
            f"{condition}/{replicate}: {joined[task.task_id, condition, replicate].reason}"
            for condition in design.conditions
            for replicate in design.replicate_ids
            if joined[task.task_id, condition, replicate].outcome == "missing"
        ]
        if missing:
            excluded[task.task_id] = missing
            continue
        means = [
            sum(_value(joined[task.task_id, condition, r]) for r in design.replicate_ids)
            / len(design.replicate_ids)
            for condition in design.conditions
        ]
        estimates.append(
            TaskEstimate(task.task_id, task.family_id, means[0], means[1], means[1] - means[0])
        )

    def effect(
        *, complete: bool, u_t: int = 0, u_h: int = 0, m_t: int = 0, m_h: int = 0
    ) -> float | None:
        differences = []
        for task in design.tasks:
            if complete and task.task_id in excluded:
                continue
            differences.append(
                sum(
                    _value(joined[task.task_id, treatment, r], unknown=u_h, missing=m_h)
                    - _value(joined[task.task_id, baseline, r], unknown=u_t, missing=m_t)
                    for r in design.replicate_ids
                )
                / len(design.replicate_ids)
            )
        return math.fsum(differences) / len(differences) if differences else None

    sensitivity = {
        "complete_U_one_both": effect(complete=True, u_t=1, u_h=1),
        "complete_U_lower": effect(complete=True, u_t=1),
        "complete_U_upper": effect(complete=True, u_h=1),
        "planned_missing_lower_U_zero": effect(complete=False, m_t=1),
        "planned_missing_upper_U_zero": effect(complete=False, m_h=1),
        "planned_joint_lower": effect(complete=False, u_t=1, m_t=1),
        "planned_joint_upper": effect(complete=False, u_h=1, m_h=1),
    }
    counts: dict[str, dict[str, int]] = {
        condition: dict(
            sorted(
                Counter(
                    str(record.outcome) for record in records if record.condition == condition
                ).items()
            )
        )
        for condition in design.conditions
    }
    warnings: list[str] = []
    ci = None
    conservative = None
    table = None
    p_value = None
    clusters = 0
    clustered = design.tasks[0].family_id is not None
    if excluded:
        warnings.append("Complete-task exclusion may bias estimates if missingness is not random.")
    if estimates:
        ci, clusters, degenerate = _bootstrap(
            estimates, samples=bootstrap_samples, seed=seed, confidence=confidence
        )
        if clusters < 20:
            warnings.append("Fewer than 20 resampling units; interval stability may be poor.")
        if degenerate:
            warnings.append("Degenerate empirical bootstrap; a zero-width CI is not certainty.")
        if not clustered and len(design.replicate_ids) == 1:
            table = {"both_fail": 0, "baseline_only": 0, "treatment_only": 0, "both_pass": 0}
            labels = {
                (0, 0): "both_fail",
                (1, 0): "baseline_only",
                (0, 1): "treatment_only",
                (1, 1): "both_pass",
            }
            for estimate in estimates:
                table[labels[int(estimate.baseline), int(estimate.treatment)]] += 1
            p_value = exact_mcnemar(table["baseline_only"], table["treatment_only"])
            # Report the conservative safeguard always, avoiding a post-hoc
            # decision about how few discordances count as "few".
            marginal_confidence = 1 - (1 - confidence) / 2
            t_low, t_high = _exact_binomial_interval(
                table["baseline_only"] + table["both_pass"], len(estimates), marginal_confidence
            )
            h_low, h_high = _exact_binomial_interval(
                table["treatment_only"] + table["both_pass"], len(estimates), marginal_confidence
            )
            conservative = (h_low - t_high, h_high - t_low)
    else:
        warnings.append("No complete tasks; primary estimate and inference are unavailable.")
    if clustered or len(design.replicate_ids) != 1:
        warnings.append("McNemar omitted: repeated observations or predeclared family clustering.")
    return PairedReport(
        analysis_id=design.analysis_id,
        endpoint=design.endpoint,
        conditions=design.conditions,
        planned_tasks=len(design.tasks),
        included_tasks=len(estimates),
        planned_records_per_arm=len(design.tasks) * len(design.replicate_ids),
        included_records_per_arm=len(estimates) * len(design.replicate_ids),
        excluded_tasks=excluded,
        outcome_counts=counts,
        task_estimates=tuple(estimates),
        delta=effect(complete=True),
        bootstrap_ci=ci,
        bootstrap_unit="family" if clustered else "task",
        clusters=clusters,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
        confidence=confidence,
        mcnemar_table=table,
        mcnemar_p=p_value,
        conservative_ci=conservative,
        sensitivity=sensitivity,
        warnings=tuple(warnings),
    )


def write_outcomes(records: Sequence[OutcomeRecord], destination: str | Path) -> Path:
    """Write the final outcome ledger as UTF-8 JSONL; analyze to validate scope."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")
    return path


def read_outcomes(source: str | Path) -> list[OutcomeRecord]:
    """Read a JSONL ledger without silently ignoring malformed or blank rows."""
    records = []
    with Path(source).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("Expected an object")
                if "findings" in raw:
                    if not isinstance(raw["findings"], list):
                        raise ValueError("findings must be an array")
                    raw["findings"] = tuple(raw["findings"])
                records.append(OutcomeRecord(**raw))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid outcome record at line {line_number}: {error}"
                ) from error
    return records
