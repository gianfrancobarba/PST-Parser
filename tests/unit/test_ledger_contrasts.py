"""Every contrast of an N-condition ledger uses the same complete tasks.

Counterexample of the independent review (``s07_missing``), inverted: with E
missing on t0 and H-gold missing on t1, H-T, E-T and H-E used three different
task sets, so Delta(H-T) - [Delta(E-T) + Delta(H-E)] was 0.4 instead of 0; and
the arms outside the analysed pair were not validated.
"""

import math

import pytest

from pstparser.empirical.analysis import AnalysisDesign, OutcomeRecord, TaskSpec, analyze_paired

OUTCOMES = {
    "T": ["fail", "pass", "fail", "fail", "pass", "fail"],
    "H-gold": ["pass", "missing", "pass", "fail", "pass", "fail"],
    "E": ["missing", "fail", "fail", "pass", "pass", "fail"],
}
LEDGER = ("T", "H-gold", "E")
TASKS = tuple(TaskSpec(f"t{i}") for i in range(6))


def records(outcomes=OUTCOMES) -> list[OutcomeRecord]:
    return [
        OutcomeRecord("bcb", "functional", f"t{i}", c, 0, o, "cuda_oom" if o == "missing" else "")
        for c, values in outcomes.items()
        for i, o in enumerate(values)
    ]


def analyse(pair: tuple[str, str], rows: list[OutcomeRecord] | None = None):
    design = AnalysisDesign("bcb", "functional", TASKS, (0,), pair)
    return analyze_paired(
        design,
        rows if rows is not None else records(),
        bootstrap_samples=50,
        ledger_conditions=LEDGER,
    )


def test_every_contrast_uses_the_same_complete_tasks() -> None:
    reports = {pair: analyse(pair) for pair in (("T", "H-gold"), ("T", "E"), ("E", "H-gold"))}
    for report in reports.values():
        assert report.included_tasks == 4
        assert sorted(report.excluded_tasks) == ["t0", "t1"]
    h_t, e_t, h_e = (reports[p].delta for p in (("T", "H-gold"), ("T", "E"), ("E", "H-gold")))
    assert math.isclose(h_t, e_t + h_e, abs_tol=1e-12)


def test_an_outcome_of_another_endpoint_in_a_declared_arm_is_refused() -> None:
    rows = [
        OutcomeRecord("bcb", "functional", "t2", "E", 0, "V", "")
        if (r.condition, r.task_id) == ("E", "t2")
        else r
        for r in records()
    ]
    with pytest.raises(ValueError, match="outcome"):
        analyse(("T", "H-gold"), rows)


def test_a_declared_arm_without_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="Absent"):
        analyse(("T", "H-gold"), [r for r in records() if r.condition != "E"])
