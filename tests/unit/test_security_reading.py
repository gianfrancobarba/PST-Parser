"""The security reading, end to end: report, CR1 interval at 90%, TOST and E1-E4.

The security rows of the paired table take their equivalence interval from the
CR1 interval of the report, moved to 90% with the same centre, standard error
and t(G - 1) quantile. These tests build reports from outcome records and check
the table against values derived by hand, with the t quantile taken from tables
(t(3) at 0.95 is 2.353363; at 0.975 it is 3.182446).
"""

from __future__ import annotations

import math

import pytest

from pstparser.empirical.analysis import AnalysisDesign, OutcomeRecord, TaskSpec, analyze_paired
from pstparser.empirical.tables import paired_tables

T_3_95 = 2.3533634348018233
T_3_975 = 3.1824463052837078


def security_report(tasks_per_family: int) -> dict:
    """Four families; in each, one task carries all the difference, five replicates.

    H-gold - T per family sums to 0.4, 0.0, 0.2, 0.2 (2, 0, 1, 1 extra N out of 5).
    """
    extra_n = [2, 0, 1, 1]
    tasks, records = [], []
    for family, k in enumerate(extra_n):
        for j in range(tasks_per_family):
            task = TaskSpec(f"LSE-{family}-{j}", f"f{family}")
            tasks.append(task)
            for replicate in range(1, 6):
                records.append(OutcomeRecord("s", "security", task.task_id, "T", replicate, "V"))
                treated = "N" if j == 0 and replicate <= k else "V"
                records.append(
                    OutcomeRecord("s", "security", task.task_id, "H-gold", replicate, treated)
                )
    design = AnalysisDesign("s", "security", tuple(tasks), (1, 2, 3, 4, 5))
    return analyze_paired(design, records, bootstrap_samples=50).as_dict()


def row_of(report: dict) -> dict:
    main, _ = paired_tables({"s": report})
    (row,) = main.rows
    return dict(zip(main.columns, row, strict=True))


def hand_interval(n_tasks: int, quantile: float) -> tuple[float, float]:
    # Estimate: 0.8 / N. Residuals S_g - estimate * n_g: 0.2, -0.2, 0, 0 (n_g = N / 4).
    estimate = 0.8 / n_tasks
    se = math.sqrt(4 / 3 * (0.2**2 + 0.2**2)) / n_tasks
    return estimate - quantile * se, estimate + quantile * se


@pytest.mark.parametrize(("per_family", "reading"), [(1, "E4"), (25, "E3")])
def test_security_row_uses_the_cr1_interval_at_90_percent(per_family: int, reading: str) -> None:
    n_tasks = 4 * per_family
    values = row_of(security_report(per_family))
    assert values["delta"] == pytest.approx(0.8 / n_tasks, abs=1e-12)
    low95, high95 = hand_interval(n_tasks, T_3_975)
    assert (values["cr1_low"], values["cr1_high"]) == pytest.approx((low95, high95), abs=1e-9)
    low90, high90 = hand_interval(n_tasks, T_3_95)
    assert (values["equivalence90_low"], values["equivalence90_high"]) == pytest.approx(
        (low90, high90), abs=1e-9
    )
    # Four clusters: the exact sign-flip p is at least 2 / 16, never below 0.05.
    assert values["signflip_p"] >= 0.125
    assert values["reading_5pp"] == reading
    assert values["tost_5pp"] == ("yes" if reading == "E3" else "no")


def test_security_equivalence_bound_is_strict() -> None:
    """An upper 90% bound just under the margin is equivalent; the margin itself is not."""
    report = security_report(25)
    centre = 0.8 / 100
    se = report["cr1_se"]
    for upper, expected in ((0.0499999, "E3"), (0.05, "E4")):
        # Move the centre so that the 90% upper bound lands on ``upper``.
        shift = upper - (centre + T_3_95 * se)
        moved = {**report, "cr1_ci": [x + shift for x in report["cr1_ci"]]}
        assert row_of(moved)["reading_5pp"] == expected
