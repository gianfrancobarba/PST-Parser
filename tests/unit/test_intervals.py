"""Intervals of the analysis plan: Tango, Newcombe method 10, CR1 with t(G - 1)."""

import json
import math
from pathlib import Path
from statistics import NormalDist

import pytest

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskEstimate,
    TaskSpec,
    analyze_paired,
)
from pstparser.empirical.inference import (
    cluster_t_interval,
    midp_mcnemar,
    newcombe_interval,
    tango_interval,
    tango_score,
)

FIXTURES = Path(__file__).resolve().parents[1] / "assets" / "ratesci_pairbinci.json"
CASES = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]


def case_id(case: dict) -> str:
    cells = (case[k] for k in ("both_pass", "treatment_only", "baseline_only", "both_fail"))
    return "-".join(map(str, cells)) + f"@{case['level']}"


# --- Tango -------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_tango_matches_ratesci_pairbinci(case: dict) -> None:
    n = case["both_pass"] + case["treatment_only"] + case["baseline_only"] + case["both_fail"]
    lower, upper = tango_interval(case["baseline_only"], case["treatment_only"], n, case["level"])
    assert lower == pytest.approx(case["tango"][0], abs=1e-9)
    assert upper == pytest.approx(case["tango"][1], abs=1e-9)


def test_fixtures_cover_the_edge_cases_and_both_levels() -> None:
    levels = {case["level"] for case in CASES}
    tables = {case_id(case).split("@")[0] for case in CASES}
    assert levels == {0.95, 0.90}
    assert len(CASES) == 32
    # No discordance, one-sided discordance, all discordant, n = 800 tables.
    assert {"0-0-0-10", "0-3-0-0", "0-0-4-0", "300-40-25-435", "1-1-7-12"} <= tables


@pytest.mark.parametrize(("b", "c", "n"), [(3, 10, 40), (25, 40, 800), (7, 1, 21), (12, 12, 50)])
def test_tango_at_zero_is_the_mcnemar_statistic(b: int, c: int, n: int) -> None:
    assert tango_score(b, c, n, 0.0) == pytest.approx((c - b) / math.sqrt(b + c), rel=1e-12)


@pytest.mark.parametrize(("b", "c", "n"), [(3, 10, 40), (25, 40, 800), (30, 52, 800), (6, 1, 30)])
def test_zero_is_in_the_tango_interval_iff_asymptotic_mcnemar_accepts(
    b: int, c: int, n: int
) -> None:
    z = NormalDist().inv_cdf(0.975)
    lower, upper = tango_interval(b, c, n, 0.95)
    assert (lower <= 0 <= upper) == (abs(c - b) / math.sqrt(b + c) <= z)


def test_tango_interval_bounds_and_nesting() -> None:
    wide = tango_interval(30, 52, 800, 0.95)
    narrow = tango_interval(30, 52, 800, 0.90)
    estimate = (52 - 30) / 800
    assert -1 <= wide[0] < narrow[0] < estimate < narrow[1] < wide[1] <= 1
    # Each bound is where the score crosses the normal quantile.
    z = NormalDist().inv_cdf(0.975)
    assert tango_score(30, 52, 800, wide[0]) == pytest.approx(z, abs=1e-6)
    assert tango_score(30, 52, 800, wide[1]) == pytest.approx(-z, abs=1e-6)


def test_tango_swapping_the_arms_mirrors_the_interval() -> None:
    lower, upper = tango_interval(30, 52, 800)
    mirrored = tango_interval(52, 30, 800)
    assert mirrored == pytest.approx((-upper, -lower), abs=1e-12)


@pytest.mark.parametrize(
    ("b", "c", "n", "confidence"),
    [(-1, 2, 10, 0.95), (2, 2, 3, 0.95), (1, 1, 0, 0.95), (1, 1, 10, 1.0), (1.0, 1, 10, 0.95)],
)
def test_tango_rejects_invalid_input(b, c, n, confidence) -> None:
    with pytest.raises(ValueError):
        tango_interval(b, c, n, confidence)


# --- Newcombe -----------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_newcombe_matches_ratesci_pairbinci(case: dict) -> None:
    lower, upper = newcombe_interval(case, case["level"])
    assert lower == pytest.approx(case["newcombe"][0], abs=1e-9)
    assert upper == pytest.approx(case["newcombe"][1], abs=1e-9)


def test_newcombe_requires_every_cell() -> None:
    with pytest.raises(ValueError, match="missing table cell"):
        newcombe_interval({"both_pass": 1, "treatment_only": 1, "baseline_only": 1})


# --- CR1 with t(G - 1) --------------------------------------------------------------------


def estimate(task: str, family: str | None, difference: float) -> TaskEstimate:
    return TaskEstimate(task, family, 0.0, difference, difference)


def test_cr1_with_one_task_per_cluster_is_the_paired_t_interval() -> None:
    differences = [0.2, -0.2, 0.4, 0.0, 0.6, 0.2]
    result = cluster_t_interval(
        [estimate(f"t{i}", None, d) for i, d in enumerate(differences)], by_family=False
    )
    mean = sum(differences) / 6
    sd = math.sqrt(sum((d - mean) ** 2 for d in differences) / 5)
    # t quantile 0.975 with 5 degrees of freedom (2.570581835636315, standard tables).
    half = 2.570581835636315 * sd / math.sqrt(6)
    assert result.estimate == pytest.approx(mean)
    assert result.standard_error == pytest.approx(sd / math.sqrt(6))
    assert result.degrees_of_freedom == 5
    assert result.interval == pytest.approx((mean - half, mean + half))


def test_cr1_on_families_by_hand() -> None:
    # Family sums S = (1.0, -0.2, 0.6), sizes n = (2, 1, 3): estimate 1.4/6.
    rows = [
        estimate("a1", "A", 0.6),
        estimate("a2", "A", 0.4),
        estimate("b1", "B", -0.2),
        estimate("c1", "C", 0.2),
        estimate("c2", "C", 0.2),
        estimate("c3", "C", 0.2),
    ]
    result = cluster_t_interval(rows, confidence=0.90)
    centre = 1.4 / 6
    residuals = [1.0 - 2 * centre, -0.2 - centre, 0.6 - 3 * centre]
    se = math.sqrt(3 / 2 * sum(r * r for r in residuals)) / 6
    # t quantile 0.95 with 2 degrees of freedom: 2.919985580353724.
    assert result.clusters == 3
    assert result.standard_error == pytest.approx(se)
    assert result.interval == pytest.approx(
        (centre - 2.919985580353724 * se, centre + 2.919985580353724 * se)
    )


def test_cr1_refuses_missing_families_and_a_single_cluster() -> None:
    with pytest.raises(ValueError, match="family_id"):
        cluster_t_interval([estimate("a", None, 0.1), estimate("b", "F", 0.1)])
    with pytest.raises(ValueError, match="two clusters"):
        cluster_t_interval([estimate("a", "F", 0.1), estimate("b", "F", 0.2)])


# --- Report ---------------------------------------------------------------------------------


def ledger(pairs: list[tuple[str, str]], families: list[str] | None = None) -> tuple:
    tasks = tuple(TaskSpec(f"t{i}", families[i] if families else None) for i in range(len(pairs)))
    design = AnalysisDesign("a", "functional", tasks)
    records = [
        OutcomeRecord("a", "functional", f"t{i}", condition, 0, outcome)
        for i, pair in enumerate(pairs)
        for condition, outcome in zip(("T", "H-gold"), pair, strict=True)
    ]
    return design, records


def test_the_report_carries_tango_newcombe_and_no_cluster_interval() -> None:
    pairs = [("pass", "fail")] * 2 + [("fail", "pass")] * 7 + [("pass", "pass")] * 6
    design, records = ledger(pairs + [("fail", "fail")] * 5)
    report = analyze_paired(design, records, bootstrap_samples=200)
    assert report.mcnemar_table == {
        "both_fail": 5,
        "baseline_only": 2,
        "treatment_only": 7,
        "both_pass": 6,
    }
    assert report.tango_ci == tango_interval(2, 7, 20, 0.95)
    assert report.tango_ci_90 == tango_interval(2, 7, 20, 0.90)
    assert report.newcombe_ci == newcombe_interval(report.mcnemar_table, 0.95)
    assert report.cr1_ci is None and report.cr1_se is None and report.cr1_df is None
    assert report.as_dict()["tango_ci"] == report.tango_ci


def test_a_disagreement_between_mid_p_and_tango_is_declared() -> None:
    # b = 0, c = 5 on n = 5: mid-p 0.03125 rejects; Tango's interval for (0, 5, 5).
    design, records = ledger([("fail", "pass")] * 5)
    report = analyze_paired(design, records, bootstrap_samples=200)
    midp = midp_mcnemar(0, 5)
    excludes = report.tango_ci[0] > 0 or report.tango_ci[1] < 0
    disagree = (midp <= 0.05) != excludes
    assert ("Mid-p and Tango disagree" in " ".join(report.warnings)) == disagree


def test_families_give_a_cr1_interval_and_no_tango() -> None:
    pairs = [
        ("fail", "pass"),
        ("pass", "pass"),
        ("fail", "fail"),
        ("pass", "fail"),
        ("fail", "pass"),
    ]
    design, records = ledger(pairs, families=["A", "A", "B", "C", "C"])
    report = analyze_paired(design, records, bootstrap_samples=200)
    expected = cluster_t_interval(report.task_estimates)
    assert report.tango_ci is None and report.newcombe_ci is None
    assert report.cr1_ci == expected.interval
    assert report.cr1_se == expected.standard_error
    assert report.cr1_df == 2


# --- Coverage and agreement of the intervals ---------------------------------


def test_mid_p_and_tango_disagree_at_six_and_fifteen() -> None:
    # Independent values: the mid-p of Fagerland et al. (2013) from the binomial
    # Bin(21, 1/2), and Tango's score test at delta = 0, which is McNemar's
    # z = (c - b) / sqrt(b + c) = 9 / sqrt(21) = 1.964 > 1.960.
    tail = sum(math.comb(21, k) for k in range(7)) / 2**21
    expected_midp = 2 * (tail - math.comb(21, 6) / 2**21 / 2)
    assert midp_mcnemar(6, 15) == pytest.approx(expected_midp, abs=1e-12)
    assert expected_midp > 0.05
    assert (15 - 6) / math.sqrt(21) > NormalDist().inv_cdf(0.975)
    pairs = [("pass", "fail")] * 6 + [("fail", "pass")] * 15 + [("pass", "pass")] * 19
    design, records = ledger(pairs)
    report = analyze_paired(design, records, bootstrap_samples=200)
    assert report.tango_ci[0] > 0
    assert "Mid-p and Tango disagree about zero" in " ".join(report.warnings)


def _tango_coverage(n: int, p10: float, p01: float) -> float:
    """Exact coverage of the 95% Tango interval; cells below exp(-28) are left out."""
    delta = p01 - p10
    covered = mass = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            rest = n - b - c
            log_p = (
                math.lgamma(n + 1)
                - math.lgamma(b + 1)
                - math.lgamma(c + 1)
                - math.lgamma(rest + 1)
                + b * math.log(p10)
                + c * math.log(p01)
                + rest * math.log(1 - p10 - p01)
            )
            if log_p < -28:
                continue
            p = math.exp(log_p)
            low, high = tango_interval(b, c, n, 0.95)
            mass += p
            covered += p * (low <= delta <= high)
    return covered / mass


@pytest.mark.parametrize("n, tolerance", [(30, 0.02), (100, 0.01), (800, 0.01)])
@pytest.mark.parametrize("p10, p01", [(0.05, 0.10), (0.10, 0.10), (0.05, 0.15)])
def test_tango_interval_covers_close_to_95_percent(n, tolerance, p10, p01) -> None:
    assert abs(_tango_coverage(n, p10, p01) - 0.95) <= tolerance
