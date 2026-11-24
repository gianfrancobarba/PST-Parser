"""Primary tests of the analysis: mid-p McNemar, studentized sign-flip, weighted Holm."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from pstparser.empirical.analysis import TaskEstimate, exact_mcnemar
from pstparser.empirical.inference import (
    CONFIRMATORY_WEIGHTS,
    midp_mcnemar,
    sign_flip_test,
    weighted_holm,
)

# --- mid-p McNemar -------------------------------------------------------------------------


def midp_by_fractions(b: int, c: int) -> float:
    """2 * (P(X <= k) - P(X = k) / 2), X ~ Bin(b + c, 1/2), k = min(b, c) (Fagerland 2013)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    cdf = sum(Fraction(math.comb(n, x), 2**n) for x in range(k + 1))
    return float(min(Fraction(1), 2 * (cdf - Fraction(math.comb(n, k), 2 ** (n + 1)))))


def test_midp_known_values() -> None:
    assert midp_mcnemar(0, 5) == midp_mcnemar(5, 0) == 0.03125  # 0.0625 - 0.03125
    assert midp_mcnemar(3, 10) == 470 / 8192  # (2 * 378 - 286) / 2^13
    assert midp_mcnemar(0, 0) == 1.0
    # Ties: 2 * (P(X <= k) - P(X = k) / 2) = 1 exactly, the usual two-sided mid-p.
    # An early draft of the specification gave 0.75 for (1, 1), a value that no
    # single definition gives together with (0, 5).
    assert midp_mcnemar(1, 1) == 1.0
    assert midp_mcnemar(7, 7) == 1.0


def test_midp_matches_the_rational_definition_and_never_exceeds_the_exact_test() -> None:
    for b in range(0, 31):
        for c in range(0, 31):
            assert midp_mcnemar(b, c) == pytest.approx(midp_by_fractions(b, c), rel=1e-12)
            assert midp_mcnemar(b, c) <= exact_mcnemar(b, c)
            assert midp_mcnemar(b, c) == midp_mcnemar(c, b)


def test_midp_scales_to_the_planned_sample() -> None:
    assert midp_mcnemar(40, 72) == pytest.approx(midp_by_fractions(40, 72), rel=1e-12)
    assert 0 < midp_mcnemar(300, 420) < 1e-5


@pytest.mark.parametrize("counts", [(-1, 2), (1.0, 2), (True, 2)])
def test_midp_refuses_invalid_counts(counts) -> None:
    with pytest.raises(ValueError):
        midp_mcnemar(*counts)


# --- Studentized sign-flip -----------------------------------------------------------------


def estimates(differences, families=None):
    return [
        TaskEstimate(f"t{i}", None if families is None else families[i], 0.0, d, d)
        for i, d in enumerate(differences)
    ]


def brute_force(sums, sizes, studentized):
    """Enumerate every sign vector with itertools: the reference for the exact p."""
    sums, sizes = np.asarray(sums, float), np.asarray(sizes, float)
    g, total = len(sums), sizes.sum()

    def statistic(s):
        est = s.sum() / total
        if not studentized:
            return abs(est)
        se = math.sqrt(g / (g - 1) * float(np.sum((s - est * sizes) ** 2))) / total
        if se == 0:
            return math.inf if est != 0 else 0.0
        return abs(est) / se

    observed = statistic(sums)
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=g):
        value = statistic(np.array(signs) * sums)
        hits += value == math.inf if observed == math.inf else value >= observed - 1e-9
    return hits / 2**g


def test_all_zero_differences_give_one() -> None:
    result = sign_flip_test(estimates([0.0] * 12, [f"f{i // 3}" for i in range(12)]))
    assert result.p_value == 1.0 and result.exact and result.clusters == 4


@pytest.mark.parametrize("studentized", [True, False])
def test_five_equal_positive_families_give_two_in_thirty_two(studentized) -> None:
    families = [f"f{i // 3}" for i in range(15)]
    result = sign_flip_test(estimates([0.4] * 15, families), studentized=studentized)
    assert result.p_value == 2 / 32
    assert result.exact and result.samples == 32 and result.exceedances == 2


def test_exact_p_equals_the_enumeration() -> None:
    rng = np.random.default_rng(3)
    sizes = [3, 3, 2, 3, 1, 3, 3, 2, 3]
    families = [f"f{g}" for g, size in enumerate(sizes) for _ in range(size)]
    differences = rng.choice([-1.0, -0.2, 0.0, 0.2, 0.6, 1.0], size=len(families))
    sums = [
        sum(d for d, f in zip(differences, families, strict=True) if f == name)
        for name in sorted(set(families))
    ]
    for studentized in (True, False):
        result = sign_flip_test(estimates(differences, families), studentized=studentized)
        assert result.exact
        assert result.p_value == pytest.approx(brute_force(sums, sizes, studentized), abs=1e-12)


def test_order_invariance_and_task_clusters() -> None:
    rng = np.random.default_rng(5)
    differences = list(rng.choice([-1.0, 0.0, 1.0], size=11))
    families = [f"f{i % 4}" for i in range(11)]
    data = estimates(differences, families)
    assert sign_flip_test(data) == sign_flip_test(list(reversed(data)))
    by_task = sign_flip_test(data, by_family=False)
    assert by_task.clusters == 11 and by_task.exact
    assert by_task.p_value == pytest.approx(brute_force(differences, [1] * 11, True), abs=1e-12)


def test_monte_carlo_agrees_with_enumeration_at_sixteen_families() -> None:
    rng = np.random.default_rng(11)
    sizes = [3] * 15 + [2]
    families = [f"f{g:02d}" for g, size in enumerate(sizes) for _ in range(size)]
    differences = rng.choice([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6], size=len(families)) + 0.05
    data = estimates(differences, families)
    exact = sign_flip_test(data)
    assert exact.exact and exact.samples == 2**16
    sampled = sign_flip_test(data, exact_threshold=15, samples=9_999, seed=1)
    assert not sampled.exact and sampled.samples == 9_999
    assert sampled.p_value == (sampled.exceedances + 1) / (9_999 + 1)
    error = math.sqrt(exact.p_value * (1 - exact.p_value) / 9_999)
    assert abs(sampled.p_value - exact.p_value) <= 3 * error + 1 / 10_000
    assert sampled == sign_flip_test(data, exact_threshold=15, samples=9_999, seed=1)


def test_size_under_the_strong_null_is_at_most_alpha() -> None:
    # Strong null: within each task, T and H outcomes have the same distribution.
    rng = np.random.default_rng(20261116)
    sizes = [3] * 10
    families = [f"f{g}" for g, size in enumerate(sizes) for _ in range(size)]
    sims, rejected = 1_000, 0
    for _ in range(sims):
        family_rate = rng.beta(2.2, 1.8, size=len(sizes))
        p = np.repeat(family_rate, sizes)
        y_t = rng.binomial(5, p) / 5
        y_h = rng.binomial(5, p) / 5
        rejected += sign_flip_test(estimates(y_h - y_t, families)).p_value <= 0.05
    assert rejected / sims <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / sims)


def test_sign_flip_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sign_flip_test(estimates([0.1, 0.2], ["f", "f"]))
    with pytest.raises(ValueError, match="family"):
        sign_flip_test(estimates([0.1, 0.2]))
    with pytest.raises(ValueError, match="samples"):
        sign_flip_test(estimates([0.1, 0.2, 0.3], ["a", "b", "c"]), exact_threshold=1, samples=0)


# --- Weighted Holm on {H1, H3} -------------------------------------------------------------


def test_confirmatory_weights() -> None:
    assert CONFIRMATORY_WEIGHTS == {"H1": 0.8, "H3": 0.2}  # 0.04 and 0.01 at alpha 0.05


@pytest.mark.parametrize(
    ("p1", "p3", "expected"),
    [
        (0.045, 0.008, ("rejected", "rejected")),
        (0.045, 0.02, ("not_rejected", "not_rejected")),
        (0.03, 0.04, ("rejected", "rejected")),
        (0.03, 0.06, ("rejected", "not_rejected")),
        (0.06, 0.005, ("not_rejected", "rejected")),
    ],
)
def test_weighted_holm_known_cases(p1, p3, expected) -> None:
    decisions = weighted_holm({"H1": p1, "H3": p3}, CONFIRMATORY_WEIGHTS)
    assert (decisions["H1"], decisions["H3"]) == expected


def test_h3_out_of_the_family_leaves_h1_at_alpha() -> None:
    assert weighted_holm({"H1": 0.045, "H3": None}, CONFIRMATORY_WEIGHTS) == {
        "H1": "rejected",
        "H3": "dropped",
    }
    assert weighted_holm({"H1": 0.051, "H3": None}, CONFIRMATORY_WEIGHTS)["H1"] == "not_rejected"


def test_weighted_holm_follows_the_four_rules_of_the_plan() -> None:
    grid = sorted(
        {0.0, 0.005, 0.01, 0.0100001, 0.02, 0.04, 0.0400001, 0.045, 0.05, 0.0500001, 0.3, 1.0}
    )
    for p1, p3 in itertools.product(grid, repeat=2):
        decisions = weighted_holm({"H1": p1, "H3": p3}, CONFIRMATORY_WEIGHTS)
        h1 = p1 <= 0.04 or (p1 <= 0.05 and p3 <= 0.01)
        h3 = p3 <= 0.01 or (p3 <= 0.05 and p1 <= 0.04)
        assert decisions["H1"] == ("rejected" if h1 else "not_rejected"), (p1, p3)
        assert decisions["H3"] == ("rejected" if h3 else "not_rejected"), (p1, p3)


@pytest.mark.parametrize(
    ("p_values", "weights", "message"),
    [
        ({"H1": 0.01}, {"H1": 0.8, "H3": 0.2}, "same hypotheses"),
        ({"H1": 0.01, "H3": 0.2}, {"H1": 0.8, "H3": 0.3}, "sum to 1"),
        ({"H1": 1.5, "H3": 0.2}, CONFIRMATORY_WEIGHTS, "p-values"),
        ({"H1": None, "H3": None}, CONFIRMATORY_WEIGHTS, "at least one"),
    ],
)
def test_weighted_holm_refuses_bad_input(p_values, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_holm(p_values, weights)


# --- Primary tests in the paired report -----------------------------------------


def ledger(arms, *, endpoint="functional", families=None, replicates=(0,)):
    from pstparser.empirical.analysis import AnalysisDesign, OutcomeRecord, TaskSpec

    tasks = tuple(TaskSpec(str(i), families[i] if families else None) for i in range(len(arms)))
    design = AnalysisDesign("a", endpoint, tasks, replicates)
    records = [
        OutcomeRecord("a", endpoint, str(i), condition, replicate, outcome)
        for i, pair in enumerate(arms)
        for condition, values in zip(design.conditions, pair, strict=True)
        for replicate, outcome in zip(replicates, values, strict=True)
    ]
    return design, records


def test_single_greedy_pair_reports_the_midp_value() -> None:
    from pstparser.empirical.analysis import analyze_paired

    arms = [(["fail"], ["pass"])] * 5 + [(["pass"], ["fail"])] + [(["pass"], ["pass"])] * 4
    report = analyze_paired(*ledger(arms), bootstrap_samples=50)
    assert (
        report.mcnemar_table["baseline_only"] == 1 and report.mcnemar_table["treatment_only"] == 5
    )
    assert report.midp_p == midp_mcnemar(1, 5) == 0.125  # (2 * (1 + 6) - 6) / 2^6
    assert report.signflip_p is None and report.signflip_unit is None
    assert report.as_dict()["midp_p"] == 0.125


def test_families_report_the_studentized_sign_flip() -> None:
    from pstparser.empirical.analysis import analyze_paired

    outcomes = [("N", "V"), ("V", "N"), ("N", "N"), ("V", "N")] * 6
    families = [f"f{i // 3}" for i in range(24)]
    arms = [([t] * 2, [h] * 2) for t, h in outcomes]
    report = analyze_paired(
        *ledger(arms, endpoint="security", families=families, replicates=(1, 2)),
        bootstrap_samples=50,
        seed=7,
    )
    expected = sign_flip_test(report.task_estimates, by_family=True, seed=7)
    assert report.signflip_p == expected.p_value
    assert report.signflip_exact is True and report.signflip_samples == 2**8
    assert report.signflip_unit == "family" and report.midp_p is None


def test_replicates_without_families_flip_tasks() -> None:
    from pstparser.empirical.analysis import analyze_paired

    arms = [(["fail", "pass"], ["pass", "pass"])] * 3 + [(["pass", "pass"], ["fail", "pass"])] * 2
    report = analyze_paired(*ledger(arms, replicates=(1, 2)), bootstrap_samples=50)
    assert report.signflip_unit == "task" and report.signflip_samples == 2**5
    assert report.signflip_p == sign_flip_test(report.task_estimates, by_family=False).p_value
