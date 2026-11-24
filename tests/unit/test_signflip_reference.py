"""The studentized sign-flip by family against values worked out by hand.

Five families with task differences a = [1], b = [0.5, 0], c = [-0.5],
d = [1, 0, 0], e = [0]: family sums S = (1, 0.5, -0.5, 1, 0), sizes n = (1, 2, 1, 3, 1),
N = 8. Estimate 2/8 = 0.25; residuals S_g - 0.25 n_g = (0.75, 0, -0.75, 0.25, -0.25),
squares 1.25; CR1 standard error sqrt(5/4 * 1.25) / 8 = 0.15625; statistic 1.6.
Enumerating the 32 sign vectors, 8 reach 1.6, so p = 0.25. Without studentization
12 of 32 reach |0.25|, so p = 0.375.
"""

import pytest

from pstparser.empirical.analysis import TaskEstimate
from pstparser.empirical.inference import cluster_t_interval, sign_flip_test

DIFFERENCES = {"a": [1.0], "b": [0.5, 0.0], "c": [-0.5], "d": [1.0, 0.0, 0.0], "e": [0.0]}
ESTIMATES = [
    TaskEstimate(f"{family}{i}", family, 0.0, value, value)
    for family, values in DIFFERENCES.items()
    for i, value in enumerate(values)
]


def test_studentized_exact_p_by_hand() -> None:
    result = sign_flip_test(ESTIMATES, by_family=True, studentized=True)
    assert result.exact and result.samples == 32 and result.clusters == 5
    assert result.exceedances == 8
    assert result.p_value == pytest.approx(0.25)


def test_plain_exact_p_by_hand() -> None:
    result = sign_flip_test(ESTIMATES, by_family=True, studentized=False)
    assert result.exceedances == 12
    assert result.p_value == pytest.approx(0.375)


def test_cr1_standard_error_by_hand() -> None:
    interval = cluster_t_interval(ESTIMATES, by_family=True)
    assert interval.estimate == pytest.approx(0.25)
    assert interval.standard_error == pytest.approx(0.15625)
    assert interval.degrees_of_freedom == 4


def test_monte_carlo_uses_the_planned_seed_by_default() -> None:
    # The plan fixes seed 42 for the sign-flip draws when G > 16.
    default = sign_flip_test(ESTIMATES, exact_threshold=0, samples=999)
    seeded = sign_flip_test(ESTIMATES, exact_threshold=0, samples=999, seed=42)
    assert default.seed == 42
    assert default.p_value == seeded.p_value and default.exceedances == seeded.exceedances
