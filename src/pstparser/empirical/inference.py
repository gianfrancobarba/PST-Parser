"""Primary inference of the analysis plan: mid-p McNemar, studentized sign-flip, weighted Holm.

- Correctness (one greedy pair per independent task): McNemar mid-p, the exact
  conditional p-value minus the probability of the observed split (Fagerland,
  Lydersen and Laake 2013). The exact test stays in ``analysis.py`` as a
  conservative sensitivity analysis.
- Security (families of prompts, sampled replicates): sign-flip of whole
  clusters with the statistic studentized by the CR1 standard error, recomputed
  for every sign vector; exact enumeration up to 16 clusters, Monte Carlo with
  a seed beyond.
- Confirmatory family {H1, H3}: Holm with weights 0.8 / 0.2 (levels 0.04 / 0.01
  at alpha 0.05) and recycling of the weight of a rejected hypothesis, the
  two-hypothesis case of the graphical method of Bretz et al. (2009).
- Intervals (paired-v2, P1): Tango's asymptotic score interval for a paired
  difference (primary for correctness, at 95% and 90%), Newcombe's method 10
  (sensitivity) and the CR1 cluster-robust interval with t(G - 1) quantiles
  (primary for security).

Effects are on the proportion scale, treatment minus baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from pstparser.empirical.analysis import TaskEstimate

#: Weights of the confirmatory family (00 A-3): levels 0.04 and 0.01 at alpha 0.05.
CONFIRMATORY_WEIGHTS: Final = {"H1": 0.8, "H3": 0.2}
EXACT_THRESHOLD: Final = 16
Decision = Literal["rejected", "not_rejected", "dropped"]


def midp_mcnemar(baseline_only: int, treatment_only: int) -> float:
    """Two-sided mid-p McNemar test on the discordant pairs.

    ``2 * (P(X <= k) - P(X = k) / 2)`` with ``X ~ Bin(b + c, 1/2)`` and
    ``k = min(b, c)``, capped at 1. Computed with exact integers. A tie gives 1.

    Raises:
        ValueError: If a count is not a non-negative integer.
    """
    if any(type(n) is not int or n < 0 for n in (baseline_only, treatment_only)):
        raise ValueError("Discordant counts must be nonnegative integers")
    n = baseline_only + treatment_only
    k = min(baseline_only, treatment_only)
    numerator = 2 * sum(math.comb(n, x) for x in range(k + 1)) - math.comb(n, k)
    return float(min(1.0, numerator / 2**n))


@dataclass(frozen=True)
class SignFlipResult:
    """Outcome of a sign-flip test; ``p_value`` is exceedances / samples when exact."""

    p_value: float
    exceedances: int
    samples: int
    exact: bool
    clusters: int
    studentized: bool
    seed: int | None


def _statistics(
    sums: np.ndarray, sizes: np.ndarray, signs: np.ndarray, studentized: bool
) -> np.ndarray:
    flipped = signs * sums
    total = sizes.sum()
    estimate = flipped.sum(axis=1) / total
    if not studentized:
        return np.asarray(np.abs(estimate), dtype=float)
    clusters = len(sums)
    residuals = flipped - estimate[:, None] * sizes
    se = np.sqrt(clusters / (clusters - 1) * np.sum(residuals**2, axis=1)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(estimate) / se
    # A zero standard error is an infinite statistic, unless the estimate is zero too.
    return np.where(se > 0, ratio, np.where(estimate != 0, np.inf, 0.0))


def sign_flip_test(
    estimates: Sequence[TaskEstimate],
    *,
    by_family: bool = True,
    studentized: bool = True,
    samples: int = 9_999,
    seed: int = 42,
    exact_threshold: int = EXACT_THRESHOLD,
) -> SignFlipResult:
    """Test "no effect of the form" by flipping the sign of whole clusters.

    Clusters are families (``by_family``) or single tasks. The statistic is the
    task-weighted mean difference, divided by its CR1 standard error when
    ``studentized``; it is recomputed for every sign vector. With at most
    ``exact_threshold`` clusters every one of the 2^G vectors is enumerated and
    ``p = exceedances / 2^G``; otherwise ``samples`` Rademacher vectors are
    drawn with ``seed`` and ``p = (exceedances + 1) / (samples + 1)``.

    Raises:
        ValueError: If a family is missing, fewer than two clusters remain, or
            the Monte Carlo sample size is not positive.
    """
    groups: dict[str, list[float]] = {}
    for estimate in estimates:
        if by_family:
            if estimate.family_id is None:
                raise ValueError("by_family requires a family_id for every task")
            key = estimate.family_id
        else:
            key = estimate.task_id
        groups.setdefault(key, []).append(estimate.difference)
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError("the sign-flip test needs at least two clusters")
    sums = np.array([math.fsum(groups[name]) for name in names])
    sizes = np.array([len(groups[name]) for name in names], dtype=float)
    observed = float(_statistics(sums, sizes, np.ones((1, len(names))), studentized)[0])
    exact = len(names) <= exact_threshold
    if exact:
        index = np.arange(2 ** len(names))[:, None]
        signs = np.where((index >> np.arange(len(names))) & 1, -1.0, 1.0)
    else:
        if type(samples) is not int or samples < 1:
            raise ValueError("samples must be a positive integer")
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(names)))
    values = _statistics(sums, sizes, signs, studentized)
    if math.isinf(observed):
        hits = int(np.sum(np.isinf(values)))
    else:
        hits = int(np.sum(values >= observed - 1e-9 * max(1.0, observed)))
    drawn = len(signs)
    return SignFlipResult(
        p_value=hits / drawn if exact else (hits + 1) / (drawn + 1),
        exceedances=hits,
        samples=drawn,
        exact=exact,
        clusters=len(names),
        studentized=studentized,
        seed=None if exact else seed,
    )


def weighted_holm(
    p_values: Mapping[str, float | None],
    weights: Mapping[str, float],
    alpha: float = 0.05,
) -> dict[str, Decision]:
    """Weighted Holm with recycling; a ``None`` p-value drops its hypothesis.

    At each step a remaining hypothesis is rejected if ``p <= alpha * w / W``,
    where ``W`` sums the weights of the hypotheses still in the family; a
    rejected hypothesis leaves the family and its weight goes to the others.
    A dropped hypothesis (for example H3 after a failed parser gate) gives its
    weight away from the start, so H1 alone is read at ``alpha``.

    Raises:
        ValueError: If names differ, weights are invalid, a p-value is outside
            [0, 1], or every hypothesis is dropped.
    """
    if set(p_values) != set(weights):
        raise ValueError("p-values and weights must name the same hypotheses")
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()) or not math.isclose(
        math.fsum(weights.values()), 1.0, abs_tol=1e-12
    ):
        raise ValueError("weights must be positive and sum to 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    for value in p_values.values():
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("p-values must be finite and in [0, 1]")
    remaining = {name for name, value in p_values.items() if value is not None}
    if not remaining:
        raise ValueError("at least one hypothesis must stay in the family")
    decisions: dict[str, Decision] = {
        name: "dropped" if p_values[name] is None else "not_rejected" for name in p_values
    }
    while True:
        total = math.fsum(weights[name] for name in remaining)
        rejected = {
            name
            for name in remaining
            if p_values[name] is not None and p_values[name] <= alpha * (weights[name] / total)  # type: ignore[operator]
        }
        if not rejected:
            return decisions
        for name in rejected:
            decisions[name] = "rejected"
        remaining -= rejected
        if not remaining:
            return decisions


def _check_table(both_pass: int, treatment_only: int, baseline_only: int, both_fail: int) -> int:
    cells = (both_pass, treatment_only, baseline_only, both_fail)
    if any(type(n) is not int or n < 0 for n in cells):
        raise ValueError("table cells must be nonnegative integers")
    total = sum(cells)
    if total == 0:
        raise ValueError("the table is empty")
    return total


def _check_confidence(confidence: float) -> float:
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def tango_score(baseline_only: int, treatment_only: int, n: int, delta: float) -> float:
    """Tango's (1998) score statistic for ``H0: p_treatment - p_baseline = delta``.

    ``(c - b - n*delta) / sqrt(n * (2*q + delta*(1 - delta)))``, where ``q`` is the
    restricted maximum-likelihood estimate of the baseline-only cell probability
    (the positive root of Tango's quadratic). At ``delta = 0`` it is McNemar's
    ``(c - b) / sqrt(b + c)``. A null variance gives an infinite statistic with
    the sign of the numerator (0 when the numerator is 0).
    """
    b, c = float(baseline_only), float(treatment_only)
    quadratic = 2.0 * n
    linear = -b - c + (2.0 * n - c + b) * delta
    constant = -b * delta * (1.0 - delta)
    root = (math.sqrt(max(linear * linear - 4.0 * quadratic * constant, 0.0)) - linear) / (
        2.0 * quadratic
    )
    variance = n * (2.0 * max(root, 0.0) + delta * (1.0 - delta))
    numerator = c - b - n * delta
    if variance <= 0:
        return 0.0 if numerator == 0 else math.copysign(math.inf, numerator)
    return numerator / math.sqrt(variance)


def tango_interval(
    baseline_only: int, treatment_only: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Tango's asymptotic score interval for a paired difference of proportions.

    The set of ``delta`` in [-1, 1] not rejected by :func:`tango_score` at the
    two-sided level ``1 - confidence``; the bounds are found by bisection to
    machine precision. Checked against ``ratesci::pairbinci`` (method "Score",
    ``skew = FALSE``, ``bcf = FALSE``).

    Raises:
        ValueError: If the counts are not a valid table or ``confidence`` is
            outside (0, 1).
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    _check_table(0, treatment_only, baseline_only, n - baseline_only - treatment_only)
    z = _check_confidence(confidence)
    estimate = (treatment_only - baseline_only) / n

    def score(delta: float) -> float:
        return tango_score(baseline_only, treatment_only, n, delta)

    def crossing(left: float, right: float, level: float) -> float:
        # The score decreases in delta: score(left) > level >= score(right).
        for _ in range(200):
            middle = (left + right) / 2
            if middle in (left, right):
                break
            if score(middle) > level:
                left = middle
            else:
                right = middle
        return (left + right) / 2

    lower = -1.0 if score(-1.0) <= z else crossing(-1.0, estimate, z)
    upper = 1.0 if score(1.0) >= -z else crossing(estimate, 1.0, -z)
    return lower, upper


def _wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_interval(table: Mapping[str, int], confidence: float = 0.95) -> tuple[float, float]:
    """Newcombe's (1998) method 10 for a paired difference (sensitivity analysis).

    Wilson intervals of the two marginal proportions combined through the
    correlation coefficient phi, with Newcombe's correction: when
    ``AD > BC`` the numerator of phi is ``max(AD - BC - n/2, 0)``; phi is 0 when
    a margin is empty. ``table`` uses the keys of ``PairedReport.mcnemar_table``.
    Checked against ``ratesci::pairbinci`` (method "MOVER_newc", Wilson base).

    Raises:
        ValueError: If a cell is missing or invalid, or ``confidence`` is
            outside (0, 1).
    """
    try:
        a, b, c, d = (
            table["both_pass"],
            table["treatment_only"],
            table["baseline_only"],
            table["both_fail"],
        )
    except KeyError as error:
        raise ValueError(f"missing table cell: {error}") from error
    n = _check_table(a, b, c, d)
    z = _check_confidence(confidence)
    treatment, baseline = (a + b) / n, (a + c) / n
    l1, u1 = _wilson(a + b, n, z)
    l2, u2 = _wilson(a + c, n, z)
    margins = (a + b) * (c + d) * (a + c) * (b + d)
    cross = a * d - b * c
    if margins == 0:
        phi = 0.0
    else:
        phi = (max(cross - n / 2, 0.0) if cross > 0 else cross) / math.sqrt(margins)
    estimate = treatment - baseline
    lo1, up2 = treatment - l1, u2 - baseline
    up1, lo2 = u1 - treatment, baseline - l2
    down = lo1**2 - 2 * phi * lo1 * up2 + up2**2
    up = up1**2 - 2 * phi * up1 * lo2 + lo2**2
    return estimate - math.sqrt(max(down, 0.0)), estimate + math.sqrt(max(up, 0.0))


@dataclass(frozen=True)
class ClusterInterval:
    """Task-weighted mean difference with its CR1 standard error and t interval."""

    estimate: float
    standard_error: float
    clusters: int
    degrees_of_freedom: int
    interval: tuple[float, float]
    confidence: float


def cluster_t_interval(
    estimates: Sequence[TaskEstimate], *, by_family: bool = True, confidence: float = 0.95
) -> ClusterInterval:
    """CR1 cluster-robust interval with ``t(G - 1)`` quantiles.

    The estimate is ``sum(S_g) / sum(n_g)`` over clusters (families, or single
    tasks); the standard error is ``sqrt(G/(G-1) * sum((S_g - estimate*n_g)^2)) /
    sum(n_g)``, the same one that studentizes :func:`sign_flip_test`. With one
    task per cluster it is the paired t interval.

    Raises:
        ValueError: If a family is missing, fewer than two clusters remain, or
            ``confidence`` is outside (0, 1).
    """
    _check_confidence(confidence)
    groups: dict[str, list[float]] = {}
    for estimate in estimates:
        if by_family and estimate.family_id is None:
            raise ValueError("by_family requires a family_id for every task")
        key = estimate.family_id if by_family else estimate.task_id
        groups.setdefault(str(key), []).append(estimate.difference)
    if len(groups) < 2:
        raise ValueError("a cluster interval needs at least two clusters")
    sums = [math.fsum(values) for values in groups.values()]
    sizes = [len(values) for values in groups.values()]
    total = sum(sizes)
    centre = math.fsum(sums) / total
    clusters = len(groups)
    squares = math.fsum((s - centre * m) ** 2 for s, m in zip(sums, sizes, strict=True))
    se = math.sqrt(clusters / (clusters - 1) * squares) / total
    quantile = float(stats.t.ppf(1 - (1 - confidence) / 2, clusters - 1))
    return ClusterInterval(
        estimate=centre,
        standard_error=se,
        clusters=clusters,
        degrees_of_freedom=clusters - 1,
        interval=(centre - quantile * se, centre + quantile * se),
        confidence=confidence,
    )


#: Equivalence margin of the primary reading (00 A-4), on the proportion scale.
TOST_MARGIN: Final = 0.05
Reading = Literal["E1", "E2", "E3", "E4"]


def tost(interval: Sequence[float], margin: float = TOST_MARGIN) -> bool:
    """Two one-sided tests at level alpha each, read on the (1 - 2 alpha) interval.

    Equivalence is declared only if the whole interval (90% for alpha = 0.05:
    Tango's for one greedy pair per task, CR1 with families) lies strictly
    inside ``(-margin, margin)``. A non-significant difference is never
    equivalence by itself.

    Raises:
        ValueError: If the margin is not positive or the interval is not ordered.
    """
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("the margin must be positive")
    low, high = interval
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError("the interval must be finite and ordered")
    return -margin < low and high < margin


def cluster_interval_at(
    interval: Sequence[float], standard_error: float, degrees_of_freedom: int, confidence: float
) -> tuple[float, float]:
    """The CR1 interval of a report at another confidence (same centre, se and t(G - 1))."""
    _check_confidence(confidence)
    if degrees_of_freedom < 1 or not math.isfinite(standard_error) or standard_error < 0:
        raise ValueError("need at least one degree of freedom and a finite standard error")
    centre = (interval[0] + interval[1]) / 2
    quantile = float(stats.t.ppf(1 - (1 - confidence) / 2, degrees_of_freedom))
    return centre - quantile * standard_error, centre + quantile * standard_error


def four_way_reading(
    p_value: float,
    interval_90: Sequence[float],
    *,
    alpha: float = 0.05,
    margin: float = TOST_MARGIN,
) -> Reading:
    """The frozen four-way reading (06 §3.5.2): difference test crossed with TOST.

    E1 relevant effect (different, not equivalent); E2 different from zero but
    negligible (different and equivalent); E3 equivalence (only equivalent);
    E4 inconclusive (neither). The difference is read from the primary test.
    """
    if not 0 < alpha < 1 or not 0 <= p_value <= 1:
        raise ValueError("alpha must lie in (0, 1) and the p-value in [0, 1]")
    different = p_value <= alpha
    equivalent = tost(interval_90, margin)
    if different:
        return "E2" if equivalent else "E1"
    return "E3" if equivalent else "E4"
