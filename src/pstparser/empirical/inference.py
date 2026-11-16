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

Effects are on the proportion scale, treatment minus baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

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
