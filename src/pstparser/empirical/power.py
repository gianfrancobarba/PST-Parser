"""Power of the primary tests, computed with the functions of the real analysis (M11).

Correctness (independent tasks, one greedy pair): with discordance ``q`` and
effect ``delta`` the number of discordant pairs is ``m ~ Bin(n, q)`` and, given
``m``, the treatment-only pairs are ``c ~ Bin(m, (q + delta) / (2q))``. Any
between-task heterogeneity marginalises to these cells, so the power of the
mid-p test is a finite sum: no simulation is needed. The rejection region comes
from ``midp_mcnemar`` itself.

Security (families, replicates): the generative model of the analysis plan.
For family g and task i, ``m_g ~ Beta(mu kF, (1 - mu) kF)``,
``p_iT ~ Beta(m_g kT, (1 - m_g) kT)``, ``logit p_iH = logit p_iT + beta + w_g + w_i``
with ``w_g ~ N(0, tauF^2)`` and ``w_i ~ N(0, tauT^2)``; ``beta`` is calibrated so
that ``E[p_H - p_T] = delta``; replicates are Bernoulli given p. Each simulated
dataset is judged by ``sign_flip_test``. The model (independent replicates given
p, independent families) is an assumption of the plan, not a fact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from pstparser.empirical.analysis import TaskEstimate
from pstparser.empirical.inference import midp_mcnemar, sign_flip_test

#: Prompts per family in the security designs (00 C-5, C-6).
FAMILY_SIZES: Final = {
    "S2_100p_35f": [3] * 32 + [2] + [1] * 2,
    "S1_83p_29f": [3] * 27 + [1] * 2,
    "mirata_47p_16f": [3] * 15 + [2],
    "mirata_56p_19f": [3] * 18 + [2],
}
#: mu = P(N) under T; kF, kT Beta concentrations of family mean and task; tauF,
#: tauT logit-scale SDs of the effect between families and between tasks.
SCENARIOS: Final = {
    "base": {"mu": 0.55, "kF": 4.0, "kT": 8.0, "tauF": 0.3, "tauT": 0.6},
    "alta": {"mu": 0.55, "kF": 1.5, "kT": 4.0, "tauF": 0.8, "tauT": 0.8},
    "eterogenea": {"mu": 0.55, "kF": 4.0, "kT": 8.0, "tauF": 1.0, "tauT": 1.5},
}
MDE_STEP: Final = 0.0025
CALIBRATION_FAMILIES: Final = 100_000


@dataclass(frozen=True)
class PowerResult:
    """Rejections over simulated datasets, with the Monte Carlo standard error."""

    rejections: int
    sims: int
    rate: float
    mc_se: float


def _result(rejections: int, sims: int) -> PowerResult:
    rate = rejections / sims
    return PowerResult(rejections, sims, rate, math.sqrt(rate * (1 - rate) / sims))


def _log_factorials(n: int) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n + 1)))))


def _binomial_pmf(m: int, p: float, logfact: np.ndarray) -> np.ndarray:
    k = np.arange(m + 1)
    if p <= 0.0:
        return np.asarray(k == 0, dtype=float)
    if p >= 1.0:
        return np.asarray(k == m, dtype=float)
    log = logfact[m] - logfact[k] - logfact[m - k] + k * math.log(p) + (m - k) * math.log1p(-p)
    return np.asarray(np.exp(log), dtype=float)


def midp_rejection_bounds(n: int, alpha: float) -> list[int]:
    """For each m in 0..n, the largest min(b, c) that the mid-p test rejects, or -1.

    The mid-p value grows with min(b, c) at fixed m, so the region is
    ``min(b, c) <= bound``. Each bound is decided by ``midp_mcnemar`` itself,
    searching from the bound of ``m - 1``, which is close.
    """

    def rejected(k: int, m: int) -> bool:
        return midp_mcnemar(k, m - k) <= alpha

    bounds = []
    bound = -1
    for m in range(n + 1):
        bound = min(bound, m // 2)
        while bound >= 0 and not rejected(bound, m):
            bound -= 1
        while bound + 1 <= m // 2 and rejected(bound + 1, m):
            bound += 1
        bounds.append(bound)
    return bounds


_BOUNDS: dict[tuple[int, float], list[int]] = {}


def midp_power(n: int, q: float, delta: float, alpha: float) -> float:
    """Exact power of the two-sided mid-p McNemar test.

    Raises:
        ValueError: If ``q`` is not in (0, 1] or ``|delta| > q``.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0 < q <= 1 or abs(delta) > q + 1e-12:
        raise ValueError("need 0 < q <= 1 and |delta| <= q")
    key = (n, alpha)
    if key not in _BOUNDS:
        _BOUNDS[key] = midp_rejection_bounds(n, alpha)
    bounds = _BOUNDS[key]
    logfact = _log_factorials(n)
    share = min(1.0, max(0.0, (q + delta) / (2 * q)))
    discordant = _binomial_pmf(n, q, logfact)
    total = 0.0
    for m in range(n + 1):
        if discordant[m] < 1e-300 or bounds[m] < 0:
            continue
        pmf = _binomial_pmf(m, share, logfact)
        bound = bounds[m]
        # Rejected splits: k <= bound or k >= m - bound (the two tails, or all of them).
        tails = (
            pmf.sum() if m - bound <= bound + 1 else pmf[: bound + 1].sum() + pmf[m - bound :].sum()
        )
        total += discordant[m] * float(tails)
    return float(min(1.0, total))


def mde(
    n: int, q: float, alpha: float, *, target: float = 0.8, step: float = MDE_STEP
) -> float | None:
    """Smallest delta on a grid of ``step`` with power at least ``target``, or None up to q."""
    index = 0
    while index * step <= q + 1e-12:
        delta = round(index * step, 10)
        if midp_power(n, q, min(delta, q), alpha) >= target:
            return delta
        index += 1
    return None


def simulate_paired(
    *,
    n_tasks: int,
    p_baseline: float,
    discordance: float,
    delta: float,
    alpha: float,
    sims: int,
    seed: int,
) -> PowerResult:
    """Monte Carlo power of the mid-p test on multinomial paired cells (checks ``midp_power``)."""
    pb, pc = (discordance - delta) / 2, (discordance + delta) / 2
    pa = p_baseline - pb
    pd = 1 - pa - pb - pc
    cells = np.array([pa, pb, pc, pd])
    if discordance <= 0 or np.any(cells < -1e-12):
        raise ValueError("impossible paired scenario")
    rng = np.random.default_rng(seed)
    tables = rng.multinomial(n_tasks, np.clip(cells, 0, 1) / np.clip(cells, 0, 1).sum(), size=sims)
    rejections = sum(midp_mcnemar(int(b), int(c)) <= alpha for b, c in tables[:, 1:3])
    return _result(int(rejections), sims)


def draw_latent(
    rng: np.random.Generator, sizes: Sequence[int], scenario: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Family index, P(N | T) and effect deviation of every task (logit scale)."""
    groups = len(sizes)
    mu, kf, kt = scenario["mu"], scenario["kF"], scenario["kT"]
    family_mean = np.clip(rng.beta(mu * kf, (1 - mu) * kf, size=groups), 1e-4, 1 - 1e-4)
    family = np.repeat(np.arange(groups), sizes)
    p_t = np.clip(
        rng.beta(family_mean[family] * kt, (1 - family_mean[family]) * kt), 1e-6, 1 - 1e-6
    )
    deviation = rng.normal(0, scenario["tauF"], size=groups)[family] + rng.normal(
        0, scenario["tauT"], size=len(family)
    )
    return family, p_t, deviation


def _logit(p: np.ndarray) -> np.ndarray:
    return np.asarray(np.log(p / (1 - p)), dtype=float)


def _expit(x: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-x)), dtype=float)


def calibrate_shift(scenario: dict[str, float], delta: float, *, seed: int) -> float:
    """Logit shift beta with ``E[p_H - p_T] = delta``, by bisection on a large draw."""
    if delta == 0 and scenario["tauF"] == 0 and scenario["tauT"] == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    _, p_t, deviation = draw_latent(rng, [3] * CALIBRATION_FAMILIES, scenario)
    base = _logit(p_t)
    low, high = -6.0, 6.0
    for _ in range(80):
        middle = (low + high) / 2
        if float(np.mean(_expit(base + middle + deviation) - p_t)) < delta:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def simulate_clustered(
    *,
    sizes: Sequence[int],
    scenario: str,
    delta: float,
    replicates: int,
    alpha: float,
    sims: int,
    seed: int,
    samples: int = 9_999,
) -> PowerResult:
    """Power (or size, at delta 0) of the studentized sign-flip over families."""
    parameters = SCENARIOS[scenario]
    seeds = np.random.SeedSequence(seed).spawn(2)
    shift = calibrate_shift(parameters, delta, seed=int(seeds[0].generate_state(1)[0]))
    rng = np.random.default_rng(seeds[1])
    names = [f"f{index:03d}" for index in range(len(sizes))]
    rejections = 0
    for _ in range(sims):
        family, p_t, deviation = draw_latent(rng, sizes, parameters)
        p_h = _expit(_logit(p_t) + shift + deviation)
        y_t = rng.binomial(replicates, p_t) / replicates
        y_h = rng.binomial(replicates, p_h) / replicates
        estimates = [
            TaskEstimate(
                f"t{i}", names[family[i]], float(y_t[i]), float(y_h[i]), float(y_h[i] - y_t[i])
            )
            for i in range(len(family))
        ]
        test_seed = int(rng.integers(0, 2**31 - 1))
        result = sign_flip_test(estimates, samples=samples, seed=test_seed)
        rejections += result.p_value <= alpha
    return _result(rejections, sims)


COLUMNS: Final = (
    "kind",
    "n",
    "q",
    "alpha",
    "size",
    "power_3pp",
    "power_5pp",
    "mde80",
    "design",
    "families",
    "tasks",
    "scenario",
    "replicates",
    "delta",
    "sims",
    "samples",
    "rejection_rate",
    "mc_se",
)


def power_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of the power table for the paired (exact) and clustered (simulated) cells."""
    if set(grid) - {"seed", "paired", "clustered"} or "seed" not in grid:
        raise ValueError("the grid holds seed, paired and clustered only")
    rows: list[dict[str, Any]] = []
    for cell in grid.get("paired", []):
        n, q, alpha = cell["n"], cell["q"], cell["alpha"]
        rows.append(
            {
                "kind": "paired",
                "n": n,
                "q": q,
                "alpha": alpha,
                "size": midp_power(n, q, 0.0, alpha),
                "power_3pp": midp_power(n, q, 0.03, alpha) if q >= 0.03 else None,
                "power_5pp": midp_power(n, q, 0.05, alpha) if q >= 0.05 else None,
                "mde80": mde(n, q, alpha),
            }
        )
    cells = grid.get("clustered", [])
    streams = np.random.SeedSequence(grid["seed"]).spawn(len(cells))
    for cell, stream in zip(cells, streams, strict=True):
        sizes = FAMILY_SIZES[cell["design"]]
        alpha = cell.get("alpha", 0.05)
        samples = cell.get("samples", 9_999)
        result = simulate_clustered(
            sizes=sizes,
            scenario=cell["scenario"],
            delta=cell["delta"],
            replicates=cell["replicates"],
            alpha=alpha,
            sims=cell["sims"],
            seed=int(stream.generate_state(1)[0]),
            samples=samples,
        )
        rows.append(
            {
                "kind": "clustered",
                "alpha": alpha,
                "design": cell["design"],
                "families": len(sizes),
                "tasks": sum(sizes),
                "scenario": cell["scenario"],
                "replicates": cell["replicates"],
                "delta": cell["delta"],
                "sims": cell["sims"],
                "samples": samples,
                "rejection_rate": result.rate,
                "mc_se": result.mc_se,
            }
        )
    return rows
