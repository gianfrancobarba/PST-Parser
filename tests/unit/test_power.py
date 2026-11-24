"""Power of the primary tests: exact mid-p power and MDE, clustered sign-flip simulation."""

import json
import math

import numpy as np
import pytest
from typer.testing import CliRunner

from pstparser.empirical.cli import app
from pstparser.empirical.inference import midp_mcnemar
from pstparser.empirical.power import (
    FAMILY_SIZES,
    SCENARIOS,
    calibrate_shift,
    mde,
    midp_power,
    midp_rejection_bounds,
    simulate_clustered,
    simulate_paired,
)

Q = (0.10, 0.15, 0.20, 0.30)


def test_family_sizes_and_scenarios_of_the_plan() -> None:
    assert {name: (sum(s), len(s)) for name, s in FAMILY_SIZES.items()} == {
        "S2_100p_35f": (100, 35),
        "S1_83p_29f": (83, 29),
        "mirata_47p_16f": (47, 16),
        "mirata_56p_19f": (56, 19),
    }
    assert SCENARIOS["base"] == {"mu": 0.55, "kF": 4.0, "kT": 8.0, "tauF": 0.3, "tauT": 0.6}


@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.3])
def test_the_rejection_region_is_the_one_of_the_midp_function(alpha) -> None:
    bounds = midp_rejection_bounds(160, alpha)
    for m in range(161):
        region = {k for k in range(m + 1) if midp_mcnemar(k, m - k) <= alpha}
        assert region == {k for k in range(m + 1) if min(k, m - k) <= bounds[m]}, m


@pytest.mark.parametrize("alpha", [0.05, 0.04, 0.01])
def test_the_size_of_the_midp_test_never_exceeds_alpha(alpha) -> None:
    sizes = [midp_power(800, q, 0.0, alpha) for q in np.arange(0.02, 0.41, 0.02)]
    assert max(sizes) <= alpha
    assert max(sizes) > 0.75 * alpha  # not wastefully conservative like the exact test


@pytest.mark.parametrize(
    ("n", "alpha", "expected"),
    [
        (800, 0.05, (0.0325, 0.04, 0.045, 0.055)),  # before the weighted Holm
        (800, 0.04, (0.0325, 0.04, 0.0475, 0.0575)),  # H1
        (600, 0.01, (0.045, 0.055, 0.0625, 0.0775)),  # H3 if H1 is not rejected
        (600, 0.05, (0.0375, 0.045, 0.0525, 0.0625)),  # H3 if H1 is rejected
    ],
)
def test_minimum_detectable_effects_of_the_plan(n, alpha, expected) -> None:
    assert tuple(mde(n, q, alpha) for q in Q) == pytest.approx(expected, abs=1e-12)


def test_power_at_five_points_for_h1() -> None:
    powers = [midp_power(800, q, 0.05, 0.04) for q in Q]
    assert powers == pytest.approx([0.995, 0.95, 0.87, 0.70], abs=0.005)


def test_mde_is_none_when_the_target_is_out_of_reach() -> None:
    # At alpha 0.05 the MDE is reached at delta = q (power 0.855); at alpha 0.025
    # it is not (power 0.737). An early power table printed the second case under
    # alpha 0.05.
    assert mde(145, 0.05, 0.05) == 0.05
    assert midp_power(145, 0.05, 0.05, 0.025) == pytest.approx(0.7369150134695646, rel=1e-9)
    assert mde(145, 0.05, 0.025) is None


@pytest.mark.parametrize(
    ("q", "delta", "alpha"), [(0.3, 0.1, 0.05), (0.5, -0.2, 0.2), (0.4, 0.4, 0.05)]
)
def test_exact_power_equals_full_enumeration_for_ten_tasks(q, delta, alpha) -> None:
    n = 10
    pb, pc = (q - delta) / 2, (q + delta) / 2
    total = 0.0
    for b in range(n + 1):
        for c in range(n + 1 - b):
            probability = (
                math.factorial(n)
                / (math.factorial(b) * math.factorial(c) * math.factorial(n - b - c))
                * pb**b
                * pc**c
                * (1 - q) ** (n - b - c)
            )
            if midp_mcnemar(b, c) <= alpha:
                total += probability
    assert midp_power(n, q, delta, alpha) == pytest.approx(total, abs=1e-12)


def test_monte_carlo_agrees_with_the_exact_power() -> None:
    exact = midp_power(145, 0.15, 0.06, 0.05)
    result = simulate_paired(
        n_tasks=145, p_baseline=0.41, discordance=0.15, delta=0.06, alpha=0.05, sims=4000, seed=9
    )
    assert result.sims == 4000 and result.rate == result.rejections / 4000
    assert abs(result.rate - exact) <= 3 * math.sqrt(exact * (1 - exact) / 4000)
    assert result == simulate_paired(
        n_tasks=145, p_baseline=0.41, discordance=0.15, delta=0.06, alpha=0.05, sims=4000, seed=9
    )


@pytest.mark.parametrize("bad", [{"delta": 0.2}, {"discordance": 0.0}, {"p_baseline": 0.01}])
def test_impossible_paired_scenarios_are_refused(bad) -> None:
    arguments = {
        "n_tasks": 50,
        "p_baseline": 0.4,
        "discordance": 0.1,
        "delta": 0.05,
        "alpha": 0.05,
        "sims": 10,
        "seed": 1,
    }
    with pytest.raises(ValueError):
        simulate_paired(**{**arguments, **bad})


def test_the_shift_is_calibrated_to_the_requested_difference() -> None:
    rng = np.random.default_rng(4)
    for delta in (0.0, 0.10):
        shift = calibrate_shift(SCENARIOS["base"], delta, seed=3)
        sizes = [3] * 100_000
        from pstparser.empirical.power import draw_latent

        _, p_t, w = draw_latent(rng, sizes, SCENARIOS["base"])
        p_h = 1 / (1 + np.exp(-(np.log(p_t / (1 - p_t)) + shift + w)))
        assert float(np.mean(p_h - p_t)) == pytest.approx(delta, abs=0.003)


def test_clustered_size_under_the_null_and_reproducibility() -> None:
    arguments = {
        "sizes": FAMILY_SIZES["S2_100p_35f"],
        "scenario": "base",
        "delta": 0.0,
        "replicates": 5,
        "alpha": 0.05,
        "sims": 300,
        "seed": 20261116,
        "samples": 199,
    }
    result = simulate_clustered(**arguments)
    assert result.rate <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 300)
    assert result == simulate_clustered(**arguments)


def test_clustered_power_grows_with_the_effect() -> None:
    common = {
        "sizes": FAMILY_SIZES["S2_100p_35f"],
        "scenario": "base",
        "replicates": 5,
        "alpha": 0.05,
        "sims": 200,
        "seed": 5,
        "samples": 199,
    }
    assert simulate_clustered(delta=0.20, **common).rate > 0.9  # MDE 9 pp


def test_power_command_writes_the_same_csv_for_the_same_seed(tmp_path) -> None:
    grid = {
        "seed": 11,
        "paired": [{"n": 800, "q": 0.15, "alpha": 0.04}, {"n": 145, "q": 0.05, "alpha": 0.025}],
        "clustered": [
            {
                "design": "mirata_47p_16f",
                "scenario": "base",
                "replicates": 5,
                "delta": 0.0,
                "sims": 20,
                "samples": 99,
            }
        ],
    }
    (tmp_path / "grid.json").write_text(json.dumps(grid), encoding="utf-8")
    outputs = []
    for name in ("a", "b"):
        out = tmp_path / name / "power.csv"
        result = CliRunner().invoke(
            app, ["power", "--grid", str(tmp_path / "grid.json"), "--out", str(out)]
        )
        assert result.exit_code == 0, result.exception
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]
    lines = outputs[0].decode("utf-8").split("\n")
    header = lines[0].split(",")
    assert header[:4] == ["kind", "n", "q", "alpha"]
    first = dict(zip(header, lines[1].split(","), strict=True))
    assert first["kind"] == "paired" and first["mde80"] == "0.04"
    second = dict(zip(header, lines[2].split(","), strict=True))
    assert second["mde80"] == ""
    third = dict(zip(header, lines[3].split(","), strict=True))
    assert third["kind"] == "clustered" and third["families"] == "16" and third["tasks"] == "47"
    again = CliRunner().invoke(
        app,
        [
            "power",
            "--grid",
            str(tmp_path / "grid.json"),
            "--out",
            str(tmp_path / "a" / "power.csv"),
        ],
    )
    assert again.exit_code != 0
