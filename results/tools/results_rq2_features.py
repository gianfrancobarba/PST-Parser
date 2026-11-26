"""Task and representation features associated with changes of the correctness outcome.

Written after the frozen results were seen, to study which features of the
structured prompt go with the tasks that the reviewed PST wins or loses against
the original text. Committed and published before it was first run on the
study data; every number it writes is exploratory and descriptive, with 95%
intervals and no test. It reads only existing files and runs no generation and
no oracle.

Fixed before the first run:
- Reproduction first: the frozen subgroups and logistic model of the primary
  contrast (``exploratory/bcb-h1``) are recomputed with the frozen tool
  ``exploratory.py`` and must be identical, otherwise nothing is written.
- Features of each of the 600 tasks, none read from a model output:
  - frozen covariates: log ratio of PST to text tokens, log text tokens, number
    of segments, reordered segments, judgement units;
  - from the frozen reviewed annotation: share of characters in
    ``main_instruction``, ``context.constraints``, ``context.format`` and
    ``context.data``, and whether ``examples`` is present;
  - from the frozen dataset, as task difficulty independent of the answers:
    log lines of the reference solution, number of test methods, number of
    libraries, and the Hard flag.
- (P1) Associations, one feature at a time, by logistic regression (frozen
  Newton fit, Wald 95%): (a) on the discordant pairs of text against reviewed
  PST, the outcome is "the PST wins"; (b) on all 600 tasks, the outcome is
  "unstable", i.e. text, JSON envelope, flat JSON and reviewed PST do not all
  give the same outcome. Continuous features are standardised (odds ratio per
  standard deviation); binary features are used as they are.
- (P2) Ratio against absolute length and difficulty: the joint model of the
  three standardised features log ratio, log text tokens and log solution lines,
  for (a) and (b), read only if it converges and every VIF is at most 5 (the
  frozen rule of plan §6); the 3 x 3 cross of the terciles of log ratio and of
  log text tokens with wins and losses of the PST per cell (counts only); and
  the reviewed PST against the text by terciles of solution lines, with the
  frozen Tango interval from 20 tasks on.
- (P3) Profile of unstable against stable tasks: mean of each feature per
  group and the standardised mean difference; distribution of the number of
  forms that pass; the tasks that change outcome only with the reviewed PST,
  listed with their features for a qualitative reading.
- (P4) Answer side: per-task difference (form minus text) of output tokens and
  of extracted code lines, for the reviewed PST, the JSON envelope and the flat
  JSON, split by paired outcome (form wins, form loses, both pass, both fail);
  mean, median and a percentile bootstrap interval of the mean (10,000 draws,
  seed 42).

usage (repository root, .venv):
  python -I results/tools/results_rq2_features.py --root . --reproduce-only
  python -I results/tools/results_rq2_features.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np

VERSION: Final = "results-rq2-features-v1"
Z95: Final = 1.959963984540054
MAX_VIF: Final = 5.0
BOOTSTRAP_DRAWS: Final = 10_000
BOOTSTRAP_SEED: Final = 42
FORMS: Final = ("T", "E", "F", "H-gold")
PST: Final = "H-gold"
TEXT: Final = "T"
CATEGORIES: Final = {
    "main_instruction": "share_main_instruction",
    "context.constraints": "share_constraints",
    "context.format": "share_format",
    "context.data": "share_data",
}
CONTINUOUS: Final = (
    "log_ratio_HT",
    "log_tokens_T",
    "n_segments",
    "reorder",
    "n_judgement_units",
    "share_main_instruction",
    "share_constraints",
    "share_format",
    "share_data",
    "log_solution_lines",
    "test_methods",
    "n_libs",
)
BINARY: Final = ("has_examples", "hard")
JOINT: Final = ("log_ratio_HT", "log_tokens_T", "log_solution_lines")
GOLD: Final = "data/annotated/downstream/bcb-gold/annotations.jsonl"
DATASET: Final = "outputs/valutazione-ricerca/03/raw/full_v0.1.4.parquet"


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frame = _load("results_frame")
exploratory = _load("exploratory")
heterogeneity = _load("results_heterogeneity")


# --- features -----------------------------------------------------------------


def category_shares(spans: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Share of the annotated characters in each context leaf, and whether examples exist."""
    by_path: Counter[str] = Counter()
    for span in spans:
        by_path[span["path"]] += span["end"] - span["start"]
    total = sum(by_path.values()) or 1
    shares: dict[str, float] = {
        name: by_path.get(path, 0) / total for path, name in CATEGORIES.items()
    }
    shares["has_examples"] = int(by_path.get("examples", 0) > 0)
    return shares


def solution_lines(code: str) -> int:
    """Lines of a reference solution that are neither blank nor comments."""
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


def test_methods(source: str) -> int:
    """Test functions of a test file (``def test...``)."""
    return sum(1 for line in source.splitlines() if line.strip().startswith("def test"))


def stability(outcomes: Mapping[str, str]) -> dict[str, int]:
    """Whether the four forms disagree, whether the PST changes, and whether only the PST does."""
    values = {form: outcomes[form] for form in FORMS}
    unstable = len(set(values.values())) > 1
    pst_change = values[PST] != values[TEXT]
    others_equal = values["E"] == values[TEXT] and values["F"] == values[TEXT]
    return {
        "unstable": int(unstable),
        "pst_change": int(pst_change),
        "pst_only": int(pst_change and others_equal),
    }


def standardize(values: Mapping[str, float]) -> dict[str, float]:
    """Values centred on their mean and divided by their sample standard deviation."""
    mean = statistics.fmean(values.values())
    sd = statistics.stdev(values.values())
    return {key: (value - mean) / sd for key, value in values.items()}


# --- associations -------------------------------------------------------------


def association(x: Sequence[float], y: Sequence[int]) -> dict[str, Any]:
    """Logistic regression of ``y`` on one covariate with the frozen Newton fit (Wald 95%)."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if np.ptp(xs) == 0 or ys.min() == ys.max():
        return {"read": False, "reason": "no variation", "n": len(ys)}
    fit = exploratory._fit(np.column_stack([np.ones(len(ys)), xs]), ys)
    beta, se = float(fit["beta"][1]), float(fit["se"][1])
    return {
        "read": bool(fit["converged"]),
        "reason": None if fit["converged"] else "no convergence or separation",
        "n": len(ys),
        "events": int(ys.sum()),
        "log_odds": beta,
        "se": se,
        "odds_ratio": math.exp(beta),
        "or_ci95": [math.exp(beta - Z95 * se), math.exp(beta + Z95 * se)],
    }


def joint_model(columns: Mapping[str, Sequence[float]], y: Sequence[int]) -> dict[str, Any]:
    """Joint logistic model, read only if it converges and every VIF is at most 5."""
    names = list(columns)
    matrix = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    vif = dict(zip(names, exploratory._vif(matrix), strict=True))
    ys = np.asarray(y, dtype=float)
    result: dict[str, Any] = {"covariates": names, "n": len(ys), "vif": vif}
    if max(vif.values()) > MAX_VIF:
        return {**result, "read": False, "reason": f"a VIF above {MAX_VIF}"}
    fit = exploratory._fit(np.column_stack([np.ones(len(ys)), matrix]), ys)
    terms = {}
    for j, name in enumerate(names, start=1):
        beta, se = float(fit["beta"][j]), float(fit["se"][j])
        terms[name] = {
            "log_odds": beta,
            "se": se,
            "odds_ratio": math.exp(beta),
            "or_ci95": [math.exp(beta - Z95 * se), math.exp(beta + Z95 * se)],
        }
    return {
        **result,
        "read": bool(fit["converged"]),
        "reason": None if fit["converged"] else "no convergence or separation",
        "terms": terms,
    }


def cross_strata(
    pairs: Sequence[Mapping[str, Any]],
    first: Mapping[str, str],
    second: Mapping[str, str],
    treatment: str,
) -> list[dict[str, Any]]:
    """Tasks, treatment wins and losses in each cell of two tercile classifications."""
    winners = {pair["task_id"]: pair["winner"] for pair in pairs}
    cells = []
    for a in ("T1", "T2", "T3"):
        for b in ("T1", "T2", "T3"):
            tasks = [t for t in first if first[t] == a and second.get(t) == b]
            wins = sum(1 for t in tasks if winners.get(t) == treatment)
            losses = sum(1 for t in tasks if t in winners and winners[t] != treatment)
            cells.append(
                {
                    "first": a,
                    "second": b,
                    "tasks": len(tasks),
                    "pst_wins": wins,
                    "pst_losses": losses,
                }
            )
    return cells


# --- answer side --------------------------------------------------------------


def response_groups(
    outcome: Mapping[str, tuple[str, str]],
    values: Mapping[tuple[str, str], float | None],
    baseline: str,
    treatment: str,
) -> dict[str, list[float]]:
    """Per-task difference treatment minus baseline, split by paired outcome."""
    groups: dict[str, list[float]] = {
        "pst_wins": [],
        "pst_loses": [],
        "both_pass": [],
        "both_fail": [],
    }
    for task, (base, treat) in outcome.items():
        left, right = values.get((task, baseline)), values.get((task, treatment))
        if left is None or right is None:
            continue
        if base == treat:
            key = "both_pass" if base == "pass" else "both_fail"
        else:
            key = "pst_wins" if treat == "pass" else "pst_loses"
        groups[key].append(right - left)
    return groups


def summary(values: Sequence[float]) -> dict[str, Any]:
    """n, mean, median and a percentile bootstrap interval of the mean (seed 42)."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "mean_ci95": None}
    data = list(values)
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(statistics.fmean(rng.choices(data, k=len(data))) for _ in range(BOOTSTRAP_DRAWS))
    low = means[int(0.025 * BOOTSTRAP_DRAWS)]
    high = means[int(0.975 * BOOTSTRAP_DRAWS) - 1]
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "mean_ci95": [low, high],
    }


# --- study data ---------------------------------------------------------------


def _strip(document: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in document.items() if k not in ("version", "analysis_id")}


def reproduce(root: Path) -> dict[str, Any]:
    """Recompute the frozen subgroups and logistic model of the primary contrast."""
    tree = root / frame.TREE
    design = exploratory.load_design(tree / "protocol/analysis/designs/bcb-h1/design.json")
    records = exploratory.read_outcomes(root / frame.SOURCES["bcb-h1"])
    covariates = exploratory.read_covariates(tree / "protocol/covariates/bcb/covariates.jsonl")
    prefix = json.loads(
        (tree / "protocol/samples/prefix-600/bcb-prefix-600.json").read_text("utf-8")
    )
    hard = {t["task_id"] for t in prefix["tasks"] if t["hard"]}
    conditions = sorted({r.condition for r in records})
    fresh_subgroups = exploratory.subgroups(
        design, records, covariates, hard=hard, ledger_conditions=conditions
    )
    pairs = exploratory.discordant_pairs(design, records)
    fresh_logistic = exploratory.logistic(design, pairs, covariates)
    folder = root / frame.RUN / "exploratory/bcb-h1"
    diffs = frame.compare(
        _strip(json.loads((folder / "subgroups.json").read_text("utf-8"))),
        json.loads(json.dumps(fresh_subgroups)),
    ) + frame.compare(
        _strip(json.loads((folder / "logistic.json").read_text("utf-8"))),
        json.loads(json.dumps(fresh_logistic)),
    )
    return {"identical": not diffs, "differences": diffs, "pairs": pairs, "records": records}


def task_features(root: Path, tasks: Sequence[str]) -> dict[str, dict[str, float]]:
    """Every feature of every task, from frozen inputs only."""
    import pandas as pd

    tree = root / frame.TREE
    covariates = exploratory.read_covariates(tree / "protocol/covariates/bcb/covariates.jsonl")
    gold = {
        row["task_id"]: row
        for row in (
            json.loads(line)
            for line in (tree / GOLD).read_bytes().decode("utf-8").split("\n")
            if line.strip()
        )
    }
    prefix = json.loads(
        (tree / "protocol/samples/prefix-600/bcb-prefix-600.json").read_text("utf-8")
    )
    hard = {t["task_id"]: int(t["hard"]) for t in prefix["tasks"]}
    dataset = pd.read_parquet(root / DATASET).set_index("task_id")
    features: dict[str, dict[str, float]] = {}
    for task in tasks:
        row = dataset.loc[task]
        cov = covariates[task]
        features[task] = {
            "log_ratio_HT": cov["log_ratio_HT"],
            "log_tokens_T": cov["log_tokens_T"],
            "n_segments": cov["n_segments"],
            "reorder": cov["reorder"],
            "n_judgement_units": cov["n_judgement_units"],
            **category_shares(gold[task]["spans"]),
            "log_solution_lines": math.log(max(solution_lines(row["canonical_solution"]), 1)),
            "test_methods": test_methods(row["test"]),
            "n_libs": len(frame.parse_libs(row["libs"])),
            "hard": hard[task],
        }
    return features


def _values(features: Mapping[str, Mapping[str, float]], name: str) -> dict[str, float]:
    return {task: float(row[name]) for task, row in features.items()}


def _associations(
    features: Mapping[str, Mapping[str, float]], outcome: Mapping[str, int]
) -> dict[str, Any]:
    tasks = sorted(outcome)
    result = {}
    for name in (*CONTINUOUS, *BINARY):
        raw = _values(features, name)
        sd = statistics.stdev(raw[t] for t in tasks)
        scale = standardize({t: raw[t] for t in tasks}) if name in CONTINUOUS else raw
        result[name] = {
            "scale": "per standard deviation" if name in CONTINUOUS else "binary",
            "sd": sd,
            **association([scale[t] for t in tasks], [outcome[t] for t in tasks]),
        }
    return result


def _joint(
    features: Mapping[str, Mapping[str, float]], outcome: Mapping[str, int]
) -> dict[str, Any]:
    tasks = sorted(outcome)
    columns = {}
    for name in JOINT:
        z = standardize({t: float(features[t][name]) for t in tasks})
        columns[name] = [z[t] for t in tasks]
    return joint_model(columns, [outcome[t] for t in tasks])


def build(root: Path) -> dict[str, Any]:
    """All analyses P1-P4, after the reproduction of the frozen estimates."""
    check = reproduce(root)
    if not check["identical"]:
        raise SystemExit(f"frozen estimates not reproduced: {check['differences'][:5]}")
    records = check["records"]
    pairs = check["pairs"]
    outcomes: dict[str, dict[str, str]] = {}
    for record in records:
        outcomes.setdefault(record.task_id, {})[record.condition] = record.outcome
    tasks = sorted(t for t, o in outcomes.items() if all(f in o for f in FORMS))
    features = task_features(root, tasks)
    classes = {t: stability(outcomes[t]) for t in tasks}
    unstable = {t: classes[t]["unstable"] for t in tasks}
    pst_wins = {p["task_id"]: int(p["winner"] == PST) for p in pairs}

    # P2: crosses and difficulty strata.
    ratio = exploratory._terciles(_values({t: features[t] for t in tasks}, "log_ratio_HT"))
    length = exploratory._terciles(_values({t: features[t] for t in tasks}, "log_tokens_T"))
    difficulty = exploratory._terciles(
        _values({t: features[t] for t in tasks}, "log_solution_lines")
    )
    cells = [
        {"task_id": t, "condition": c, "outcome": outcomes[t][c]} for t in tasks for c in FORMS
    ]
    by_difficulty = [
        {
            "tercile": level,
            **heterogeneity.paired_counts(
                cells, (TEXT, PST), {t for t in tasks if difficulty[t] == level}
            ),
        }
        for level in ("T1", "T2", "T3")
    ]

    # P3: profile of unstable and stable tasks.
    profile = {}
    for name in (*CONTINUOUS, *BINARY):
        raw = _values(features, name)
        groups = {k: [raw[t] for t in tasks if unstable[t] == k] for k in (1, 0)}
        pooled = math.sqrt(
            (
                (len(groups[1]) - 1) * statistics.variance(groups[1])
                + (len(groups[0]) - 1) * statistics.variance(groups[0])
            )
            / (len(groups[1]) + len(groups[0]) - 2)
        )
        diff = statistics.fmean(groups[1]) - statistics.fmean(groups[0])
        profile[name] = {
            "mean_unstable": statistics.fmean(groups[1]),
            "mean_stable": statistics.fmean(groups[0]),
            "standardised_difference": diff / pooled if pooled > 0 else None,
        }
    passing = Counter(sum(outcomes[t][f] == "pass" for f in FORMS) for t in tasks)
    only = [
        {
            "task_id": t,
            "pst": outcomes[t][PST],
            "text": outcomes[t][TEXT],
            "features": features[t],
        }
        for t in tasks
        if classes[t]["pst_only"]
    ]

    # P4: answer side.
    replicate = {(r.task_id, r.condition): r.replicate_id for r in records}
    attributes = frame.bcb_cells(root)
    answer: dict[str, Any] = {}
    for measure in ("output_tokens", "sanitized_lines"):
        values = {
            (t, c): attributes[(t, c, replicate[(t, c)])].get(measure) for t in tasks for c in FORMS
        }
        answer[measure] = {}
        for form in (PST, "E", "F"):
            paired = {t: (outcomes[t][TEXT], outcomes[t][form]) for t in tasks}
            groups = response_groups(paired, values, TEXT, form)
            answer[measure][form] = {key: summary(v) for key, v in groups.items()}

    return {
        "version": VERSION,
        "reproduction": {"identical": True, "of": "exploratory/bcb-h1 subgroups and logistic"},
        "tasks": len(tasks),
        "discordant_pairs": len(pairs),
        "unstable_tasks": sum(unstable.values()),
        "pst_change_tasks": sum(c["pst_change"] for c in classes.values()),
        "pst_only_tasks": sum(c["pst_only"] for c in classes.values()),
        "p1_pst_wins": _associations(features, pst_wins),
        "p1_unstable": _associations(features, unstable),
        "p2_joint_pst_wins": _joint(features, pst_wins),
        "p2_joint_unstable": _joint(features, unstable),
        "p2_correlation_ratio_length": float(
            np.corrcoef(
                [features[t]["log_ratio_HT"] for t in tasks],
                [features[t]["log_tokens_T"] for t in tasks],
            )[0, 1]
        ),
        "p2_cross_ratio_by_length": cross_strata(pairs, ratio, length, PST),
        "p2_by_solution_lines": by_difficulty,
        "p3_profile": profile,
        "p3_forms_passing": {str(k): passing[k] for k in sorted(passing)},
        "p3_pst_only": only,
        "p4_answer_side": answer,
    }


def _figure(result: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [*CONTINUOUS, *BINARY]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.35 * len(names) + 1.5), sharey=True)
    for ax, key, title in (
        (axes[0], "p1_pst_wins", "PST wins (discordant pairs)"),
        (axes[1], "p1_unstable", "Unstable across forms (600 tasks)"),
    ):
        for index, name in enumerate(names):
            row = result[key][name]
            y = len(names) - 1 - index
            if not row.get("read"):
                continue
            ax.plot(row["or_ci95"], [y, y], color="0.2", linewidth=1.5)
            ax.plot(row["odds_ratio"], y, "o", color="white", markeredgecolor="black", zorder=3)
        ax.axvline(1, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("odds ratio (per SD; binary as is), Wald 95%")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(names)), names[::-1], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``rq2-features.json`` and the association figure into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--reproduce-only",
        action="store_true",
        help="only recompute the frozen estimates and report whether they are identical",
    )
    args = parser.parse_args(argv)
    if args.reproduce_only:
        check = reproduce(args.root)
        print(json.dumps({"identical": check["identical"], "differences": check["differences"]}))
        if not check["identical"]:
            raise SystemExit(1)
        return
    if args.out is None:
        parser.error("--out is required")
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "rq2-features.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _figure(result, args.out / "rq2-associations.png")
    print(json.dumps({"out": str(args.out), "reproduction": result["reproduction"]}))


if __name__ == "__main__":
    main()
