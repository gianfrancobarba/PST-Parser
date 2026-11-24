"""Reading tables of the correctness contrasts: H1, mechanism, concordance, external anchor.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md`` (A1, A3, A5, A6, A9, A10). The
frozen estimates are read from the frozen tables; what is added is descriptive:

- (A5) risk ratio of the two pass rates, and odds ratio of the discordant
  pairs c/b with the exact conditional interval (Clopper-Pearson on c/(b+c));
- (A3) minimum detectable effect at the realised n and discordant share
  (exact mid-p power, 80%, alpha 0.05), by the frozen ``power.mde``;
- (A1) per-protocol: the frozen ``analyze_paired`` on the tasks where both
  answers have code (no ``no_code``, no parser failure), read E1-E4 as the
  frozen tables do; labelled as conditioning on a post-treatment variable;
- (A6) concordance: share of tasks with the same outcome, every pair of
  conditions, on the tasks present in both;
- the spread of the pass rates of T, E, F and H-gold (the "format spread");
- (A10) pass@1 of T on the 600 and on the 73 Hard tasks, with the
  Clopper-Pearson interval, beside the leaderboard value of the generator;
- (A9) a forest plot of the correctness contrasts and a concordance heatmap.

usage (repository root, .venv):
  python -I results/tools/results_correctness.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from scipy.stats import beta

from pstparser.empirical.analysis import analyze_paired, read_outcomes
from pstparser.empirical.inference import four_way_reading, tost
from pstparser.empirical.power import mde

VERSION: Final = "results-correctness-v1"
ALPHA: Final = 0.05
MODEL: Final = "Qwen2.5-Coder-7B-Instruct"
LEADERBOARD: Final = "outputs/valutazione-ricerca/03/raw"
CORE_CONDITIONS: Final = ("T", "E", "F", "H-gold")
ALL_CONDITIONS: Final = ("T", "E", "F", "H-gold", "H-auto")


def _frame_module() -> ModuleType:
    path = Path(__file__).with_name("results_frame.py")
    spec = importlib.util.spec_from_file_location("results_frame", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frame = _frame_module()


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval."""
    tail = (1 - confidence) / 2
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, n - successes + 1))
    high = 1.0 if successes == n else float(beta.ppf(1 - tail, successes + 1, n - successes))
    return low, high


def discordant_odds_ratio(b: int, c: int) -> dict[str, Any]:
    """c/b with the exact conditional interval; ``None`` where unbounded or undefined."""
    if b + c == 0:
        return {"odds_ratio": None, "ci95": [None, None]}
    low, high = clopper_pearson(c, b + c)

    def odds(p: float) -> float | None:
        return None if p >= 1 else p / (1 - p)

    return {"odds_ratio": None if b == 0 else c / b, "ci95": [odds(low), odds(high)]}


def risk_ratio(table: Mapping[str, int]) -> float | None:
    """Pass rate of the treatment over the pass rate of the baseline."""
    baseline = table["both_pass"] + table["baseline_only"]
    treatment = table["both_pass"] + table["treatment_only"]
    return None if baseline == 0 else treatment / baseline


def _by_task(cells: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    tasks: dict[str, dict[str, Mapping[str, Any]]] = {}
    for cell in cells:
        tasks.setdefault(cell["task_id"], {})[cell["condition"]] = cell
    return tasks


def concordance(cells: Sequence[Mapping[str, Any]], conditions: Sequence[str]) -> dict[str, Any]:
    """Share of tasks with the same outcome in two conditions, and its n."""
    tasks = _by_task(cells)
    shares: dict[str, Any] = {a: {} for a in conditions}
    counts: dict[str, dict[str, int]] = {a: {} for a in conditions}
    for a in conditions:
        for b in conditions:
            both = [arms for arms in tasks.values() if a in arms and b in arms]
            same = sum(arms[a]["outcome"] == arms[b]["outcome"] for arms in both)
            shares[a][b] = same / len(both) if both else None
            counts[a][b] = len(both)
    return {**shares, "n": counts}


def per_protocol_tasks(cells: Sequence[Mapping[str, Any]], pair: Sequence[str]) -> set[str]:
    """Tasks where both answers exist and have code (no ``no_code``, no parser failure)."""

    def has_code(cell: Mapping[str, Any]) -> bool:
        reason = str(cell["reason"])
        return reason != "no_code" and not reason.startswith("parser_failure")

    return {
        task
        for task, arms in _by_task(cells).items()
        if all(arm in arms and has_code(arms[arm]) for arm in pair)
    }


def format_spread(passes: Mapping[str, int], n: int) -> dict[str, float]:
    """Lowest and highest pass rate among the given conditions, and the range in pp."""
    rates = [count / n for count in passes.values()]
    return {"min": min(rates), "max": max(rates), "range_pp": 100 * (max(rates) - min(rates))}


def leaderboard_value(table: Any, model: str, column: str = "instruct") -> float:
    """The leaderboard score of ``model``.

    Raises:
        ValueError: If the model is absent or listed more than once.
    """
    rows = table[table["model"] == model]
    if len(rows) != 1:
        raise ValueError(f"{model}: {len(rows)} leaderboard rows")
    return float(rows[column].iloc[0])


def _paired_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_bytes().decode("utf-8")
    body = "\n".join(line for line in text.split("\n") if not line.startswith("#"))
    return list(csv.DictReader(io.StringIO(body)))


def _float(text: str) -> float | None:
    return None if text == "" else float(text)


def _per_protocol(root: Path, name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    design = frame.load_design(
        root / frame.TREE / "protocol/analysis/designs" / name / "design.json"
    )
    ledger = root / frame.SOURCES[name]
    records = read_outcomes(ledger)
    cells = [
        {"task_id": r.task_id, "condition": r.condition, "outcome": r.outcome, "reason": r.reason}
        for r in records
    ]
    keep = per_protocol_tasks(cells, design.conditions)
    cut = replace(design, tasks=tuple(t for t in design.tasks if t.task_id in keep))
    kept = [r for r in records if r.task_id in keep]
    result = analyze_paired(cut, kept, ledger_conditions=frame.ledger_conditions(ledger)).as_dict()
    reading = (
        four_way_reading(result["midp_p"], result["tango_ci_90"], alpha=ALPHA)
        if result["midp_p"] is not None and result["tango_ci_90"] is not None
        else None
    )
    return {
        "tasks": result["included_tasks"],
        "dropped": report["included_tasks"] - result["included_tasks"],
        "delta": result["delta"],
        "tango_ci": result["tango_ci"],
        "tango_ci_90": result["tango_ci_90"],
        "midp_p": result["midp_p"],
        "tost_5pp": tost(result["tango_ci_90"]) if result["tango_ci_90"] else None,
        "reading_5pp": reading,
    }


def contrasts(root: Path) -> list[dict[str, Any]]:
    """One row per frozen correctness design: frozen reading plus A1, A3 and A5."""
    rows_by_key = {
        (row["analysis_id"], row["baseline"], row["treatment"]): row
        for row in _paired_rows(root / frame.RUN / "thesis/paired.csv")
    }
    rows = []
    for name in frame.SOURCES:
        if not name.startswith("bcb-"):
            continue
        report = json.loads(
            (root / frame.RUN / "analysis" / name / "report.json").read_text("utf-8")
        )
        baseline, treatment = report["conditions"]
        frozen = rows_by_key[(report["analysis_id"], baseline, treatment)]
        table = report["mcnemar_table"]
        n = report["included_tasks"]
        b, c = table["baseline_only"], table["treatment_only"]
        q = (b + c) / n
        rows.append(
            {
                "design": name,
                "analysis_id": report["analysis_id"],
                "baseline": baseline,
                "treatment": treatment,
                "n": n,
                "pass_baseline": table["both_pass"] + b,
                "pass_treatment": table["both_pass"] + c,
                "b": b,
                "c": c,
                "concordant_share": (n - b - c) / n,
                "delta": report["delta"],
                "tango_ci": report["tango_ci"],
                "tango_ci_90": report["tango_ci_90"],
                "newcombe_ci": report["newcombe_ci"],
                "midp_p": report["midp_p"],
                "exact_p": report["mcnemar_p"],
                "reading_3pp": frozen["reading_3pp"],
                "reading_5pp": frozen["reading_5pp"],
                "reading_10pp": frozen["reading_10pp"],
                "decision": frozen["decision"],
                "holm_alpha": _float(frozen["holm_alpha"]),
                "risk_ratio": risk_ratio(table),
                **discordant_odds_ratio(b, c),
                "discordant_share_q": q,
                "mde_80": mde(n, q, ALPHA) if q > 0 else None,
                "sensitivity": report["sensitivity"],
                "per_protocol": _per_protocol(root, name, report),
            }
        )
    return rows


def anchor(root: Path, cells: Sequence[Mapping[str, Any]], hard: set[str]) -> dict[str, Any]:
    """T pass@1 on the 600 and on the Hard tasks, beside the leaderboard."""
    import pandas as pd

    t = [cell for cell in cells if cell["condition"] == "T"]
    passed = sum(cell["outcome"] == "pass" for cell in t)
    t_hard = [cell for cell in t if cell["task_id"] in hard]
    passed_hard = sum(cell["outcome"] == "pass" for cell in t_hard)
    full = pd.read_parquet(root / LEADERBOARD / "results_full.parquet")
    hard_board = pd.read_parquet(root / LEADERBOARD / "results_hard.parquet")
    return {
        "t_600": {
            "pass": passed,
            "n": len(t),
            "rate": passed / len(t),
            "ci95": clopper_pearson(passed, len(t)),
        },
        "t_hard": {
            "pass": passed_hard,
            "n": len(t_hard),
            "rate": passed_hard / len(t_hard),
            "ci95": clopper_pearson(passed_hard, len(t_hard)),
        },
        "leaderboard_instruct_full": leaderboard_value(full, MODEL),
        "leaderboard_instruct_hard": leaderboard_value(hard_board, MODEL),
        "differences": [
            "leaderboard: 1,140 tasks (Hard: 148); here the 600-task prefix (73 Hard)",
            "leaderboard: calibrated pass@1 and prefill; here the frozen wrapper and system prompt",
            "leaderboard: the authors' generation setup; here bf16 on one L4, frozen decoding",
        ],
    }


def _figures(rows: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        ax.plot([100 * v for v in row["tango_ci"]], [y, y], color="0.35", linewidth=1)
        ax.plot([100 * v for v in row["tango_ci_90"]], [y, y], color="0.1", linewidth=3.5)
        ax.plot(100 * row["delta"], y, "o", color="white", markeredgecolor="black", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    for edge in (-5, 5):
        ax.axvline(edge, color="0.5", linestyle="--", linewidth=0.8)
    labels = [f"{r['treatment']} - {r['baseline']} ({r['design']}, n={r['n']})" for r in rows]
    ax.set_yticks(range(len(rows)), labels[::-1], fontsize=8)
    ax.set_xlabel("Δ pass@1 (pp): thick 90% (TOST), thin 95% Tango; dashed ±5 pp")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "correctness-forest.png", dpi=200)
    plt.close(fig)

    conditions = [c for c in ALL_CONDITIONS if c in matrix]
    values = [[matrix[a][b] for b in conditions] for a in conditions]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    image = ax.imshow(values, vmin=0.85, vmax=1.0, cmap="Greys")
    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            ax.text(
                j,
                i,
                f"{100 * matrix[a][b]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[a][b] > 0.95 else "black",
            )
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_yticks(range(len(conditions)), conditions)
    fig.colorbar(image, ax=ax, label="same outcome (share of tasks)")
    fig.tight_layout()
    fig.savefig(out / "correctness-concordance.png", dpi=200)
    plt.close(fig)


def build(root: Path) -> dict[str, Any]:
    """Every table of the correctness reading."""
    records = read_outcomes(root / frame.EVAL / "ledgers/final-bcb.jsonl")
    cells = [
        {"task_id": r.task_id, "condition": r.condition, "outcome": r.outcome, "reason": r.reason}
        for r in records
    ]
    prefix = json.loads(
        (root / frame.TREE / "protocol/samples/prefix-600/bcb-prefix-600.json").read_text("utf-8")
    )
    hard = {t["task_id"] for t in prefix["tasks"] if t["hard"]}
    passes = {
        c: sum(cell["outcome"] == "pass" for cell in cells if cell["condition"] == c)
        for c in CORE_CONDITIONS
    }
    return {
        "version": VERSION,
        "contrasts": contrasts(root),
        "concordance": concordance(cells, ALL_CONDITIONS),
        "format_spread": {"passes": passes, **format_spread(passes, 600)},
        "anchor": anchor(root, cells, hard),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``correctness.json`` and the two figures into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "correctness.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _figures(result["contrasts"], result["concordance"], args.out)
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
