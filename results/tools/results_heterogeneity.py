"""Heterogeneity of the correctness contrasts: frozen subgroups, domains, contamination.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md`` (A4, A7, A9). Descriptive only.

- Frozen subgroups of plan §6, read from ``exploratory/<design>/subgroups.json``
  and ``logistic.json`` (computed before this analysis); the log-odds of the
  continuous covariate are also given per step of 0.1 for reading.
- (A7) BigCodeBench domain: a task belongs to every domain of its libraries,
  by the authors' table ``bigcodebench-lib2domain.json`` (``analysis/lib2domain.json``
  of bigcode-project/bigcodebench, branch main, last commit 2a11c993 of
  25/07/2024, git blob d06a48ae), stored before any per-domain count.
- (A4) contamination: tasks with a high 10-gram overlap with the reference
  solution in either arm against the others.
- Each level: n, b, c, and under 20 tasks the counts only; otherwise Δ with the
  frozen Tango interval, no test.

usage (repository root, .venv):
  python -I results/tools/results_heterogeneity.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from pstparser.empirical.inference import tango_interval

VERSION: Final = "results-heterogeneity-v1"
MIN_TASKS: Final = 20
Z95: Final = 1.959963984540054
DOMAIN_TABLE: Final = Path(__file__).with_name("bigcodebench-lib2domain.json")
DESIGNS: Final = {
    "bcb-h1": ("T", "H-gold"),
    "bcb-h3-s-parse": ("T", "H-auto"),
    "bcb-rq3c": ("H-gold", "H-auto"),
}
CORE_PAIRS: Final = (("T", "H-gold"), ("T", "E"), ("T", "F"), ("E", "H-gold"), ("F", "H-gold"))


def _frame_module() -> ModuleType:
    path = Path(__file__).with_name("results_frame.py")
    spec = importlib.util.spec_from_file_location("results_frame", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frame = _frame_module()


def domains(
    libs: Mapping[str, Sequence[str]], mapping: Mapping[str, str]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Tasks of every domain (a task in each domain of its libraries), and unmapped libraries."""
    found: dict[str, set[str]] = {}
    unmapped: Counter[str] = Counter()
    for task, names in libs.items():
        for name in names:
            if name in mapping:
                found.setdefault(mapping[name], set()).add(task)
            else:
                unmapped[name] += 1
    return found, dict(unmapped)


def paired_counts(
    cells: Sequence[Mapping[str, Any]], pair: Sequence[str], tasks: set[str]
) -> dict[str, Any]:
    """n, b, c of a pair on ``tasks``; Δ and the Tango interval from 20 tasks on."""
    base, treat = pair
    arms: dict[str, dict[str, bool]] = {}
    for cell in cells:
        if cell["task_id"] in tasks and cell["condition"] in pair:
            arms.setdefault(cell["task_id"], {})[cell["condition"]] = cell["outcome"] == "pass"
    both = [a for a in arms.values() if base in a and treat in a]
    n = len(both)
    b = sum(a[base] and not a[treat] for a in both)
    c = sum(a[treat] and not a[base] for a in both)
    small = n < MIN_TASKS
    return {
        "n": n,
        "pass_baseline": sum(a[base] for a in both),
        "pass_treatment": sum(a[treat] for a in both),
        "b": b,
        "c": c,
        "reported": "counts only" if small else "estimate",
        "delta": None if small else (c - b) / n,
        "ci95": None if small else list(tango_interval(b, c, n)),
    }


def contaminated(overlap: Sequence[Mapping[str, Any]], pair: Sequence[str]) -> set[str]:
    """Tasks with a high overlap with the reference solution in either arm of ``pair``."""
    return {
        row["task_id"] for row in overlap if row["condition"] in pair and row["high_similarity"]
    }


def per_step(term: Mapping[str, float], step: float) -> dict[str, Any]:
    """A logistic term for an increase of ``step`` in the covariate (Wald 95%)."""
    log_odds = term["log_odds"] * step
    se = term["se"] * step
    return {
        "step": step,
        "odds_ratio": math.exp(log_odds),
        "or_ci95": [math.exp(log_odds - Z95 * se), math.exp(log_odds + Z95 * se)],
    }


def _frozen_subgroups(root: Path) -> dict[str, Any]:
    result = {}
    for design in DESIGNS:
        folder = root / frame.RUN / "exploratory" / design
        subgroups = json.loads((folder / "subgroups.json").read_text("utf-8"))
        logistic = json.loads((folder / "logistic.json").read_text("utf-8"))
        rows = []
        for grouping, value in subgroups.items():
            if not isinstance(value, dict) or "levels" not in value:
                continue
            for level in value["levels"]:
                counts = level["counts"]
                rows.append(
                    {
                        "grouping": grouping,
                        "level": level["level"],
                        "n": level["included_tasks"],
                        "b": counts["baseline_only"] if counts else None,
                        "c": counts["treatment_only"] if counts else None,
                        "reported": level["reported"],
                        "delta": level["delta"],
                        "ci95": level["ci95"],
                        "left_out_tasks": value["left_out_tasks"],
                    }
                )
        models = []
        for model in [*logistic.get("one_covariate", []), logistic.get("joint")]:
            if not model or not model.get("read"):
                continue
            terms = {}
            for name, term in model["terms"].items():
                terms[name] = {**term}
                if name == "log_ratio_HT":
                    terms[name]["per_0.1"] = per_step(term, 0.1)
            models.append({"covariates": model["covariates"], "terms": terms})
        result[design] = {
            "pair": DESIGNS[design],
            "levels": rows,
            "logistic": {
                "pairs": logistic.get("pairs"),
                "outcome": logistic.get("outcome"),
                "models": models,
            },
        }
    return result


def build(root: Path) -> dict[str, Any]:
    """Frozen subgroups, domains and contamination strata."""
    records = frame.read_outcomes(root / frame.EVAL / "ledgers/final-bcb.jsonl")
    cells = [
        {"task_id": r.task_id, "condition": r.condition, "outcome": r.outcome} for r in records
    ]
    attributes = frame.bcb_attributes(root)
    mapping = json.loads(DOMAIN_TABLE.read_text("utf-8"))
    found, unmapped = domains({t: a["libs"] for t, a in attributes.items()}, mapping)
    s_parse = {t for t, a in attributes.items() if a["s_parse"]}
    domain_rows = []
    for name in sorted(found):
        for pair in (*CORE_PAIRS, ("T", "H-auto"), ("H-gold", "H-auto")):
            tasks = found[name] & s_parse if "H-auto" in pair else found[name]
            domain_rows.append(
                {"domain": name, "pair": list(pair), **paired_counts(cells, pair, tasks)}
            )
    overlap = [
        json.loads(x)
        for x in (root / frame.RUN / "contamination/overlap.jsonl").read_text("utf-8").split("\n")
        if x.strip()
    ]
    all_tasks = set(attributes)
    contamination = []
    for pair in CORE_PAIRS:
        high = contaminated(overlap, pair)
        for stratum, tasks in (("high", high), ("not high", all_tasks - high)):
            contamination.append(
                {"pair": list(pair), "stratum": stratum, **paired_counts(cells, pair, tasks)}
            )
    return {
        "version": VERSION,
        "domain_table": {
            "file": DOMAIN_TABLE.name,
            "libraries": len(mapping),
            "unmapped": unmapped,
        },
        "tasks_per_domain": {name: len(tasks) for name, tasks in sorted(found.items())},
        "domains": domain_rows,
        "contamination": contamination,
        "frozen_subgroups": _frozen_subgroups(root),
    }


def _figure(result: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    for row in result["frozen_subgroups"]["bcb-h1"]["levels"]:
        if row["delta"] is not None and row["grouping"] not in ("leaves", "has_code"):
            entries.append(
                (f"{row['grouping']} = {row['level']} (n={row['n']})", row["delta"], row["ci95"])
            )
    for row in result["domains"]:
        if row["pair"] == ["T", "H-gold"] and row["delta"] is not None:
            entries.append((f"domain {row['domain']} (n={row['n']})", row["delta"], row["ci95"]))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(entries) + 1.2))
    for index, (_, delta, ci) in enumerate(entries):
        y = len(entries) - 1 - index
        ax.plot([100 * v for v in ci], [y, y], color="0.2", linewidth=1.5)
        ax.plot(100 * delta, y, "o", color="white", markeredgecolor="black", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    for edge in (-5, 5):
        ax.axvline(edge, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(entries)), [e[0] for e in entries][::-1], fontsize=7)
    ax.set_xlabel("H-gold - T, Δ pass@1 (pp), 95% Tango; descriptive, no test")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``heterogeneity.json`` and the subgroup figure into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "heterogeneity.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _figure(result, args.out / "h1-subgroups.png")
    print(json.dumps({"out": str(args.out), "unmapped": result["domain_table"]["unmapped"]}))


if __name__ == "__main__":
    main()
