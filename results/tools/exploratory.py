"""Exploratory analyses of plan §6 that the frozen package has no command for.

Written after the freeze (394c893) and before any outcome exists; it lives in
results/ because the freeze admits unlisted files only there. Every estimate is
the frozen ``analyze_paired`` on a restricted design, so subgroups use the same
method, seed and intervals as the primary analyses. Nothing here is
confirmatory: point estimates with 95% intervals, no p-values in the tables.

What it computes (plan §6, the PST package and the subgroups):
- ``restrict``: a design and its ledger cut to a subset of tasks under the
  analysis_id of the design (the 83 Python-origin prompts of LLMSecEval).
- ``subgroups``: non-empty leaves 1/2/>=3, judgement units 1/2/>=3, code in the
  prompt, CWE (families), the Hard tasks, terciles of log(H/T) tokens,
  reorder 0 against >0. Under 20 tasks, or 8 families, only the counts.
- ``logistic``: on the discordant pairs of a greedy design, P(treatment wins)
  on log(H/T) tokens, single leaf, reorder > 0; one-covariate models, and the
  joint model only if every VIF <= 5; no model under 40 discordant pairs.
- ``taxonomy``: failure categories per condition and the baseline -> treatment
  transitions among discordant pairs.
- ``qualitative``: all discordant pairs if 40 or fewer, otherwise 20 + 20 (one
  set per direction) drawn with seed 7319, with the arms relabelled A/B at
  random so the reviewer is blind to the condition; the key is a separate file.

usage (repository root, .venv):
  python -I results/tools/exploratory.py restrict DESIGN LEDGER OUT.jsonl
  python -I results/tools/exploratory.py run DESIGN LEDGER --covariates C.jsonl \
      [--ledger-conditions T,H-gold,E] [--hard prefix-600.json] \
      [--generations G.jsonl] --out NEW_DIR
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    PairedReport,
    TaskSpec,
    analyze_paired,
    read_outcomes,
    write_outcomes,
)

EXPLORATORY_VERSION: Final = "exploratory-tools-v1"
MIN_TASKS: Final = 20
MIN_FAMILIES: Final = 8
MIN_PAIRS: Final = 40
MAX_VIF: Final = 5.0
QUALITATIVE_SEED: Final = 7319
QUALITATIVE_PER_DIRECTION: Final = 20
MODEL_COVARIATES: Final = ("log_ratio_HT", "single_leaf", "reordered")


def load_design(path: Path) -> AnalysisDesign:
    """Read a design file as ``analyze`` does."""
    payload = json.loads(path.read_text("utf-8"))
    payload["tasks"] = tuple(TaskSpec(**task) for task in payload["tasks"])
    for key in ("replicate_ids", "conditions"):
        if key in payload:
            payload[key] = tuple(payload[key])
    return AnalysisDesign(**payload)


def restrict(
    design: AnalysisDesign, records: Sequence[OutcomeRecord], tasks: Sequence[str] | None = None
) -> tuple[AnalysisDesign, list[OutcomeRecord]]:
    """Cut a design to ``tasks`` (default: its own) and the ledger to that design.

    The kept records take the analysis_id of the design, so a ledger of the full
    population serves a frozen design of a subset (the 83 Python prompts).

    Raises:
        ValueError: If a task is not in the design, or the ledger has another endpoint.
    """
    wanted = set(tasks) if tasks is not None else {task.task_id for task in design.tasks}
    known = {task.task_id for task in design.tasks}
    if not wanted <= known:
        raise ValueError(f"tasks outside the design: {sorted(wanted - known)[:3]}")
    if any(record.endpoint != design.endpoint for record in records):
        raise ValueError("the ledger has another endpoint than the design")
    if len({record.analysis_id for record in records}) > 1:
        raise ValueError("the ledger mixes analysis ids")
    cut = replace(design, tasks=tuple(t for t in design.tasks if t.task_id in wanted))
    kept = [
        replace(record, analysis_id=design.analysis_id)
        for record in records
        if record.task_id in wanted
    ]
    return cut, kept


def read_covariates(path: Path) -> dict[str, dict[str, Any]]:
    """Covariates by task, with the derived flags the models use."""
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_bytes().decode("utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            row["single_leaf"] = int(row["n_leaves"] == 1)
            row["reordered"] = int(row["reorder"] > 0)
            rows[row["task_id"]] = row
    return rows


def _level(value: int) -> str:
    return str(value) if value < 3 else ">=3"


def _cwe(family: str | None) -> str | None:
    return None if family is None else "CWE-" + family.split("-", 1)[0]


def _terciles(values: Mapping[str, float]) -> dict[str, str]:
    low, high = np.quantile(np.array(list(values.values()), dtype=float), [1 / 3, 2 / 3])
    return {
        task: "T1" if value <= low else "T2" if value <= high else "T3"
        for task, value in values.items()
    }


def groupings(
    design: AnalysisDesign,
    covariates: Mapping[str, Mapping[str, Any]],
    hard: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Subgroup level of every task of the design, per grouping.

    A task without a prespecified level (0 judgement units) is left out of that
    grouping and counted in the report.
    """
    ids = [task.task_id for task in design.tasks]
    missing = [task for task in ids if task not in covariates]
    if missing:
        raise ValueError(f"tasks without covariates: {missing[:3]}")
    found: dict[str, dict[str, str]] = {
        "leaves": {t: _level(covariates[t]["n_leaves"]) for t in ids},
        "judgement_units": {
            t: _level(covariates[t]["n_judgement_units"])
            for t in ids
            if covariates[t]["n_judgement_units"] > 0
        },
        "has_code": {t: str(bool(covariates[t]["has_code"])).lower() for t in ids},
        "log_ratio_tercile": _terciles({t: covariates[t]["log_ratio_HT"] for t in ids}),
        "reorder": {t: "r>0" if covariates[t]["reorder"] > 0 else "r=0" for t in ids},
    }
    if design.tasks[0].family_id is not None:
        found["cwe"] = {task.task_id: str(_cwe(task.family_id)) for task in design.tasks}
    if hard is not None:
        found["hard"] = {t: str(t in hard).lower() for t in ids}
    return found


def _interval(report: PairedReport) -> tuple[float, float] | None:
    for interval in (report.tango_ci, report.cr1_ci, report.bootstrap_ci):
        if interval is not None:
            return interval
    return None


def _direction_counts(report: PairedReport) -> dict[str, int]:
    if report.mcnemar_table is not None:
        return dict(report.mcnemar_table)
    return {
        "treatment_higher": sum(1 for e in report.task_estimates if e.difference > 0),
        "baseline_higher": sum(1 for e in report.task_estimates if e.difference < 0),
        "equal": sum(1 for e in report.task_estimates if e.difference == 0),
    }


def subgroups(
    design: AnalysisDesign,
    records: Sequence[OutcomeRecord],
    covariates: Mapping[str, Mapping[str, Any]],
    *,
    hard: set[str] | None = None,
    ledger_conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The frozen paired analysis inside every subgroup level."""
    report: dict[str, Any] = {}
    for name, levels in groupings(design, covariates, hard).items():
        rows = []
        for level in sorted(set(levels.values())):
            members = sorted(task for task, value in levels.items() if value == level)
            cut, kept = restrict(design, records, members)
            result = analyze_paired(cut, kept, ledger_conditions=ledger_conditions)
            families = len({t.family_id for t in cut.tasks if t.family_id is not None})
            small = result.included_tasks < MIN_TASKS or (0 < families < MIN_FAMILIES)
            rows.append(
                {
                    "level": level,
                    "planned_tasks": result.planned_tasks,
                    "included_tasks": result.included_tasks,
                    "families": families or None,
                    "counts": _direction_counts(result),
                    "reported": "counts only" if small else "estimate",
                    "delta": None if small else result.delta,
                    "ci95": None if small else _interval(result),
                }
            )
        report[name] = {
            "left_out_tasks": len(design.tasks) - len(levels),
            "levels": rows,
        }
    return report


def discordant_pairs(
    design: AnalysisDesign,
    records: Sequence[OutcomeRecord],
) -> list[dict[str, Any]]:
    """Discordant pairs of a design with one functional replicate.

    Tasks with a missing arm are left out, as in the primary analysis.

    Raises:
        ValueError: If the design is not one greedy functional replicate.
    """
    if design.endpoint != "functional" or len(design.replicate_ids) != 1:
        raise ValueError("discordant pairs need one functional replicate per arm")
    baseline, treatment = design.conditions
    replicate = design.replicate_ids[0]
    cells = {
        (r.task_id, r.condition): r
        for r in records
        if r.replicate_id == replicate and r.condition in design.conditions
    }
    pairs = []
    for task in design.tasks:
        left = cells.get((task.task_id, baseline))
        right = cells.get((task.task_id, treatment))
        if left is None or right is None or "missing" in (left.outcome, right.outcome):
            continue
        if left.outcome != right.outcome:
            pairs.append(
                {
                    "task_id": task.task_id,
                    "winner": treatment if right.outcome == "pass" else baseline,
                    "baseline_reason": left.reason,
                    "treatment_reason": right.reason,
                }
            )
    return pairs


def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Logistic regression by Newton steps, with Wald intervals."""
    beta = np.zeros(x.shape[1])
    converged = False
    for _ in range(100):
        p = 1 / (1 + np.exp(-(x @ beta)))
        hessian = x.T @ (x * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian, x.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    p = 1 / (1 + np.exp(-(x @ beta)))
    hessian = x.T @ (x * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    separated = not converged or bool(np.any(np.abs(beta) > 15))
    return {"beta": beta, "se": se, "converged": converged and not separated}


def _vif(columns: np.ndarray) -> list[float]:
    values = []
    for j in range(columns.shape[1]):
        others = np.column_stack([np.ones(len(columns)), np.delete(columns, j, axis=1)])
        target = columns[:, j]
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        residual = target - others @ coef
        total = float(np.sum((target - target.mean()) ** 2))
        r2 = 1 - float(np.sum(residual**2)) / total if total > 0 else 1.0
        values.append(math.inf if r2 >= 1 else 1 / (1 - r2))
    return values


def _model(
    pairs: Sequence[Mapping[str, Any]],
    covariates: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    treatment: str,
) -> dict[str, Any]:
    y = np.array([float(pair["winner"] == treatment) for pair in pairs])
    columns = np.array(
        [[float(covariates[pair["task_id"]][name]) for name in names] for pair in pairs]
    )
    constant = [name for j, name in enumerate(names) if np.ptp(columns[:, j]) == 0]
    if constant or y.min() == y.max():
        return {
            "covariates": list(names),
            "read": False,
            "reason": "no variation",
            "constant": constant,
        }
    fit = _fit(np.column_stack([np.ones(len(y)), columns]), y)
    terms = {}
    for j, name in enumerate(names, start=1):
        beta, se = float(fit["beta"][j]), float(fit["se"][j])
        terms[name] = {
            "log_odds": beta,
            "se": se,
            "odds_ratio": math.exp(beta),
            "or_ci95": (
                math.exp(beta - 1.959963984540054 * se),
                math.exp(beta + 1.959963984540054 * se),
            ),
        }
    return {
        "covariates": list(names),
        "read": bool(fit["converged"]),
        "reason": None if fit["converged"] else "no convergence or separation",
        "terms": terms,
    }


def logistic(
    design: AnalysisDesign,
    pairs: Sequence[Mapping[str, Any]],
    covariates: Mapping[str, Mapping[str, Any]],
    names: Sequence[str] = MODEL_COVARIATES,
) -> dict[str, Any]:
    """P(treatment wins) among discordant pairs, by the PST-package covariates."""
    treatment = design.conditions[1]
    result: dict[str, Any] = {"pairs": len(pairs), "outcome": f"{treatment} wins"}
    if len(pairs) < MIN_PAIRS:
        result["read"] = False
        result["reason"] = f"fewer than {MIN_PAIRS} discordant pairs"
        return result
    result["one_covariate"] = [_model(pairs, covariates, [name], treatment) for name in names]
    columns = np.array([[float(covariates[p["task_id"]][n]) for n in names] for p in pairs])
    # A covariate constant among the pairs cannot be estimated: it is named and
    # left out of the joint model, which needs at least two that vary.
    varying = [n for j, n in enumerate(names) if np.ptp(columns[:, j]) > 0]
    result["constant_covariates"] = [n for n in names if n not in varying]
    if len(varying) < 2:
        result["joint"] = {"read": False, "reason": "fewer than two varying covariates"}
        return result
    vif = _vif(columns[:, [names.index(n) for n in varying]])
    result["vif"] = dict(zip(varying, vif, strict=True))
    if max(vif) > MAX_VIF:
        result["vif"] = {n: (v if math.isfinite(v) else None) for n, v in result["vif"].items()}
        result["joint"] = {"read": False, "reason": f"a VIF above {MAX_VIF}"}
    else:
        result["joint"] = _model(pairs, covariates, varying, treatment)
    return result


def taxonomy(
    design: AnalysisDesign,
    records: Sequence[OutcomeRecord],
    pairs: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Outcome categories per condition, and transitions among discordant pairs."""
    tasks = {task.task_id for task in design.tasks}
    per_condition: dict[str, Counter[str]] = {}
    for record in records:
        if record.task_id in tasks and record.replicate_id in design.replicate_ids:
            label = record.reason or record.outcome
            per_condition.setdefault(record.condition, Counter())[label] += 1
    result: dict[str, Any] = {
        "per_condition": {c: dict(sorted(n.items())) for c, n in sorted(per_condition.items())}
    }
    if pairs is not None:
        moves = Counter(f"{p['baseline_reason']} -> {p['treatment_reason']}" for p in pairs)
        result["transitions"] = dict(sorted(moves.items()))
    return result


def qualitative(
    design: AnalysisDesign,
    pairs: Sequence[Mapping[str, Any]],
    seed: int = QUALITATIVE_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Blind sample of discordant pairs, and its key.

    All pairs if there are 40 or fewer, otherwise 20 drawn from each direction
    (all of a direction with fewer than 20). The sample is shuffled and each
    pair gets the arms as A and B in random order.
    """
    rng = random.Random(seed)
    ordered = sorted(pairs, key=lambda pair: pair["task_id"])
    if len(ordered) <= 2 * QUALITATIVE_PER_DIRECTION:
        chosen = list(ordered)
    else:
        chosen = []
        for side in design.conditions:
            pool = [pair for pair in ordered if pair["winner"] == side]
            chosen += rng.sample(pool, min(QUALITATIVE_PER_DIRECTION, len(pool)))
    rng.shuffle(chosen)
    sample, key = [], []
    for rank, pair in enumerate(chosen, start=1):
        arms = list(design.conditions)
        rng.shuffle(arms)
        sample.append({"item": rank, "task_id": pair["task_id"]})
        key.append({"item": rank, "task_id": pair["task_id"], "A": arms[0], "B": arms[1]})
    return sample, key


def _attach(
    sample: list[dict[str, Any]],
    key: Sequence[Mapping[str, Any]],
    generations: Path,
    replicate: int,
) -> None:
    texts = {}
    for line in generations.read_bytes().decode("utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            if row["replicate_id"] == replicate:
                texts[(row["task_id"], row["condition"])] = row["raw_output"]
    for item, entry in zip(sample, key, strict=True):
        item["A"] = texts[(entry["task_id"], entry["A"])]
        item["B"] = texts[(entry["task_id"], entry["B"])]


def _write(path: Path, document: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def run(
    design_path: Path,
    ledger: Path,
    covariates_path: Path,
    out: Path,
    *,
    ledger_conditions: Sequence[str] | None = None,
    hard_path: Path | None = None,
    generations: Path | None = None,
) -> list[str]:
    """Write every exploratory report of one design into a new directory."""
    out.mkdir(parents=True)
    design, records = restrict(load_design(design_path), read_outcomes(ledger))
    covariates = read_covariates(covariates_path)
    hard = None
    if hard_path is not None:
        tasks = json.loads(hard_path.read_text("utf-8"))["tasks"]
        hard = {task["task_id"] for task in tasks if task.get("hard")}
    written = []
    header = {"version": EXPLORATORY_VERSION, "analysis_id": design.analysis_id}
    _write(
        out / "subgroups.json",
        {
            **header,
            **subgroups(
                design, records, covariates, hard=hard, ledger_conditions=ledger_conditions
            ),
        },
    )
    written.append("subgroups.json")
    greedy = design.endpoint == "functional" and len(design.replicate_ids) == 1
    pairs = discordant_pairs(design, records) if greedy else None
    _write(out / "taxonomy.json", {**header, **taxonomy(design, records, pairs)})
    written.append("taxonomy.json")
    if pairs is not None:
        _write(out / "logistic.json", {**header, **logistic(design, pairs, covariates)})
        sample, key = qualitative(design, pairs)
        if generations is not None:
            _attach(sample, key, generations, design.replicate_ids[0])
        _write(
            out / "qualitative-sample.json", {**header, "seed": QUALITATIVE_SEED, "items": sample}
        )
        _write(out / "qualitative-key.json", {**header, "seed": QUALITATIVE_SEED, "items": key})
        written += ["logistic.json", "qualitative-sample.json", "qualitative-key.json"]
    return written


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: restrict a ledger, or run the exploratory reports of a design."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cut = commands.add_parser("restrict")
    cut.add_argument("design", type=Path)
    cut.add_argument("ledger", type=Path)
    cut.add_argument("out", type=Path)
    every = commands.add_parser("run")
    every.add_argument("design", type=Path)
    every.add_argument("ledger", type=Path)
    every.add_argument("--covariates", type=Path, required=True)
    every.add_argument("--ledger-conditions")
    every.add_argument("--hard", type=Path)
    every.add_argument("--generations", type=Path)
    every.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "restrict":
        if args.out.exists():
            raise FileExistsError(args.out)
        design, records = restrict(load_design(args.design), read_outcomes(args.ledger))
        write_outcomes(records, args.out)
        print(json.dumps({"analysis_id": design.analysis_id, "records": len(records)}))
    else:
        conditions = args.ledger_conditions.split(",") if args.ledger_conditions else None
        names = run(
            args.design,
            args.ledger,
            args.covariates,
            args.out,
            ledger_conditions=conditions,
            hard_path=args.hard,
            generations=args.generations,
        )
        print(json.dumps({"out": str(args.out), "files": names}))


if __name__ == "__main__":
    main()
