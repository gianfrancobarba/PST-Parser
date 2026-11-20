"""Decomposition of Δ_auto by parser quality on the task (plan §6, H-auto).

Written after the freeze (394c893) and before any outcome exists; it lives in
results/ because the freeze admits unlisted files only there. Exploratory only:
point estimates with 95% intervals, no p-values.

On the same tasks, Δ_auto = Δ_gold + Δ_par:
- Δ_auto = H-auto - T (end to end: a parser failure counts as fail);
- Δ_gold = H-gold - T;
- Δ_par  = H-auto - H-gold.

Every estimate is the frozen ``analyze_paired`` on the end-to-end design cut
to a level, with a ledger that joins T and H-auto of the end-to-end ledger and
H-gold of the core ledger (T must agree between the two). In every level the
three contrasts use the same tasks: those complete in all three arms, so the
identity holds exactly.

Levels of parser quality (S_gold, 600 tasks):
- ``validity``: valid output against ``invalid_output`` (frozen failure causes);
- ``strict_reconstruction``: in S_parse against a strict failure;
- ``f1_tercile_s_parse``: terciles of the per-task F1 on the S_parse tasks.

Per-task F1: the macro F1 of the gate applied to one prediction (mean over the
leaves with gold support of the leaf token F1); ``task_quality`` checks first
that the pooled F1 of the predictions equals the frozen ``score`` results.
Under 20 tasks a level reports only the counts (as ``exploratory.py``).

usage (repository root, .venv):
  python -I results/tools/hauto_decomposition.py DESIGN E2E_LEDGER CORE_LEDGER \
      --s-parse S.json --failures F.jsonl --predictions P.jsonl --id-map M.json \
      --score score-results.json --out NEW_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from pstparser.data.predictions import PredictionRecord, load_predictions
from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    analyze_paired,
    read_outcomes,
)
from pstparser.empirical.auto import macro_f1
from pstparser.evaluation.metrics.field_f1 import field_f1_scores


def _exploratory() -> ModuleType:
    path = Path(__file__).with_name("exploratory.py")
    spec = importlib.util.spec_from_file_location("exploratory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exploratory = _exploratory()

VERSION: Final = "hauto-decomposition-v1"
ARMS: Final = ("T", "H-gold", "H-auto")
CONTRASTS: Final = {
    "delta_auto": ("T", "H-auto"),
    "delta_gold": ("T", "H-gold"),
    "delta_par": ("H-gold", "H-auto"),
}
TOLERANCE: Final = 1e-9


def _f1_by_field(rows: Sequence[PredictionRecord]) -> dict[str, dict[str, Any]]:
    scores = field_f1_scores([r.prediction for r in rows], [r.target for r in rows])
    return {name: value.as_dict() for name, value in scores.items()}


def task_quality(
    predictions: Sequence[PredictionRecord],
    tasks: Sequence[str],
    failures: Mapping[str, str],
    s_parse: set[str],
    score: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Parser quality of every task: validity, S_parse, failure cause, F1.

    Raises:
        ValueError: If the predictions do not reproduce the frozen score, or the
            index map does not cover them.
    """
    if sorted(r.index for r in predictions) != list(range(len(tasks))):
        raise ValueError("the index map does not cover the predictions")
    pooled = _f1_by_field(predictions)
    frozen = score["field_f1"]
    if set(pooled) != set(frozen) or any(
        (pooled[name]["f1"] is None) != (frozen[name]["f1"] is None)
        or (
            pooled[name]["f1"] is not None
            and abs(pooled[name]["f1"] - frozen[name]["f1"]) > TOLERANCE
        )
        for name in pooled
    ):
        raise ValueError("the predictions do not reproduce the frozen score results")
    quality = {}
    for record in predictions:
        task = tasks[record.index]
        try:
            json.loads(record.prediction)
            json.loads(record.target)
        except json.JSONDecodeError:
            f1 = None
        else:
            f1, _ = macro_f1(_f1_by_field([record]))
        cause = failures.get(task)
        quality[task] = {
            "task_id": task,
            "valid": cause != "invalid_output",
            "s_parse": task in s_parse,
            "cause": cause,
            "f1": f1,
        }
    return quality


def combine(
    end_to_end: Sequence[OutcomeRecord], core: Sequence[OutcomeRecord], analysis_id: str
) -> list[OutcomeRecord]:
    """T and H-auto of the end-to-end ledger with H-gold of the core ledger.

    Raises:
        ValueError: If T differs between the two ledgers.
    """

    def cells(records: Sequence[OutcomeRecord]) -> dict[tuple[str, int], str]:
        return {(r.task_id, r.replicate_id): r.outcome for r in records if r.condition == "T"}

    if cells(end_to_end) != cells(core):
        raise ValueError("T differs between the end-to-end and the core ledger")
    kept = [r for r in end_to_end if r.condition in ("T", "H-auto")]
    kept += [r for r in core if r.condition == "H-gold"]
    return [replace(r, analysis_id=analysis_id) for r in kept]


def _levels(quality: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    f1 = {t: q["f1"] for t, q in quality.items() if q["s_parse"] and q["f1"] is not None}
    return {
        "all": {t: "all" for t in quality},
        "validity": {t: "valid" if q["valid"] else "invalid" for t, q in quality.items()},
        "strict_reconstruction": {
            t: "s_parse" if q["s_parse"] else "failed" for t, q in quality.items()
        },
        "f1_tercile_s_parse": exploratory._terciles(f1) if f1 else {},
    }


def decompose(
    design: AnalysisDesign,
    records: Sequence[OutcomeRecord],
    quality: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The three contrasts inside every level, on the tasks complete in all arms.

    Raises:
        ValueError: If the identity Δ_auto = Δ_gold + Δ_par fails in a level.
    """
    replicates = set(design.replicate_ids)
    present = {
        (r.task_id, r.condition)
        for r in records
        if r.replicate_id in replicates and r.condition in ARMS and r.outcome != "missing"
    }
    complete = {t.task_id for t in design.tasks if all((t.task_id, arm) in present for arm in ARMS)}
    ledger = [replace(r, analysis_id=design.analysis_id) for r in records]
    report: dict[str, Any] = {}
    for name, levels in _levels(quality).items():
        rows = []
        for level in sorted(set(levels.values())):
            members = sorted(t for t, value in levels.items() if value == level)
            kept = [t for t in members if t in complete]
            small = len(kept) < exploratory.MIN_TASKS
            contrasts = {}
            inside = [r for r in ledger if r.task_id in set(kept)]
            for key, pair in CONTRASTS.items():
                cut = replace(
                    design,
                    tasks=tuple(t for t in design.tasks if t.task_id in set(kept)),
                    conditions=pair,
                )
                result = analyze_paired(cut, inside, ledger_conditions=ARMS) if kept else None
                contrasts[key] = {
                    "conditions": list(pair),
                    "included_tasks": 0 if result is None else result.included_tasks,
                    "counts": None if result is None else exploratory._direction_counts(result),
                    "delta": None if small or result is None else result.delta,
                    "ci95": None if small or result is None else exploratory._interval(result),
                }
            deltas = [contrasts[key]["delta"] for key in CONTRASTS]
            if all(d is not None for d in deltas):
                residual = deltas[0] - deltas[1] - deltas[2]
                if abs(residual) > TOLERANCE:
                    raise ValueError(f"identity fails in {name}={level}: {residual}")
            rows.append(
                {
                    "level": level,
                    "tasks": len(kept),
                    "incomplete_tasks": sorted(set(members) - set(kept)),
                    "reported": "counts only" if small else "estimate",
                    "f1_range": _range(quality, kept) if name == "f1_tercile_s_parse" else None,
                    "contrasts": contrasts,
                }
            )
        report[name] = {"left_out_tasks": len(design.tasks) - len(levels), "levels": rows}
    return report


def _range(quality: Mapping[str, Mapping[str, Any]], tasks: Sequence[str]) -> list[float] | None:
    values = [quality[t]["f1"] for t in tasks]
    return [min(values), max(values)] if values else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def run(args: argparse.Namespace) -> None:
    """Read the inputs, decompose and write the two reports into a new directory."""
    if args.out.exists():
        raise FileExistsError(args.out)
    design = exploratory.load_design(args.design)
    tasks = json.loads(args.id_map.read_text("utf-8"))["tasks"]
    s_parse = set(json.loads(args.s_parse.read_text("utf-8"))["tasks"])
    failures = {row["task_id"]: row["cause"] for row in _read_jsonl(args.failures)}
    quality = task_quality(
        load_predictions(args.predictions),
        tasks,
        failures,
        s_parse,
        json.loads(args.score.read_text("utf-8")),
    )
    if set(quality) != {t.task_id for t in design.tasks}:
        raise ValueError("the predictions do not cover the tasks of the design")
    records = combine(read_outcomes(args.end_to_end), read_outcomes(args.core), design.analysis_id)
    report = decompose(design, records, quality)
    args.out.mkdir(parents=True)
    with (args.out / "decomposition.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "version": VERSION,
                "analysis_id": design.analysis_id,
                "contrasts": CONTRASTS,
                **report,
            },
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    with (args.out / "parser-quality.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for task in sorted(quality):
            value = quality[task]
            if value["f1"] is not None and not math.isfinite(value["f1"]):
                raise ValueError(f"F1 not finite for {task}")
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "tasks": len(quality)}))


def main(argv: Sequence[str] | None = None) -> None:
    """Command line of the decomposition."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("design", type=Path)
    parser.add_argument("end_to_end", type=Path)
    parser.add_argument("core", type=Path)
    for option in ("s-parse", "failures", "predictions", "id-map", "score", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
