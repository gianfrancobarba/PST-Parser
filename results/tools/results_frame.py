"""Base of the results analysis: exact reproduction of the frozen reports and a task table.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md``; it lives in results/ because the
freeze admits unlisted files only there. It computes no new estimate: it
checks that every later analysis starts from the frozen numbers.

What it does:
- ``reconcile``: re-runs the frozen ``analyze_paired`` on the 14 frozen designs,
  with the same ledgers, details and near-duplicate groups as the frozen
  analysis, and compares every value with the frozen ``report.json`` (absolute
  tolerance 1e-9). It then rebuilds the paired tables with the frozen
  ``paired_tables`` (H1 as the confirmatory family, parser gate report) and
  compares them cell by cell with ``paired.csv`` and ``paired-sensitivity.csv``.
- ``frame``: one row per task, condition and replicate of the correctness
  ledger, joined with task attributes (Hard, frozen covariates, parser quality,
  libraries) and cell attributes (harness details, generation tokens,
  contamination overlap), for the descriptive analyses that follow.

usage (repository root, .venv):
  python -I results/tools/results_frame.py reconcile --root . --out REPORT.json
  python -I results/tools/results_frame.py frame --root . --out FRAME.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    analyze_paired,
    read_outcomes,
)
from pstparser.empirical.tables import Table, paired_tables

VERSION: Final = "results-frame-v1"
TOLERANCE: Final = 1e-9
CONFIRMATORY: Final = {"H1": "final-bcb-core"}

TREE: Final = "outputs/ripresa/d/tree"
EVAL: Final = "outputs/ripresa/d/eval"
RUN: Final = "outputs/ripresa/e/final-01"
NEAR: Final = "results/near-duplicates/bcb/near-duplicates.json"
GATE: Final = f"{RUN}/parser/gate/gate.json"
DETAILS: Final = f"{RUN}/evaluation/bcb-details.jsonl"

# Frozen design -> the ledger the frozen analysis read (outputs/ripresa/e/e_final_analysis.py).
SOURCES: Final = {
    "bcb-h1": f"{EVAL}/check-core/outcomes.jsonl",
    "bcb-e-t": f"{EVAL}/check-core/outcomes.jsonl",
    "bcb-h-e": f"{EVAL}/check-core/outcomes.jsonl",
    "bcb-f-t": f"{EVAL}/check-core/outcomes.jsonl",
    "bcb-h-f": f"{EVAL}/check-core/outcomes.jsonl",
    "bcb-h3-s-parse": f"{EVAL}/ledgers/hauto/outcomes-s-parse.jsonl",
    "bcb-h3-end-to-end": f"{EVAL}/ledgers/hauto/outcomes-end-to-end.jsonl",
    "bcb-rq3c": f"{EVAL}/ledgers/hauto/outcomes-s-parse-all.jsonl",
    "bcb-gold-on-s-parse": f"{EVAL}/ledgers/hauto/outcomes-s-parse-all.jsonl",
    "lse-general": f"{EVAL}/codeql-final-lse-sampled-general/outcomes.jsonl",
    "lse-general-python": f"{EVAL}/ledgers/final-lse-sampled-general-python.jsonl",
    "lse-greedy-general": f"{EVAL}/codeql-final-lse-greedy-general/outcomes.jsonl",
    "lse-targeted": f"{EVAL}/codeql-final-lse-sampled-targeted/outcomes.jsonl",
    "lse-targeted-fp": f"{EVAL}/codeql-final-lse-sampled-targeted-fp/outcomes.jsonl",
}

LEDGER_FIELDS: Final = ("analysis_id", "endpoint", "task_id", "condition", "replicate_id")
RECORD_FIELDS: Final = (*LEDGER_FIELDS, "outcome", "reason")


# --- comparison -------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def compare(frozen: Any, fresh: Any, tolerance: float = TOLERANCE, path: str = "$") -> list[str]:
    """Every difference between a frozen JSON value and a recomputed one.

    Numbers agree within ``tolerance`` (absolute); booleans, strings and
    ``None`` must be equal and of the same type; keys and list lengths must match.
    """
    if isinstance(frozen, dict) and isinstance(fresh, dict):
        diffs = [f"{path}.{key}: only in the frozen value" for key in sorted(frozen.keys() - fresh)]
        diffs += [f"{path}.{key}: only in the new value" for key in sorted(fresh.keys() - frozen)]
        for key in sorted(frozen.keys() & fresh.keys()):
            diffs += compare(frozen[key], fresh[key], tolerance, f"{path}.{key}")
        return diffs
    if isinstance(frozen, list) and isinstance(fresh, list):
        if len(frozen) != len(fresh):
            return [f"{path}: length {len(frozen)} frozen, {len(fresh)} new"]
        diffs = []
        for index, (left, right) in enumerate(zip(frozen, fresh, strict=True)):
            diffs += compare(left, right, tolerance, f"{path}[{index}]")
        return diffs
    if _is_number(frozen) and _is_number(fresh):
        if not (math.isfinite(frozen) and math.isfinite(fresh)):
            return [] if frozen == fresh else [f"{path}: {frozen!r} frozen, {fresh!r} new"]
        if abs(frozen - fresh) > tolerance:
            return [f"{path}: {frozen!r} frozen, {fresh!r} new"]
        return []
    if type(frozen) is not type(fresh) or frozen != fresh:
        return [f"{path}: {frozen!r} frozen, {fresh!r} new"]
    return []


# --- reproduction of the frozen reports ---------------------------------------------


def load_design(path: Path) -> AnalysisDesign:
    """Read a design file as the frozen ``analyze`` command does."""
    payload = json.loads(path.read_text("utf-8"))
    payload["tasks"] = tuple(TaskSpec(**task) for task in payload["tasks"])
    for key in ("replicate_ids", "conditions"):
        if key in payload:
            payload[key] = tuple(payload[key])
    return AnalysisDesign(**payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def ledger_conditions(ledger: Path) -> tuple[str, ...] | None:
    """The conditions declared to ``analyze``: all of them when there are more than two."""
    conditions = sorted({row["condition"] for row in _read_jsonl(ledger)})
    return tuple(conditions) if len(conditions) > 2 else None


def reproduce_report(
    design: Path,
    ledger: Path,
    details: Path | None = None,
    near_duplicates: Path | None = None,
) -> dict[str, Any]:
    """The report of the frozen estimator, as the JSON that ``analyze`` writes."""
    report = analyze_paired(
        load_design(design),
        read_outcomes(ledger),
        ledger_conditions=ledger_conditions(ledger),
        details=_read_jsonl(details) if details is not None else None,
        near_duplicate_groups=(
            json.loads(near_duplicates.read_text("utf-8"))["groups"]
            if near_duplicates is not None
            else None
        ),
    )
    result: dict[str, Any] = json.loads(json.dumps(report.as_dict(), allow_nan=False))
    return result


def reconcile_report(
    design: Path,
    ledger: Path,
    report: Path,
    details: Path | None = None,
    near_duplicates: Path | None = None,
) -> list[str]:
    """Differences between a frozen report and its recomputation."""
    frozen = json.loads(report.read_text("utf-8"))
    return compare(frozen, reproduce_report(design, ledger, details, near_duplicates))


def _csv_rows(path: Path) -> list[list[str]]:
    text = path.read_bytes().decode("utf-8")
    body = "\n".join(line for line in text.split("\n") if not line.startswith("#"))
    return list(csv.reader(io.StringIO(body)))


def compare_table(table: Table, path: Path, tolerance: float = TOLERANCE) -> list[str]:
    """Differences between a rebuilt table and its frozen CSV file, cell by cell."""
    rows = [row for row in _csv_rows(path) if row]
    if not rows or tuple(rows[0]) != table.columns:
        return [f"{path.name}: header differs"]
    body = rows[1:]
    if len(body) != len(table.rows):
        return [f"{path.name}: {len(body)} rows frozen, {len(table.rows)} new"]
    diffs = []
    for index, (frozen_row, row) in enumerate(zip(body, table.rows, strict=True)):
        for column, text, cell in zip(table.columns, frozen_row, row, strict=True):
            where = f"{path.name} row {index} column {column}"
            if cell is None:
                same = text == ""
            elif isinstance(cell, bool | int):
                same = text == str(cell)
            elif isinstance(cell, float):
                try:
                    value = float(text)
                except ValueError:
                    same = False
                else:
                    same = (
                        abs(value - cell) <= tolerance
                        if math.isfinite(cell) and math.isfinite(value)
                        else str(value) == str(cell)
                    )
            else:
                same = text == str(cell)
            if not same:
                diffs.append(f"{where}: {text!r} frozen, {cell!r} new")
    return diffs


def reconcile(root: Path) -> dict[str, Any]:
    """Reproduce the 14 frozen reports and the two paired tables; list every difference."""
    designs = root / TREE / "protocol/analysis/designs"
    run = root / RUN
    fresh: dict[str, dict[str, Any]] = {}
    per_report: dict[str, list[str]] = {}
    for name, ledger in SOURCES.items():
        bcb = name.startswith("bcb-")
        report = reproduce_report(
            designs / name / "design.json",
            root / ledger,
            root / DETAILS if bcb else None,
            root / NEAR if bcb else None,
        )
        frozen = json.loads((run / "analysis" / name / "report.json").read_text("utf-8"))
        per_report[name] = compare(frozen, report)
        fresh[f"analysis/{name}/report.json"] = report
    gate = json.loads((root / GATE).read_text("utf-8"))
    per_table = {
        table.name: compare_table(table, run / "thesis" / f"{table.name}.csv")
        for table in paired_tables(fresh, CONFIRMATORY, gate)
    }
    identical = not any(per_report.values()) and not any(per_table.values())
    return {
        "version": VERSION,
        "tolerance": TOLERANCE,
        "reports": {name: {"identical": not d, "differences": d} for name, d in per_report.items()},
        "tables": {name: {"identical": not d, "differences": d} for name, d in per_table.items()},
        "identical": identical,
    }


# --- task frame ---------------------------------------------------------------


def task_frame(
    records: Sequence[OutcomeRecord],
    attributes: Mapping[str, Mapping[str, Any]],
    cells: Mapping[tuple[str, str, int], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per ledger cell, with the attributes of its task and of the cell.

    Raises:
        ValueError: On a duplicate cell, an attribute that would overwrite a ledger
            field, or cell attributes for a cell the ledger does not have.
    """
    frame = []
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        key = (record.task_id, record.condition, record.replicate_id)
        if key in seen:
            raise ValueError(f"duplicate cell {key}")
        seen.add(key)
        row = {name: value for name, value in asdict(record).items() if name in RECORD_FIELDS}
        for extra in (attributes.get(record.task_id, {}), (cells or {}).get(key, {})):
            clash = set(extra) & set(row)
            if clash:
                raise ValueError(f"attributes would overwrite {sorted(clash)} in {key}")
            row.update(extra)
        frame.append(row)
    orphan = set(cells or {}) - seen
    if orphan:
        raise ValueError(f"cell attributes with no outcome: {sorted(orphan)[:3]}")
    return frame


def bcb_attributes(root: Path) -> dict[str, dict[str, Any]]:
    """Task attributes of the 600 correctness tasks from frozen and collected files."""
    import pandas as pd

    prefix = json.loads(
        (root / TREE / "protocol/samples/prefix-600/bcb-prefix-600.json").read_text("utf-8")
    )
    tasks = {t["task_id"]: {"position": t["position"], "hard": t["hard"]} for t in prefix["tasks"]}
    for row in _read_jsonl(root / TREE / "protocol/covariates/bcb/covariates.jsonl"):
        tasks[row["task_id"]].update(
            {
                key: row[key]
                for key in (
                    "log_ratio_HT",
                    "log_tokens_T",
                    "n_leaves",
                    "n_segments",
                    "reorder",
                    "has_code",
                    "n_judgement_units",
                )
            }
        )
    quality = _read_jsonl(root / RUN / "exploratory/hauto-decomposition/parser-quality.jsonl")
    for row in quality:
        tasks[row["task_id"]].update(
            {
                "parser_valid": row["valid"],
                "s_parse": row["s_parse"],
                "parser_cause": row["cause"],
                "parser_f1": row["f1"],
            }
        )
    dataset = pd.read_parquet(root / "outputs/valutazione-ricerca/03/raw/full_v0.1.4.parquet")
    for task_id, libs in zip(dataset["task_id"], dataset["libs"], strict=True):
        if task_id in tasks:
            tasks[task_id]["libs"] = parse_libs(libs)
    return tasks


def parse_libs(text: str) -> list[str]:
    """The libraries of a task: the dataset stores them as a Python list literal.

    Raises:
        ValueError: If the text is not a list of strings.
    """
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"not a list literal: {text!r}") from error
    if not isinstance(value, list) or not all(isinstance(lib, str) for lib in value):
        raise ValueError(f"not a list of library names: {text!r}")
    return sorted(value)


def bcb_cells(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Cell attributes: harness details, generation tokens and contamination overlap."""
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}

    def add(key: tuple[str, str, int], values: Mapping[str, Any]) -> None:
        cell = cells.setdefault(key, {})
        if set(values) & set(cell):
            raise ValueError(f"two sources give {sorted(set(values) & set(cell))} for {key}")
        cell.update(values)

    for row in _read_jsonl(root / DETAILS):
        add(
            (row["task_id"], row["condition"], row["replicate_id"]),
            {
                "flaky": row["flaky"],
                "sanitized_lines": row["sanitized_lines"],
                "finish_reason": row["finish_reason"],
            },
        )
    for journal in ("bcb-hauto", "bcb-rest"):
        for row in _read_jsonl(root / RUN / "generation" / journal / "generations.jsonl"):
            add(
                (row["task_id"], row["condition"], row["replicate_id"]),
                {"input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]},
            )
    for row in _read_jsonl(root / RUN / "contamination/overlap.jsonl"):
        add(
            (row["task_id"], row["condition"], row["replicate_id"]),
            {"overlap": row["overlap"], "high_similarity": row["high_similarity"]},
        )
    return cells


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: ``reconcile`` or ``frame``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("reconcile", "frame"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "reconcile":
        result = reconcile(args.root)
        _write_new(args.out, json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"identical": result["identical"], "out": str(args.out)}))
        if not result["identical"]:
            raise SystemExit(1)
        return
    records = read_outcomes(args.root / EVAL / "ledgers/final-bcb.jsonl")
    frame = task_frame(records, bcb_attributes(args.root), bcb_cells(args.root))
    lines = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in frame)
    _write_new(args.out, lines)
    print(json.dumps({"rows": len(frame), "out": str(args.out)}))


if __name__ == "__main__":
    main()
