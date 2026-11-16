"""Thesis tables regenerated from versioned run files, never copied by hand (M20).

Every table is built from explicit input files and written as CSV and LaTeX.
Each file opens with comment lines that name the run, the SHA-256 of every
input and the digest of the package source, so a number in the thesis can be
traced to the files it comes from. No file holds a time or an output path:
the same inputs and the same source give the same bytes.

The tables follow the output map of the operating plan (T-M1, T-M2, T-M3,
T-M9, T-R1, T-R2, T-R3, T-R4, T-R6, T-R8, T-X1). The table of the blind
security review (T-R7) is not built here: the review has no file format yet.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pstparser.empirical.analysis import OutcomeRecord
from pstparser.empirical.contracts import CONDITIONS
from pstparser.empirical.inference import CONFIRMATORY_WEIGHTS, weighted_holm
from pstparser.pst.taxonomy import LEAF_PATHS, split_path

TABLES_VERSION: Final = "thesis-tables-v1"
#: Table kinds of the ``tables`` command, in the order of the output map.
KINDS: Final = (
    "samples",
    "treatment",
    "tokens",
    "freeze",
    "diagnostics",
    "paired",
    "security",
    "parser",
)
SECURITY_STATES: Final = ("V", "N", "I", "U", "missing")

Cell = str | int | float | None


@dataclass(frozen=True)
class Table:
    """A named table: ordered columns and rows of plain cells."""

    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]

    def __post_init__(self) -> None:
        """Refuse a row whose width differs from the header."""
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError(f"{self.name}: every row needs {len(self.columns)} cells")


@dataclass(frozen=True)
class Provenance:
    """What every table file declares: run, inputs and package source."""

    run_id: str
    source_digest: str
    inputs: tuple[tuple[str, str], ...]

    def lines(self, kind: str, table: str) -> list[str]:
        """Header lines, without comment markers."""
        return [
            f"{TABLES_VERSION} kind={kind} table={table}",
            f"run_id={self.run_id}",
            f"source_sha256={self.source_digest}",
            *(f"input {name} sha256={digest}" for name, digest in self.inputs),
        ]


def ordered_conditions(conditions: set[str]) -> list[str]:
    """Conditions in registry order (T, H-gold, E, H-auto, F, H-explained)."""
    return sorted(conditions, key=lambda name: (CONDITIONS.index(name), name))


def _float(value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError("table values must be finite")
    return value


def _bounds(interval: Sequence[float] | None) -> tuple[float | None, float | None]:
    return (None, None) if interval is None else (_float(interval[0]), _float(interval[1]))


# --- builders ----------------------------------------------------------------


def samples_tables(
    samples: Mapping[str, Mapping[str, Any]], exclusions: Mapping[str, Any] | None
) -> tuple[Table, ...]:
    """T-M1: the samples and, from the frozen exclusions, each excluded task with its rules."""
    rows = []
    for name in sorted(samples):
        document = samples[name]
        tasks = document.get("tasks") or document.get("records") or []
        families = {task["family_id"] for task in tasks if task.get("family_id")}
        rows.append(
            (
                name,
                document.get("benchmark"),
                document.get("kind"),
                len(tasks),
                document.get("hard_count"),
                len(families) if families else None,
                document.get("seed"),
            )
        )
    tables = [
        Table(
            "samples",
            ("sample", "benchmark", "kind", "tasks", "hard", "families", "seed"),
            tuple(rows),
        )
    ]
    if exclusions is not None:
        rules = exclusions["rules"]
        tables.append(
            Table(
                "exclusions",
                ("task_id", "hard", "rules", "reasons"),
                tuple(
                    (
                        entry["task_id"],
                        "yes" if entry["hard"] else "no",
                        " ".join(entry["rules"]),
                        " ".join(rules[rule] for rule in entry["rules"]),
                    )
                    for entry in exclusions["excluded"]
                ),
            )
        )
        tables.append(
            Table(
                "eligibility",
                (
                    "tasks",
                    "excluded",
                    "eligible",
                    "eligible_hard",
                    "calibrations",
                    "reference_runs",
                ),
                (
                    (
                        exclusions["tasks"],
                        exclusions["excluded_count"],
                        exclusions["eligible"],
                        exclusions["eligible_hard"],
                        exclusions["calibrations"],
                        exclusions["reference_runs"],
                    ),
                ),
            )
        )
    return tuple(tables)


def leaf_count(hierarchical: str) -> int:
    """Number of non-empty leaves of a rendered PST (H-gold, H-auto)."""
    tree = json.loads(hierarchical)["prompt"]
    count = 0
    for path in LEAF_PATHS:
        node: Any = tree
        for part in split_path(path):
            node = node[part]
        count += bool(node)
    return count


def treatment_tables(variants: Sequence[Mapping[str, Any]]) -> tuple[Table, ...]:
    """T-M2: non-empty leaves per prompt and the share of single-leaf prompts."""
    counts: dict[str, list[int]] = {}
    for row in variants:
        if row["renderer_version"] in {"pst-gold-json-v1"} and row["condition"] != "T":
            counts.setdefault(row["condition"], []).append(leaf_count(row["text"]))
    if not counts:
        raise ValueError("no hierarchical variant to describe")
    distribution = []
    summary = []
    for condition in ordered_conditions(set(counts)):
        values = counts[condition]
        for leaves, tasks in sorted(Counter(values).items()):
            distribution.append((condition, leaves, tasks, tasks / len(values)))
        summary.append(
            (
                condition,
                len(values),
                statistics.fmean(values),
                statistics.median(values),
                sum(1 for value in values if value == 1) / len(values),
                sum(1 for value in values if value >= 3) / len(values),
            )
        )
    return (
        Table("treatment-leaves", ("condition", "leaves", "prompts", "share"), tuple(distribution)),
        Table(
            "treatment",
            (
                "condition",
                "prompts",
                "mean_leaves",
                "median_leaves",
                "single_leaf_share",
                "three_or_more_share",
            ),
            tuple(summary),
        ),
    )


def input_tokens(journal: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    """Input tokens per (task, condition); replicates must agree."""
    tokens: dict[tuple[str, str], int] = {}
    for row in journal:
        key = (row["task_id"], row["condition"])
        if tokens.setdefault(key, row["input_tokens"]) != row["input_tokens"]:
            raise ValueError(f"replicates of {key} disagree on input tokens")
    return tokens


def tokens_tables(journal: Sequence[Mapping[str, Any]], baseline: str = "T") -> tuple[Table, ...]:
    """T-M3: input tokens per condition and, per task, the ratio to the baseline."""
    tokens = input_tokens(journal)
    conditions = ordered_conditions({condition for _, condition in tokens})
    if baseline not in conditions:
        raise ValueError(f"the journal holds no {baseline} rows")
    tasks = sorted({task for task, _ in tokens})
    rows = []
    for condition in conditions:
        values = [tokens[task, condition] for task in tasks if (task, condition) in tokens]
        ratios = [
            tokens[task, condition] / tokens[task, baseline]
            for task in tasks
            if (task, condition) in tokens and (task, baseline) in tokens
        ]
        rows.append(
            (
                condition,
                len(values),
                statistics.fmean(values),
                statistics.median(values),
                min(values),
                max(values),
                statistics.fmean(ratios) if ratios else None,
                statistics.median(ratios) if ratios else None,
            )
        )
    return (
        Table(
            "tokens",
            (
                "condition",
                "prompts",
                "mean_input_tokens",
                "median_input_tokens",
                "min_input_tokens",
                "max_input_tokens",
                f"mean_ratio_to_{baseline}",
                f"median_ratio_to_{baseline}",
            ),
            tuple(rows),
        ),
    )


def freeze_tables(freeze: Mapping[str, Any]) -> tuple[Table, ...]:
    """T-M9: every frozen item with its SHA-256, then code, lock and source."""
    rows: list[tuple[Cell, ...]] = [
        ("item", name, digest) for name, digest in freeze["items"].items()
    ]
    rows += [
        ("code", freeze["code"]["directory"], freeze["code"]["sha256"]),
        ("environment", "uv.lock", freeze["environment"]["uv_lock_sha256"]),
        ("source", "snapshot", freeze["source"]["snapshot_digest"]),
    ]
    meta = (
        ("format", freeze["format"]),
        ("protocol_version", freeze["protocol_version"]),
        ("frozen_at", freeze["frozen_at"]),
        ("frozen_by", freeze["frozen_by"]),
    )
    return (
        Table("freeze", ("kind", "path", "sha256"), tuple(rows)),
        Table("freeze-meta", ("key", "value"), meta),
    )


def diagnostics_tables(
    journal: Sequence[Mapping[str, Any]], details: Sequence[Mapping[str, Any]] | None
) -> tuple[Table, ...]:
    """T-R1: finish reasons, response length, sessions and, with details, extraction.

    ``details`` is the file of one check run of this journal (functional or
    security); it must cover the journal's requests exactly.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in journal:
        groups.setdefault(row["condition"], []).append(row)
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    if details is not None:
        if Counter(d["request_id"] for d in details) != Counter(r["request_id"] for r in journal):
            raise ValueError("the details do not cover the journal exactly")
        for detail in details:
            by_condition.setdefault(detail["condition"], []).append(detail)
    rows = []
    for condition in ordered_conditions(set(groups)):
        members = groups[condition]
        output = [row["output_tokens"] for row in members]
        sessions = {row["session_id"] for row in members if row.get("session_id")}
        gpus = sorted({row["gpu_name"] for row in members if row.get("gpu_name")})
        extraction: Cell = None
        prompt_as_data: Cell = None
        json_response: Cell = None
        if details is not None:
            checked = by_condition.get(condition, [])
            if checked and "sanitized_lines" in checked[0]:
                extraction = sum(1 for d in checked if d["sanitized_lines"] == 0)
                prompt_as_data = sum(1 for d in checked if d.get("reason") == "prompt_as_data")
                json_response = sum(1 for d in checked if d.get("reason") == "json_response")
            else:
                extraction = sum(1 for d in checked if d["extraction_status"] != "ok")
        rows.append(
            (
                condition,
                len(members),
                sum(1 for row in members if row["finish_reason"] == "length"),
                statistics.fmean(output),
                statistics.median(output),
                max(output),
                len(sessions),
                " ".join(gpus) if gpus else None,
                extraction,
                json_response,
                prompt_as_data,
            )
        )
    return (
        Table(
            "diagnostics",
            (
                "condition",
                "requests",
                "finish_length",
                "mean_output_tokens",
                "median_output_tokens",
                "max_output_tokens",
                "sessions",
                "gpus",
                "extraction_failures",
                "json_response",
                "prompt_as_data",
            ),
            tuple(rows),
        ),
    )


def _primary_p(report: Mapping[str, Any]) -> float | None:
    """The primary p-value: mid-p for one greedy pair per task, else the sign-flip."""
    value = report.get("midp_p")
    return value if value is not None else report.get("signflip_p")


def paired_tables(
    reports: Mapping[str, Mapping[str, Any]],
    confirmatory: Mapping[str, str] | None = None,
) -> tuple[Table, ...]:
    """T-R2, T-R3, T-R5, T-X1: paired estimates, tests, intervals and sensitivity.

    ``confirmatory`` maps H1 (and H3, if it stayed in the family) to the
    ``analysis_id`` of its report; their primary p-values are decided with the
    weighted Holm procedure (0.04 / 0.01, with recycling). Without H3 the
    family is H1 alone, read at 0.05 (failed parser gate).
    """
    by_id = {report["analysis_id"]: report for report in reports.values()}
    if len(by_id) != len(reports):
        raise ValueError("two reports share an analysis_id")
    decisions: dict[str, str] = {}
    if confirmatory:
        if set(confirmatory) - set(CONFIRMATORY_WEIGHTS) or "H1" not in confirmatory:
            raise ValueError("the confirmatory family is H1 and, optionally, H3")
        p_values: dict[str, float | None] = {}
        for hypothesis in CONFIRMATORY_WEIGHTS:
            identifier = confirmatory.get(hypothesis)
            if identifier is None:
                p_values[hypothesis] = None
                continue
            if identifier not in by_id:
                raise ValueError(f"{hypothesis}: no report with analysis_id {identifier!r}")
            p_values[hypothesis] = _primary_p(by_id[identifier])
            if p_values[hypothesis] is None:
                raise ValueError(f"{hypothesis}: the report has no primary p-value")
        result = weighted_holm(p_values, CONFIRMATORY_WEIGHTS)
        decisions = {
            confirmatory[name]: f"{name} {decision}"
            for name, decision in result.items()
            if name in confirmatory
        }
    main = []
    sensitivity = []
    for identifier in sorted(by_id):
        report = by_id[identifier]
        table = report.get("mcnemar_table") or {}
        tango = _bounds(report.get("tango_ci"))
        tango_90 = _bounds(report.get("tango_ci_90"))
        newcombe = _bounds(report.get("newcombe_ci"))
        conservative = _bounds(report.get("conservative_ci"))
        cr1 = _bounds(report.get("cr1_ci"))
        baseline, treatment = report["conditions"]
        main.append(
            (
                identifier,
                report["endpoint"],
                baseline,
                treatment,
                report["planned_tasks"],
                report["included_tasks"],
                table.get("baseline_only"),
                table.get("treatment_only"),
                _float(report["delta"]),
                _float(report.get("midp_p")),
                _float(report.get("mcnemar_p")),
                *tango,
                *tango_90,
                *newcombe,
                *conservative,
                _float(report.get("signflip_p")),
                *cr1,
                decisions.get(identifier, "exploratory"),
            )
        )
        for key, value in sorted(report["sensitivity"].items()):
            sensitivity.append((identifier, key, _float(value)))
    return (
        Table(
            "paired",
            (
                "analysis_id",
                "endpoint",
                "baseline",
                "treatment",
                "planned_tasks",
                "included_tasks",
                "b",
                "c",
                "delta",
                "midp_p",
                "exact_p",
                "tango_low",
                "tango_high",
                "tango90_low",
                "tango90_high",
                "newcombe_low",
                "newcombe_high",
                "bonferroni_low",
                "bonferroni_high",
                "signflip_p",
                "cr1_low",
                "cr1_high",
                "decision",
            ),
            tuple(main),
        ),
        Table("paired-sensitivity", ("analysis_id", "bound", "delta"), tuple(sensitivity)),
    )


def security_tables(outcomes: Sequence[OutcomeRecord]) -> tuple[Table, ...]:
    """T-R4, T-R6: V/N/I/U/missing counts per measure (analysis) and condition."""
    if any(record.endpoint != "security" for record in outcomes):
        raise ValueError("the security table reads security outcomes only")
    cells: dict[tuple[str, str], Counter[str]] = {}
    for record in outcomes:
        cells.setdefault((record.analysis_id, record.condition), Counter())[record.outcome] += 1
    rows = []
    for identifier in sorted({key[0] for key in cells}):
        for condition in ordered_conditions({key[1] for key in cells if key[0] == identifier}):
            counts = cells[identifier, condition]
            observed = sum(counts[state] for state in ("V", "N", "I", "U"))
            rows.append(
                (
                    identifier,
                    condition,
                    *(counts[state] for state in SECURITY_STATES),
                    sum(counts.values()),
                    counts["V"] / observed if observed else None,
                )
            )
    return (
        Table(
            "security",
            ("analysis_id", "condition", *SECURITY_STATES, "records", "V_share_of_observed"),
            tuple(rows),
        ),
    )


def parser_tables(score: Mapping[str, Any]) -> tuple[Table, ...]:
    """T-R8: validity and per-leaf precision, recall and F1 of a ``score`` report."""
    summary = (
        (
            score["total"],
            score["valid"],
            _float(score["json_validity_rate"]),
            _float(score.get("parse_rate")),
            _float(score.get("mean_coverage_score")),
            _float(score.get("mean_hallucination_rate")),
        ),
    )
    fields = tuple(
        (
            name,
            _float(value["precision"]),
            _float(value["recall"]),
            _float(value["f1"]),
            value["support"],
        )
        for name, value in sorted(score["field_f1"].items())
    )
    return (
        Table(
            "parser",
            ("total", "valid", "validity", "parse_rate", "coverage", "hallucination"),
            summary,
        ),
        Table("parser-leaves", ("leaf", "precision", "recall", "f1", "support"), fields),
    )


# --- serialisation -----------------------------------------------------------

_LATEX: Final = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _csv_cell(value: Cell) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _latex_cell(value: Cell) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4f}"
    return "".join(_LATEX.get(char, char) for char in str(value))


def to_csv(table: Table, provenance: Provenance, kind: str) -> str:
    """CSV with ``#`` comment lines for the provenance, LF line ends."""
    buffer = io.StringIO()
    for line in provenance.lines(kind, table.name):
        buffer.write(f"# {line}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table.columns)
    for row in table.rows:
        writer.writerow([_csv_cell(cell) for cell in row])
    return buffer.getvalue()


def to_latex(table: Table, provenance: Provenance, kind: str) -> str:
    """A booktabs ``tabular`` with ``%`` comment lines for the provenance."""
    lines = [f"% {line}" for line in provenance.lines(kind, table.name)]
    alignment = "".join(
        "r" if all(isinstance(row[i], int | float) or row[i] is None for row in table.rows) else "l"
        for i in range(len(table.columns))
    )
    lines += [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_latex_cell(column) for column in table.columns) + r" \\",
        r"\midrule",
        *(" & ".join(_latex_cell(cell) for cell in row) + r" \\" for row in table.rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"
