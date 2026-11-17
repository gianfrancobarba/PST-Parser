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
from pstparser.empirical.inference import (
    CONFIRMATORY_WEIGHTS,
    cluster_interval_at,
    four_way_reading,
    tango_interval,
    tost,
    weighted_holm,
    weighted_holm_levels,
)
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
    "loc",
    "review",
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


def _equivalence_interval(report: Mapping[str, Any]) -> tuple[float, float] | None:
    """The 90% interval of the TOST: Tango with one greedy pair per task, else CR1."""
    if report.get("tango_ci_90") is not None:
        low, high = report["tango_ci_90"]
        return float(low), float(high)
    if report.get("cr1_ci") is not None:
        return cluster_interval_at(report["cr1_ci"], report["cr1_se"], report["cr1_df"], 0.90)
    return None


#: What each confirmatory hypothesis must be (00 A-3, R-STA-01): endpoint,
#: conditions and, for H3, the S_parse estimate of ``hauto-ledgers``.
CONFIRMATORY_REPORTS: Final = {
    "H1": ("functional", ["T", "H-gold"], ""),
    "H3": ("functional", ["T", "H-auto"], "-s-parse"),
}


def _check_family(
    by_id: Mapping[str, Mapping[str, Any]],
    confirmatory: Mapping[str, str],
    gate: Mapping[str, Any] | None,
) -> None:
    """The family follows the parser gate, and each member is the report it must be.

    Raises:
        ValueError: If the gate report is missing or malformed, H3 is in the
            family against the gate (or left out of it), or a report has the
            wrong endpoint, conditions, suffix, test or task count.
    """
    if set(confirmatory) - set(CONFIRMATORY_WEIGHTS) or "H1" not in confirmatory:
        raise ValueError("the confirmatory family is H1 and, optionally, H3")
    if gate is None or gate.get("format") != "parser-gate-v1":
        raise ValueError("the confirmatory family needs the parser gate report (parser-gate-v1)")
    if gate.get("h3") not in {"confirmatory", "descriptive"}:
        raise ValueError("the parser gate report has no H3 decision")
    if (gate["h3"] == "confirmatory") != ("H3" in confirmatory):
        raise ValueError(
            f"the parser gate makes H3 {gate['h3']}: the family must "
            f"{'include' if gate['h3'] == 'confirmatory' else 'leave out'} H3"
        )
    for hypothesis, identifier in confirmatory.items():
        if identifier not in by_id:
            raise ValueError(f"{hypothesis}: no report with analysis_id {identifier!r}")
        report = by_id[identifier]
        endpoint, conditions, suffix = CONFIRMATORY_REPORTS[hypothesis]
        if report["endpoint"] != endpoint or list(report["conditions"]) != conditions:
            raise ValueError(
                f"{hypothesis} must be the {endpoint} {' vs '.join(conditions)} report"
            )
        if not identifier.endswith(suffix):
            raise ValueError(f"{hypothesis} must be the analysis whose id ends with {suffix!r}")
        if report.get("midp_p") is None:
            raise ValueError(f"{hypothesis}: the primary test is the mid-p of one greedy pair")
        if hypothesis == "H3" and report["planned_tasks"] != gate.get("s_parse_size"):
            raise ValueError("H3 must be planned on the S_parse of the parser gate")


def paired_tables(
    reports: Mapping[str, Mapping[str, Any]],
    confirmatory: Mapping[str, str] | None = None,
    gate: Mapping[str, Any] | None = None,
) -> tuple[Table, ...]:
    """T-R2, T-R3, T-R5, T-X1: paired estimates, tests, intervals and sensitivity.

    ``confirmatory`` maps H1 (and H3, if it stayed in the family) to the
    ``analysis_id`` of its report; their primary p-values are decided with the
    weighted Holm procedure (0.04 / 0.01, with recycling). Whether H3 is in the
    family is read from ``gate``, the report of ``parser-gate``: without H3 the
    family is H1 alone, read at 0.05 (failed parser gate).

    Every report also gets the TOST at +/-5 pp on its 90% interval and the
    four-way reading E1-E4 (00 A-4), crossed with the primary test at 0.05.
    """
    by_id = {report["analysis_id"]: report for report in reports.values()}
    if len(by_id) != len(reports):
        raise ValueError("two reports share an analysis_id")
    decisions: dict[str, str] = {}
    levels: dict[str, float] = {}
    if confirmatory:
        _check_family(by_id, confirmatory, gate)
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
        # The difference of a confirmatory row is read at the level the weighted
        # Holm assigned it, not at 0.05 (R-STA-02).
        levels = {
            confirmatory[name]: level
            for name, level in weighted_holm_levels(p_values, CONFIRMATORY_WEIGHTS).items()
            if name in confirmatory and level is not None
        }
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
        equivalence = _equivalence_interval(report)
        primary = _primary_p(report)
        equivalent = tost(equivalence) if equivalence is not None else None
        level = levels.get(identifier)
        reading = (
            four_way_reading(primary, equivalence, alpha=level if level is not None else 0.05)
            if primary is not None and equivalence is not None
            else None
        )
        # Readings at 3 and 10 pp, always reported (06 S7), on the same test and interval.
        other: list[str | None] = []
        for margin in (0.03, 0.10):
            if equivalence is None:
                other += [None, None]
                continue
            other.append("yes" if tost(equivalence, margin) else "no")
            other.append(
                four_way_reading(
                    primary, equivalence, alpha=level if level is not None else 0.05, margin=margin
                )
                if primary is not None
                else None
            )
        holm_ci: tuple[float | None, float | None] = (None, None)
        if level is not None and table:
            holm_ci = _bounds(
                tango_interval(
                    table["baseline_only"],
                    table["treatment_only"],
                    report["included_tasks"],
                    1 - level,
                )
            )
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
                *_bounds(equivalence),
                None if equivalent is None else "yes" if equivalent else "no",
                reading,
                *other,
                level,
                *holm_ci,
                decisions.get(identifier, "exploratory"),
            )
        )
        for key, value in sorted(report["sensitivity"].items()):
            sensitivity.append((identifier, key, _float(value)))
        family = report.get("family_sensitivity")
        if family:
            low, high = family["delta_fam_ci"]
            sensitivity += [
                (identifier, "delta_fam", _float(family["delta_fam"])),
                (identifier, "delta_fam_ci_low", _float(low)),
                (identifier, "delta_fam_ci_high", _float(high)),
                (identifier, "delta_fam_differs_2pp", float(family["delta_fam_differs_2pp"])),
                (identifier, "lofo_min", _float(family["lofo_min"])),
                (identifier, "lofo_max", _float(family["lofo_max"])),
            ]
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
                "equivalence90_low",
                "equivalence90_high",
                "tost_5pp",
                "reading_5pp",
                "tost_3pp",
                "reading_3pp",
                "tost_10pp",
                "reading_10pp",
                "holm_alpha",
                "holm_ci_low",
                "holm_ci_high",
                "decision",
            ),
            tuple(main),
        ),
        Table("paired-sensitivity", ("analysis_id", "bound", "delta"), tuple(sensitivity)),
    )


def loc_tables(details: Sequence[Mapping[str, Any]]) -> tuple[Table, ...]:
    """LOC and CodeQL findings per 100 LOC by condition (07 C4), usable programs only.

    A form that only writes more code gets more findings of the general
    measure; the rate per 100 LOC is the prespecified sensitivity.
    """
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in details:
        if row.get("usable"):
            by_condition.setdefault(row["condition"], []).append(row)
    rows = []
    for condition in ordered_conditions(set(by_condition)):
        members = by_condition[condition]
        loc = sum(int(row["loc"]) for row in members)
        findings = sum(len(row["alerts"]) for row in members)
        rows.append(
            (
                condition,
                len(members),
                loc / len(members),
                findings,
                100 * findings / loc if loc else None,
            )
        )
    return (
        Table(
            "security-loc",
            ("condition", "usable_programs", "mean_loc", "findings", "findings_per_100_loc"),
            tuple(rows),
        ),
    )


def review_tables(score: Mapping[str, Any]) -> tuple[Table, ...]:
    """T-R-S4: the blind review (04 §3.10) by condition, with the accuracy of the blind."""
    rows = []
    for condition in ordered_conditions(set(score["targeted"]) | set(score["general"])):
        targeted = score["targeted"].get(condition, {})
        general = score["general"].get(condition, {})
        rows.append(
            (
                condition,
                targeted.get("V"),
                targeted.get("false_positive"),
                targeted.get("N"),
                targeted.get("false_negative"),
                general.get("V"),
                general.get("false_positive"),
            )
        )
    blind = score["blind"]
    low, high = _bounds(blind["accuracy_ci95"])
    return (
        Table(
            "review",
            (
                "condition",
                "targeted_V",
                "targeted_false_positive",
                "targeted_N",
                "targeted_false_negative",
                "general_V",
                "general_false_positive",
            ),
            tuple(rows),
        ),
        Table(
            "review-blind",
            ("cases", "usability_kappa", "guesses", "correct", "accuracy", "low", "high"),
            (
                (
                    score["cases"],
                    _float(score["usability_kappa"]),
                    blind["guesses"],
                    blind["correct"],
                    _float(blind["accuracy"]),
                    low,
                    high,
                ),
            ),
        ),
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
