"""Thesis figures regenerated from versioned run files.

Figures are drawn with matplotlib's object interface (no global state) and
saved as PDF and PNG without a creation date, so the same inputs give the same
bytes. Every figure carries the run and the source digest in a footer and in
the file metadata. matplotlib belongs to the ``analysis`` extra and is imported
only here.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from pstparser.empirical.tables import Provenance, input_tokens, leaf_count, ordered_conditions

FIGURES_VERSION: Final = "thesis-figures-v1"
KINDS: Final = ("forest", "effects", "leaves", "tokens")


def _figure(width: float, height: float) -> Any:
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), dpi=100)


def _footer(figure: Any, provenance: Provenance, kind: str) -> None:
    figure.text(
        0.01,
        0.01,
        f"{FIGURES_VERSION} {kind} | run {provenance.run_id} | "
        f"source {provenance.source_digest[:12]}",
        fontsize=6,
        color="0.4",
    )


def save(figure: Any, directory: Path, name: str, provenance: Provenance, kind: str) -> list[Path]:
    """Write ``name``.pdf and ``name``.png, refusing existing files."""
    _footer(figure, provenance, kind)
    subject = "; ".join(provenance.lines(kind, name))
    written = []
    for suffix, metadata in (
        (
            "pdf",
            {"CreationDate": None, "ModDate": None, "Subject": subject, "Title": name},
        ),
        ("png", {"Description": subject, "Title": name}),
    ):
        path = directory / f"{name}.{suffix}"
        with path.open("xb") as handle:
            figure.savefig(handle, format=suffix, metadata=metadata)
        written.append(path)
    return written


def _interval(report: Mapping[str, Any]) -> tuple[float, float] | None:
    """The primary interval: Tango for one greedy pair per task, else CR1, else bootstrap."""
    for key in ("tango_ci", "cr1_ci", "bootstrap_ci"):
        if report.get(key) is not None:
            low, high = report[key]
            return float(low), float(high)
    return None


def forest(reports: Mapping[str, Mapping[str, Any]]) -> Any:
    """Each analysis' difference with its primary 95% interval, in percentage points."""
    entries = sorted(
        (report for report in reports.values() if report["delta"] is not None),
        key=lambda report: report["analysis_id"],
    )
    if not entries:
        raise ValueError("no report with an estimate")
    figure = _figure(6.0, 1.0 + 0.45 * len(entries))
    axes = figure.add_axes((0.38, 0.18, 0.58, 0.75))
    for row, report in enumerate(reversed(entries)):
        delta = 100 * report["delta"]
        interval = _interval(report)
        if interval is not None:
            axes.plot([100 * interval[0], 100 * interval[1]], [row, row], color="0.2", lw=1.2)
        axes.plot([delta], [row], marker="s", color="0.1", ms=5)
    axes.axvline(0.0, color="0.6", lw=0.8, ls="--")
    axes.set_yticks(range(len(entries)))
    axes.set_yticklabels(
        [
            f"{r['analysis_id']} ({r['conditions'][1]} - {r['conditions'][0]})"
            for r in reversed(entries)
        ],
        fontsize=7,
    )
    axes.set_xlabel("difference (percentage points)", fontsize=8)
    return figure


def effects(report: Mapping[str, Any]) -> Any:
    """Distribution of the per-task (or per-family) differences of one analysis."""
    estimates = report["task_estimates"]
    if not estimates:
        raise ValueError("the report has no complete task")
    clustered = estimates[0]["family_id"] is not None
    if clustered:
        families: dict[str, list[float]] = {}
        for estimate in estimates:
            families.setdefault(estimate["family_id"], []).append(estimate["difference"])
        values = [100 * statistics.fmean(families[name]) for name in sorted(families)]
        unit = "family"
    else:
        values = [100 * estimate["difference"] for estimate in estimates]
        unit = "task"
    counts = sorted(Counter(round(value, 6) for value in values).items())
    figure = _figure(5.0, 3.0)
    axes = figure.add_axes((0.14, 0.2, 0.8, 0.7))
    width = min([b - a for (a, _), (b, _) in pairwise(counts)] or [100.0])
    axes.bar([value for value, _ in counts], [n for _, n in counts], width=0.8 * width, color="0.4")
    axes.set_xlabel(f"difference per {unit} (percentage points)", fontsize=8)
    axes.set_ylabel(f"{unit}s", fontsize=8)
    axes.set_title(report["analysis_id"], fontsize=8)
    return figure


def leaves(report: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]) -> Any:
    """Exploratory: mean difference by the number of non-empty leaves of H-gold."""
    counts = {
        row["task_id"]: leaf_count(row["text"]) for row in variants if row["condition"] == "H-gold"
    }
    groups: dict[int, list[float]] = {}
    for estimate in report["task_estimates"]:
        if estimate["task_id"] not in counts:
            raise ValueError(f"{estimate['task_id']}: no H-gold variant")
        groups.setdefault(counts[estimate["task_id"]], []).append(estimate["difference"])
    if not groups:
        raise ValueError("the report has no complete task")
    keys = sorted(groups)
    figure = _figure(5.0, 3.0)
    axes = figure.add_axes((0.14, 0.2, 0.8, 0.7))
    axes.bar(
        [str(key) for key in keys],
        [100 * statistics.fmean(groups[key]) for key in keys],
        color="0.4",
    )
    for index, key in enumerate(keys):
        axes.annotate(f"n={len(groups[key])}", (index, 0), ha="center", va="bottom", fontsize=6)
    axes.axhline(0.0, color="0.6", lw=0.8)
    axes.set_xlabel("non-empty leaves in H-gold", fontsize=8)
    axes.set_ylabel("mean difference (pp)", fontsize=8)
    axes.set_title(report["analysis_id"], fontsize=8)
    return figure


def tokens(journal: Sequence[Mapping[str, Any]], baseline: str = "T") -> Any:
    """Per-task ratio of input tokens to the baseline, one box per condition."""
    measured = input_tokens(journal)
    conditions = [c for c in ordered_conditions({c for _, c in measured}) if c != baseline]
    tasks = sorted({task for task, _ in measured})
    if not conditions or not any((task, baseline) in measured for task in tasks):
        raise ValueError(f"the journal needs {baseline} and at least one other condition")
    data = [
        [
            measured[task, condition] / measured[task, baseline]
            for task in tasks
            if (task, condition) in measured and (task, baseline) in measured
        ]
        for condition in conditions
    ]
    figure = _figure(5.0, 3.0)
    axes = figure.add_axes((0.14, 0.2, 0.8, 0.7))
    axes.boxplot(data, tick_labels=conditions)
    axes.axhline(1.0, color="0.6", lw=0.8, ls="--")
    axes.set_ylabel(f"input tokens / {baseline}", fontsize=8)
    return figure
