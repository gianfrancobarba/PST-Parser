"""Reading run files and writing thesis tables and figures into a new directory (M20).

A run directory follows the layout of the operating plan::

    <run>/run-identity.json            run_id (required)
    <run>/protocol-freeze.json         freeze copy
    <run>/inputs/*.variants.jsonl      rendered variants
    <run>/generation/<journal>/generations.jsonl
    <run>/evaluation/<journal>/check-01/details.jsonl
    <run>/evaluation/<journal>/codeql-*/outcomes.jsonl
    <run>/analysis/<analysis>/report.json
    <run>/parser/score/results.json

``tables`` and ``figures`` take their inputs explicitly; ``thesis-tables``
reads every file of this layout that exists and lists the kinds it skipped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from pstparser.empirical import figures as fig
from pstparser.empirical import tables as tab
from pstparser.empirical.analysis import read_outcomes

STAGE: Final = "empirical-tables"


def read_json(path: Path) -> Any:
    """A UTF-8 JSON document."""
    return json.loads(path.read_bytes().decode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSON Lines split on the line feed only (model outputs may hold other separators)."""
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line]


def run_identity(run: Path) -> str:
    """The ``run_id`` of ``run/run-identity.json``."""
    path = run / "run-identity.json"
    if not path.is_file():
        raise ValueError(f"{path}: the run has no identity file")
    identifier = read_json(path).get("run_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError(f"{path}: run_id is missing")
    return identifier


def input_name(path: Path, run: Path) -> str:
    """The input as written in the files: relative to the run when it lies inside."""
    try:
        return path.resolve().relative_to(run.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class Outputs:
    """Tables and figures of one execution, with every input they read."""

    run: Path
    inputs: list[Path] = field(default_factory=list)
    tables: list[tuple[str, tab.Table]] = field(default_factory=list)
    figures: list[tuple[str, str, Any]] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def read(self, path: Path) -> Path:
        """Record ``path`` as an input; it must exist."""
        if not path.is_file():
            raise ValueError(f"{path}: no such input")
        if path not in self.inputs:
            self.inputs.append(path)
        return path

    def add(self, kind: str, built: Sequence[tab.Table], prefix: str | None = None) -> None:
        """Record tables of ``kind``, their names prefixed when a prefix is given."""
        for table in built:
            name = f"{prefix}-{table.name}" if prefix else table.name
            self.tables.append((kind, replace(table, name=name)))


def write(outputs: Outputs, out: Path, frozen: Mapping[str, Any] | None = None) -> list[Path]:
    """Write every table (CSV, LaTeX) and figure (PDF, PNG) and the stage manifest.

    Raises:
        FileExistsError: If ``out`` exists: each execution needs a new directory.
        ValueError: If two outputs would share a name.
    """
    from pstparser.repro.manifest import digest_file, source_snapshot, write_manifest

    names = [f"{table.name}.{suffix}" for _, table in outputs.tables for suffix in ("csv", "tex")]
    names += [f"{name}.{suffix}" for _, name, _ in outputs.figures for suffix in ("pdf", "png")]
    if len(set(names)) != len(names):
        raise ValueError("two outputs share a file name")
    if out.exists():
        raise FileExistsError(f"{out} exists: each execution needs a new directory")
    snapshot = source_snapshot()
    provenance = tab.Provenance(
        run_identity(outputs.run),
        snapshot["digest"],
        tuple((input_name(path, outputs.run), digest_file(path) or "") for path in outputs.inputs),
    )
    out.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []
    for kind, table in outputs.tables:
        for suffix, text in (
            ("csv", tab.to_csv(table, provenance, kind)),
            ("tex", tab.to_latex(table, provenance, kind)),
        ):
            path = out / f"{table.name}.{suffix}"
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            written.append(path)
    for kind, name, figure in outputs.figures:
        written += fig.save(figure, out, name, provenance, kind)
    write_manifest(
        {
            "stage": STAGE,
            "tables_version": tab.TABLES_VERSION,
            "figures_version": fig.FIGURES_VERSION,
            "run_id": provenance.run_id,
            "inputs": dict(provenance.inputs),
            "outputs": {path.name: digest_file(path) for path in written},
            "skipped": outputs.skipped,
            **dict(frozen or {"purpose": "exploratory", "freeze_sha256": None}),
            "source": snapshot,
        },
        out,
    )
    return written


def parse_confirmatory(values: Sequence[str]) -> dict[str, str]:
    """``H1=<analysis_id>`` options as a mapping."""
    result: dict[str, str] = {}
    for value in values:
        name, separator, identifier = value.partition("=")
        if not separator or not identifier or name in result:
            raise ValueError(f"expected HYPOTHESIS=ANALYSIS_ID once each, got {value!r}")
        result[name] = identifier
    return result


def _reports(outputs: Outputs, paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {input_name(p, outputs.run): read_json(outputs.read(p)) for p in paths}


def build_tables(
    outputs: Outputs,
    kind: str,
    *,
    samples: Sequence[Path] = (),
    exclusions: Path | None = None,
    variants: Sequence[Path] = (),
    journal: Path | None = None,
    details: Path | None = None,
    freeze: Path | None = None,
    reports: Sequence[Path] = (),
    confirmatory: Mapping[str, str] | None = None,
    gate: Path | None = None,
    outcomes: Sequence[Path] = (),
    score: Path | None = None,
    prefix: str | None = None,
) -> None:
    """Add the tables of ``kind`` built from the given files."""

    def need(value: object, option: str) -> None:
        if not value:
            raise ValueError(f"tables {kind} needs {option}")

    if kind == "samples":
        need(samples, "--sample")
        documents = {input_name(p, outputs.run): read_json(outputs.read(p)) for p in samples}
        excluded = read_json(outputs.read(exclusions)) if exclusions is not None else None
        outputs.add(kind, tab.samples_tables(documents, excluded), prefix)
    elif kind == "treatment":
        need(variants, "--variants")
        rows = [row for path in variants for row in read_jsonl(outputs.read(path))]
        outputs.add(kind, tab.treatment_tables(rows), prefix)
    elif kind == "tokens":
        need(journal, "--journal")
        assert journal is not None
        outputs.add(kind, tab.tokens_tables(read_jsonl(outputs.read(journal))), prefix)
    elif kind == "freeze":
        need(freeze, "--freeze")
        assert freeze is not None
        outputs.add(kind, tab.freeze_tables(read_json(outputs.read(freeze))), prefix)
    elif kind == "diagnostics":
        need(journal, "--journal")
        assert journal is not None
        rows = read_jsonl(outputs.read(journal))
        checked = read_jsonl(outputs.read(details)) if details is not None else None
        outputs.add(kind, tab.diagnostics_tables(rows, checked), prefix)
    elif kind == "paired":
        need(reports, "--report")
        gate_report = read_json(outputs.read(gate)) if gate is not None else None
        outputs.add(
            kind, tab.paired_tables(_reports(outputs, reports), confirmatory, gate_report), prefix
        )
    elif kind == "security":
        need(outcomes, "--outcomes")
        records = [record for path in outcomes for record in read_outcomes(outputs.read(path))]
        outputs.add(kind, tab.security_tables(records), prefix)
    elif kind == "loc":
        need(details, "--details")
        assert details is not None
        outputs.add(kind, tab.loc_tables(read_jsonl(outputs.read(details))), prefix)
    elif kind == "review":
        need(score, "--score")
        assert score is not None
        outputs.add(kind, tab.review_tables(read_json(outputs.read(score))), prefix)
    elif kind == "parser":
        need(score, "--score")
        assert score is not None
        outputs.add(kind, tab.parser_tables(read_json(outputs.read(score))), prefix)
    else:
        raise ValueError(f"unknown table kind {kind!r}; known: {', '.join(tab.KINDS)}")


def build_figures(
    outputs: Outputs,
    kind: str,
    *,
    reports: Sequence[Path] = (),
    variants: Sequence[Path] = (),
    journal: Path | None = None,
    prefix: str | None = None,
) -> None:
    """Add the figures of ``kind`` built from the given files."""
    name = f"{prefix}-{kind}" if prefix else kind
    if kind == "forest":
        if not reports:
            raise ValueError("figures forest needs --report")
        outputs.figures.append((kind, name, fig.forest(_reports(outputs, reports))))
    elif kind in {"effects", "leaves"}:
        if len(reports) != 1:
            raise ValueError(f"figures {kind} needs exactly one --report")
        (report,) = _reports(outputs, reports).values()
        if kind == "effects":
            outputs.figures.append((kind, name, fig.effects(report)))
        else:
            if not variants:
                raise ValueError("figures leaves needs --variants")
            rows = [row for path in variants for row in read_jsonl(outputs.read(path))]
            outputs.figures.append((kind, name, fig.leaves(report, rows)))
    elif kind == "tokens":
        if journal is None:
            raise ValueError("figures tokens needs --journal")
        outputs.figures.append((kind, name, fig.tokens(read_jsonl(outputs.read(journal)))))
    else:
        raise ValueError(f"unknown figure kind {kind!r}; known: {', '.join(fig.KINDS)}")


def build_thesis(
    run: Path,
    protocol: Path,
    confirmatory: Mapping[str, str] | None = None,
    gate: Path | None = None,
) -> Outputs:
    """Every table and figure the files of ``run`` (and ``protocol``) allow.

    ``gate`` decides whether H3 is in the confirmatory family: the parser gate
    report, or, under the prespecified time cut, the frozen
    ``confirmatory-family-v1`` document that fixes the family to H1 alone. It
    defaults to ``parser/gate/gate.json`` of the run.
    """
    if gate is None and (run / "parser" / "gate" / "gate.json").is_file():
        gate = run / "parser" / "gate" / "gate.json"
    outputs = Outputs(run)
    samples = sorted(
        path
        for path in (protocol / "samples").glob("*.json")
        if not path.name.endswith("_manifest.json")
    )
    exclusions = protocol / "oracles" / "bcb-exclusions.json"
    if samples:
        build_tables(
            outputs,
            "samples",
            samples=samples,
            exclusions=exclusions if exclusions.is_file() else None,
        )
    else:
        outputs.skipped["samples"] = f"no sample file in {protocol / 'samples'}"
    variants = sorted((run / "inputs").glob("*.variants.jsonl"))
    if variants:
        build_tables(outputs, "treatment", variants=variants)
    else:
        outputs.skipped["treatment"] = "no inputs/*.variants.jsonl"
    freeze = run / "protocol-freeze.json"
    if freeze.is_file():
        build_tables(outputs, "freeze", freeze=freeze)
    else:
        outputs.skipped["freeze"] = "no protocol-freeze.json"
    journals = sorted((run / "generation").glob("*/generations.jsonl"))
    for journal in journals:
        label = journal.parent.name
        details = run / "evaluation" / label / "check-01" / "details.jsonl"
        build_tables(outputs, "tokens", journal=journal, prefix=label)
        build_tables(
            outputs,
            "diagnostics",
            journal=journal,
            details=details if details.is_file() else None,
            prefix=label,
        )
        if len({row["condition"] for row in read_jsonl(journal)}) > 1:
            build_figures(outputs, "tokens", journal=journal, prefix=label)
    if not journals:
        outputs.skipped["tokens"] = outputs.skipped["diagnostics"] = "no generation journal"
    reports = sorted((run / "analysis").glob("*/report.json"))
    if reports:
        build_tables(outputs, "paired", reports=reports, confirmatory=confirmatory, gate=gate)
        build_figures(outputs, "forest", reports=reports)
        for report in reports:
            label = report.parent.name
            content = read_json(report)
            if content["task_estimates"]:
                build_figures(outputs, "effects", reports=[report], prefix=label)
                if "H-gold" in content["conditions"] and variants:
                    build_figures(
                        outputs, "leaves", reports=[report], variants=variants, prefix=label
                    )
    else:
        outputs.skipped["paired"] = "no analysis/*/report.json"
    security = sorted((run / "evaluation").glob("*/codeql-*/outcomes.jsonl"))
    if security:
        build_tables(outputs, "security", outcomes=security)
    else:
        outputs.skipped["security"] = "no evaluation/*/codeql-*/outcomes.jsonl"
    for codeql_details in sorted((run / "evaluation").glob("*/codeql-*/details.jsonl")):
        label = f"{codeql_details.parent.parent.name}-{codeql_details.parent.name}"
        build_tables(outputs, "loc", details=codeql_details, prefix=label)
    score = run / "parser" / "score" / "results.json"
    if score.is_file():
        build_tables(outputs, "parser", score=score)
    else:
        outputs.skipped["parser"] = "no parser/score/results.json"
    review = run / "review" / "blind-score.json"
    if review.is_file():
        build_tables(outputs, "review", score=review)
    else:
        outputs.skipped["review"] = "no review/blind-score.json"
    return outputs
