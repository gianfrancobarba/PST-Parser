"""Re-annotation sheets in format A' (decision of 20/11/2026), and their collection.

Format A': the units whose leaf a rule fixes (benchmark formulas, markers, the
first sentence) come precompiled with their leaf; the judgement units
(``judgement-units-v2``: G3 prose after the first sentence, B6 "Note that:")
come with an empty leaf, to be decided from scratch, with no suggestion, no
gold, no LLM proposal. The unit boundaries are those of the frozen rule draft,
so the student never splits or merges a span, as in the gold review.

Written after the freeze (394c893) and before any re-annotation exists; it lives
in results/ because the freeze admits unlisted files only there. It uses the
frozen package and changes none of it.

usage (repository root, .venv):
  python -I results/tools/reannotation_sheets.py build SAMPLE SOURCES OUT.xlsx
  python -I results/tools/reannotation_sheets.py collect SAMPLE SOURCES SHEET.xlsx \
      --date YYYY-MM-DD --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from pstparser.empirical import annotations as gold
from pstparser.empirical.benchmarks import SourceRecord, read_sources
from pstparser.empirical.contracts import SpanAnnotation
from pstparser.pst.taxonomy import LEAF_PATHS

SHEET_FORMAT: Final = "reannotation-sheet-a1-v1"
ANNOTATOR: Final = "studente"
SHEET: Final = "foglio"
HEADER: Final = (
    "task_id",
    "n",
    "tipo",
    "regola",
    "testo",
    "foglia_fissa",
    "foglia",
    "start",
    "end",
)
FIXED: Final = "fissa"
JUDGEMENT: Final = "giudizio"
_FILL: Final = PatternFill("solid", fgColor="FFF2CC")

INSTRUCTIONS: Final = (
    "Ri-annotazione da zero, formato A' (" + SHEET_FORMAT + ").",
    "",
    "Righe 'fissa': la foglia è decisa da una regola ed è già scritta. Non modificarle.",
    "Righe 'giudizio' (gialle): scegli la foglia nella colonna 'foglia' dal menu.",
    "Non spezzare e non unire le unità; non cambiare start, end, tipo, foglia_fissa.",
    "",
    "Senza nessuna assistenza: niente LLM, niente gold, niente review-log, niente",
    "proposte o predizioni. Puoi usare solo le linee guida protocol/guidelines/gold-v2.0.md.",
    "",
    "Foglie ammesse: " + ", ".join(LEAF_PATHS),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sample(sample: Path, sources: Path) -> list[SourceRecord]:
    """The sources of the frozen sample, in its order, checked against its hashes.

    Raises:
        ValueError: If the sources file or a task is not the one the sample recorded.
    """
    record = json.loads(sample.read_text("utf-8"))
    if record["kind"] != "reannotation-sample":
        raise ValueError(f"{sample}: not a re-annotation sample")
    if _sha256(sources) != record["sources_sha256"]:
        raise ValueError(f"{sources}: not the sources file of the sample")
    texts = {row.task_id: row for row in read_sources(sources)}
    chosen = []
    for task in record["tasks"]:
        source = texts[task["task_id"]]
        if source.source_sha256 != task["source_sha256"]:
            raise ValueError(f"{task['task_id']}: the source does not match the sample")
        chosen.append(source)
    return chosen


def sheet_rows(source: SourceRecord, rule: str) -> list[dict[str, Any]]:
    """One row per unit of the rule draft; judgement units carry no leaf."""
    rows = []
    for rank, unit in enumerate(gold.proposed_units(source, rule), start=1):
        judged = unit.kind == "judgement"
        rows.append(
            {
                "task_id": source.task_id,
                "n": rank,
                "tipo": JUDGEMENT if judged else FIXED,
                "regola": unit.rule,
                "testo": source.text[unit.start : unit.end],
                "foglia_fissa": None if judged else unit.path,
                "foglia": None,
                "start": unit.start,
                "end": unit.end,
            }
        )
    return rows


def build(sample: Path, sources: Path, out: Path) -> dict[str, int]:
    """Write the blank sheet of a sample; an existing file is never replaced."""
    if out.exists():
        raise FileExistsError(out)
    rule = json.loads(sample.read_text("utf-8"))["rule"]
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = SHEET
    sheet.append(list(HEADER))
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    leaves = DataValidation(type="list", formula1='"' + ",".join(LEAF_PATHS) + '"')
    leaves.error = "Scegli una foglia dal menu."
    sheet.add_data_validation(leaves)
    counts = {"tasks": 0, "units": 0, "judgement_units": 0}
    for source in load_sample(sample, sources):
        counts["tasks"] += 1
        for row in sheet_rows(source, rule):
            sheet.append([row[name] for name in HEADER])
            line = sheet.max_row
            text = sheet.cell(line, HEADER.index("testo") + 1)
            text.data_type = "s"  # a text starting with '=' is not a formula
            text.alignment = Alignment(wrap_text=True, vertical="top")
            counts["units"] += 1
            if row["tipo"] == JUDGEMENT:
                counts["judgement_units"] += 1
                target = sheet.cell(line, HEADER.index("foglia") + 1)
                leaves.add(target)
                for cell in sheet[line]:
                    cell.fill = _FILL
        sheet.append([])
    for column, width in zip("ABCDEFGHI", (20, 4, 9, 7, 90, 20, 22, 7, 7), strict=True):
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"
    notes = book.create_sheet("istruzioni")
    for line in INSTRUCTIONS:
        notes.append([line])
    notes.column_dimensions["A"].width = 110
    out.parent.mkdir(parents=True, exist_ok=True)
    book.save(out)
    return counts


def _read_sheet(path: Path) -> dict[str, list[dict[str, Any]]]:
    sheet = load_workbook(path, read_only=True)[SHEET]
    lines = list(sheet.iter_rows(values_only=True))
    if tuple(lines[0][: len(HEADER)]) != HEADER:
        raise ValueError(f"{path}: the header of the sheet was changed")
    found: dict[str, list[dict[str, Any]]] = {}
    for values in lines[1:]:
        if not any(value not in (None, "") for value in values):
            continue
        row = dict(zip(HEADER, values, strict=False))
        found.setdefault(str(row["task_id"]), []).append(row)
    return found


def _leaf(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def collect(
    sample: Path, sources: Path, sheet: Path, when: str, out: Path
) -> list[gold.AnnotationRow]:
    """Turn a filled sheet into an annotation file the frozen ``agreement`` reads.

    The fixed units must be exactly the rule draft; every judgement unit needs a
    leaf of the taxonomy. The rows are written as the student's annotation
    (never ``rule:``), status ``reviewed``.

    Raises:
        ValueError: If the sheet does not match the sample, the draft, or is incomplete.
    """
    date.fromisoformat(when)
    rule = json.loads(sample.read_text("utf-8"))["rule"]
    filled = _read_sheet(sheet)
    chosen = load_sample(sample, sources)
    if set(filled) != {source.task_id for source in chosen}:
        raise ValueError("the sheet does not hold exactly the tasks of the sample")
    digest = _sha256(sheet)
    problems: list[str] = []
    rows = []
    for source in chosen:
        expected = sheet_rows(source, rule)
        given = filled[source.task_id]
        if len(given) != len(expected):
            raise ValueError(f"{source.task_id}: units were added or removed")
        spans = []
        for want, got in zip(expected, given, strict=True):
            for name in ("n", "tipo", "regola", "start", "end", "foglia_fissa"):
                if (got[name] if got[name] != "" else None) != want[name]:
                    raise ValueError(f"{source.task_id} unit {want['n']}: {name} was changed")
            leaf = want["foglia_fissa"] if want["tipo"] == FIXED else _leaf(got["foglia"])
            if want["tipo"] == FIXED and _leaf(got["foglia"]) not in (None, leaf):
                problems.append(f"{source.task_id} unit {want['n']}: a fixed unit got a leaf")
            if leaf is None or leaf not in LEAF_PATHS:
                problems.append(f"{source.task_id} unit {want['n']}: leaf {leaf!r}")
                continue
            phrase = source.text[want["start"] : want["end"]]
            spans.append(SpanAnnotation(leaf, want["start"], want["end"], phrase))
        rows.append(
            gold.AnnotationRow(
                task_id=source.task_id,
                source_sha256=source.source_sha256,
                annotation_version=gold.ANNOTATION_VERSION,
                guidelines_version=gold.GUIDELINES_VERSION,
                annotator=ANNOTATOR,
                reviewer=ANNOTATOR,
                status="reviewed",
                spans=tuple(spans),
                notes=f"{SHEET_FORMAT}; re-annotation from scratch on {when}; "
                f"sheet sha256 {digest}",
            )
        )
    if problems:
        raise ValueError("incomplete sheet: " + "; ".join(problems[:10]))
    gold.write_annotations(rows, out)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: build a blank sheet, or collect a filled one."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "collect"):
        command = commands.add_parser(name)
        command.add_argument("sample", type=Path)
        command.add_argument("sources", type=Path)
        if name == "build":
            command.add_argument("out", type=Path)
        else:
            command.add_argument("sheet", type=Path)
            command.add_argument("--date", required=True)
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.sample, args.sources, args.out)))
    else:
        rows = collect(args.sample, args.sources, args.sheet, args.date, args.out)
        print(json.dumps({"tasks": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
