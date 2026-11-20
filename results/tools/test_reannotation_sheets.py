# ruff: noqa: D103
"""Tests of the A' re-annotation sheets, on synthetic LLMSecEval sources."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from openpyxl import load_workbook

from pstparser.empirical import annotations as gold
from pstparser.empirical.benchmarks import LLMSECEVAL, SourceRecord, write_sources
from pstparser.pst.taxonomy import LEAF_PATHS


def _tool() -> ModuleType:
    path = Path(__file__).with_name("reannotation_sheets.py")
    spec = importlib.util.spec_from_file_location("reannotation_sheets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()

TEXTS = {
    "LSE-001": "Generate Python code for the following:\nThe app reads a file. "
    "It must reject paths outside the folder. =Return the content as JSON.",
    "LSE-002": "Generate Python code for the following:\nA function checks a password.",
}


def _source(task: str, text: str) -> SourceRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceRecord(task, text, digest, digest, LLMSECEVAL, "r", "f", task, "fam", "p", ())


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, Path]:
    sources = tmp_path / "sources.jsonl"
    records = [_source(task, text) for task, text in TEXTS.items()]
    write_sources(records, sources)
    record = {
        "kind": "reannotation-sample",
        "rule": gold.LSE_RULE,
        "sources_sha256": hashlib.sha256(sources.read_bytes()).hexdigest(),
        "tasks": [{"task_id": r.task_id, "source_sha256": r.source_sha256} for r in records],
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, sources


def _fill(sheet: Path, leaf: str | None = "context.constraints") -> None:
    book = load_workbook(sheet)
    rows = book[tool.SHEET]
    for line in range(2, rows.max_row + 1):
        if rows.cell(line, 3).value == tool.JUDGEMENT:
            rows.cell(line, 7).value = leaf
    book.save(sheet)


def test_the_sheet_precompiles_rule_units_and_leaves_judgement_units_blank(
    sample: tuple[Path, Path], tmp_path: Path
) -> None:
    counts = tool.build(*sample, tmp_path / "s.xlsx")
    rows = list(load_workbook(tmp_path / "s.xlsx")[tool.SHEET].iter_rows(values_only=True))
    body = [dict(zip(tool.HEADER, r, strict=False)) for r in rows[1:] if any(r)]
    judged = [r for r in body if r["tipo"] == tool.JUDGEMENT]
    assert counts == {"tasks": 2, "units": len(body), "judgement_units": len(judged)}
    assert judged and all(r["foglia"] is None and r["foglia_fissa"] is None for r in judged)
    assert all(r["foglia_fissa"] in LEAF_PATHS for r in body if r not in judged)
    assert any(str(r["testo"]).startswith("=") for r in body)  # kept as text, not a formula


def test_a_filled_sheet_becomes_a_human_annotation_file(
    sample: tuple[Path, Path], tmp_path: Path
) -> None:
    sheet = tmp_path / "s.xlsx"
    tool.build(*sample, sheet)
    _fill(sheet)
    tool.collect(*sample, sheet, "2026-11-26", tmp_path / "out.jsonl")
    rows = gold.load_annotations(tmp_path / "out.jsonl")
    assert [r.task_id for r in rows] == list(TEXTS)
    assert all(r.annotator == "studente" and not r.annotator.startswith("rule:") for r in rows)
    first = rows[0]
    draft = gold.proposed_units(_source("LSE-001", TEXTS["LSE-001"]), gold.LSE_RULE)
    assert [(s.start, s.end) for s in first.spans] == [(u.start, u.end) for u in draft]
    judged = {(u.start, u.end) for u in draft if u.kind == "judgement"}
    assert {s.path for s in first.spans if (s.start, s.end) in judged} == {"context.constraints"}


def test_an_empty_or_unknown_leaf_is_refused(sample: tuple[Path, Path], tmp_path: Path) -> None:
    sheet = tmp_path / "s.xlsx"
    tool.build(*sample, sheet)
    with pytest.raises(ValueError, match="incomplete sheet"):
        tool.collect(*sample, sheet, "2026-11-26", tmp_path / "a.jsonl")
    _fill(sheet, "context.whatever")
    with pytest.raises(ValueError, match="incomplete sheet"):
        tool.collect(*sample, sheet, "2026-11-26", tmp_path / "b.jsonl")
    assert not (tmp_path / "a.jsonl").exists() and not (tmp_path / "b.jsonl").exists()


def test_a_changed_fixed_unit_is_refused(sample: tuple[Path, Path], tmp_path: Path) -> None:
    sheet = tmp_path / "s.xlsx"
    tool.build(*sample, sheet)
    _fill(sheet)
    book = load_workbook(sheet)
    book[tool.SHEET].cell(2, 6).value = "examples"
    book.save(sheet)
    with pytest.raises(ValueError, match="foglia_fissa was changed"):
        tool.collect(*sample, sheet, "2026-11-26", tmp_path / "out.jsonl")


def test_other_sources_are_refused(sample: tuple[Path, Path], tmp_path: Path) -> None:
    path, sources = sample
    sources.write_bytes(sources.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not the sources file"):
        tool.build(path, sources, tmp_path / "s.xlsx")


def test_an_existing_sheet_is_never_replaced(sample: tuple[Path, Path], tmp_path: Path) -> None:
    (tmp_path / "s.xlsx").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        tool.build(*sample, tmp_path / "s.xlsx")
