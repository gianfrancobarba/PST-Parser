# ruff: noqa: D103
"""Tests of the r4 training data tool (exploratory analysis R), on synthetic data."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pstparser.data.annotations import read_annotations
from pstparser.data.splits import Split, load_split, save_split
from pstparser.inference.parse_prompts import gold_target
from pstparser.pst.target import serialise_target
from pstparser.pst.taxonomy import assemble


def _tool() -> ModuleType:
    path = Path(__file__).with_name("r4_data.py")
    spec = importlib.util.spec_from_file_location("r4_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()

TEXT_A = (
    "Do the thing. It reads a file. Note that: never write.\n"
    "The function should output with:\n    int: x"
)
TEXT_B = "Compute a sum. Use numpy."


def _span(text: str, path: str, start: int, end: int) -> dict[str, Any]:
    return {"path": path, "start": start, "end": end, "phrase": text[start:end]}


def _drafts() -> list[dict[str, Any]]:
    def row(task: str, text: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "format": "pst-gold-annotations-v1",
            "task_id": task,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "annotation_version": "gold-v2.0",
            "guidelines_version": "gold-v2.0",
            "annotator": "rule:bcb-markers-v1",
            "reviewer": None,
            "status": "draft",
            "spans": spans,
            "notes": "",
        }

    a = [
        _span(TEXT_A, "main_instruction", 0, 13),
        _span(TEXT_A, "main_instruction", 14, 30),
        _span(TEXT_A, "context.constraints", 31, 54),
        _span(TEXT_A, "context.format", 55, len(TEXT_A)),
    ]
    b = [_span(TEXT_B, "main_instruction", 0, 14), _span(TEXT_B, "main_instruction", 15, 25)]
    return [row("T/1", TEXT_A, a), row("T/2", TEXT_B, b)]


UNITS = {("T/1", 14, 30), ("T/1", 31, 54), ("T/2", 15, 25)}


def _decisions(**change: str) -> list[dict[str, Any]]:
    rows = [
        {
            "task_id": "T/1",
            "start": "14",
            "end": "30",
            "proposta_ia": "main_instruction",
            "decisione": "context.data",
        },
        {
            "task_id": "T/1",
            "start": "31",
            "end": "54",
            "proposta_ia": "context.constraints",
            "decisione": "context.constraints",
        },
        {
            "task_id": "T/2",
            "start": "15",
            "end": "25",
            "proposta_ia": "context.constraints",
            "decisione": "context.constraints",
        },
    ]
    for key, value in change.items():
        task, start = key.split("_")
        for r in rows:
            if r["task_id"] == f"T/{task}" and r["start"] == start:
                r["decisione"] = value
    return rows


def test_apply_changes_only_the_decided_leaves() -> None:
    drafts = _drafts()
    annotations, log = tool.apply_decisions(drafts, _decisions(), UNITS, reviewed=False)
    a = annotations[0]["spans"]
    assert [s["path"] for s in a] == [
        "main_instruction",
        "context.data",
        "context.constraints",
        "context.format",
    ]
    assert annotations[1]["spans"][1]["path"] == "context.constraints"
    for new, old in zip(annotations, drafts, strict=True):
        assert [(s["start"], s["end"], s["phrase"]) for s in new["spans"]] == [
            (s["start"], s["end"], s["phrase"]) for s in old["spans"]
        ]
        assert {k: v for k, v in new.items() if k != "spans"} == {
            k: v for k, v in old.items() if k != "spans"
        }
    assert drafts[0]["spans"][1]["path"] == "main_instruction", "input mutated"
    assert [(r["task_id"], r["proposta_llm"], r["decisione"]) for r in log] == [
        ("T/1", "main_instruction", "context.data"),
        ("T/1", "context.constraints", "context.constraints"),
        ("T/2", "context.constraints", "context.constraints"),
    ]


def test_apply_sets_status_only_when_asked() -> None:
    annotations, _ = tool.apply_decisions(_drafts(), _decisions(), UNITS, reviewed=True)
    assert all(a["status"] == "reviewed" and a["reviewer"] == "studente" for a in annotations)


@pytest.mark.parametrize(
    ("decisions", "message"),
    [
        (_decisions(**{"1_14": ""}), "no decision"),
        (_decisions(**{"1_14": "context.datta"}), "unknown leaf"),
        (_decisions()[:2], "missing"),
        (
            [
                *_decisions(),
                {
                    "task_id": "T/1",
                    "start": "0",
                    "end": "13",
                    "proposta_ia": "x",
                    "decisione": "main_instruction",
                },
            ],
            "not a judgement unit",
        ),
        ([*_decisions(), _decisions()[0]], "twice"),
    ],
)
def test_apply_refuses_incomplete_or_foreign_decisions(
    decisions: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tool.apply_decisions(_drafts(), decisions, UNITS, reviewed=False)


def test_apply_refuses_a_unit_that_is_not_a_draft_span() -> None:
    with pytest.raises(ValueError, match="not a draft span"):
        tool.apply_decisions(_drafts(), _decisions(), {*UNITS, ("T/2", 1, 5)}, reviewed=False)


def test_read_decisions_accepts_excel_csv(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    with path.open("w", encoding="cp1252", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            ["task_id", "start", "end", "proposta_ia", "decisione", "regola", "mixed", "motivo"]
        )
        writer.writerow(
            ["T/1", "14", "30", "main_instruction", " context.data ", "G3.6", "sì", "perché"]
        )
    rows = tool.read_decisions(path)
    assert rows[0]["decisione"] == "context.data" and rows[0]["mixed"] == "sì"


def test_training_records_reproduce_the_gold_target(tmp_path: Path) -> None:
    annotations, _ = tool.apply_decisions(_drafts(), _decisions(), UNITS, reviewed=False)
    sources = {"T/1": TEXT_A, "T/2": TEXT_B}
    records = tool.training_records(annotations, sources)
    path = tool.write_records_yaml(records, tmp_path / "bcb.yaml")
    read = read_annotations(path)
    assert [r.prompt for r in read] == [TEXT_A, TEXT_B]
    for raw, annotation in zip(read, annotations, strict=True):
        assert serialise_target(assemble(raw.leaves)) == gold_target(annotation["spans"])
    assert path.read_bytes().count(b"\r") == 0


def test_training_records_refuse_a_changed_source() -> None:
    annotations, _ = tool.apply_decisions(_drafts(), _decisions(), UNITS, reviewed=False)
    with pytest.raises(ValueError, match="source"):
        tool.training_records(annotations, {"T/1": TEXT_A + " ", "T/2": TEXT_B})


def test_r4_split_keeps_r3_and_appends_the_new_records() -> None:
    r3 = Split(train=[0, 2, 3, 5], val=[1], test=[4])
    split = tool.r4_split(r3, base=7, tasks=["T/1", "T/2", "T/3"], validation={"T/2"})
    assert split.train == [0, 2, 3, 5, 7, 9] and split.val == [1, 8] and split.test == [4]


def test_r4_split_refuses_overlap_with_r3() -> None:
    with pytest.raises(ValueError, match="base"):
        tool.r4_split(
            Split(train=[0, 9], val=[1], test=[2]), base=5, tasks=["T/1"], validation=set()
        )


def test_saved_r4_split_binds_the_records(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(json.dumps({"index": i, "prompt": "p", "target": "{}"}) + "\n" for i in range(10)),
        encoding="utf-8",
        newline="\n",
    )
    split = tool.r4_split(
        Split(train=[0, 2], val=[1], test=[3]), base=4, tasks=["a", "b"], validation={"b"}
    )
    save_split(split, tmp_path / "splits", records_path=records)
    assert load_split(tmp_path / "splits", records_path=records) == split
