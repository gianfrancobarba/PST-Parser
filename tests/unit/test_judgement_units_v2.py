"""C-06 (R-TRA-01, Q-24): judgement units without the rule-fixed first sentence.

The first sentence of the description goes to ``main_instruction`` by rule
G3.1, so it holds no decision of the reviewer; the sentences of "Note that:"
(rule B6) do. Rule ``judgement-units-v2`` leaves the first out and takes the
notes in; the agreement report gives kappa per unit type and the number of
tasks without any judgement unit.
"""

import json
from dataclasses import replace
from pathlib import Path

from test_covariates import BCB, BCB_RULE, BCB_TEXT, LSE, LSE_RULE, canonical, source
from typer.testing import CliRunner

from pstparser.empirical import cli
from pstparser.empirical.annotations import propose, write_annotations
from pstparser.empirical.benchmarks import write_sources
from pstparser.empirical.contracts import SpanAnnotation
from pstparser.empirical.covariates import (
    JUDGEMENT_UNITS_RULE,
    judgement_mask,
    judgement_regions,
    judgement_units,
    unit_regions,
)

ONLY_FIRST_TEXT = (
    "Sort the list.\n"
    "The function should output with:\n    list: The sorted list.\n"
    "You should write self-contained code starting with:\n```\ndef task_func(x):\n```"
)
ONLY_FIRST = source("BigCodeBench/3", ONLY_FIRST_TEXT)


def covered(text: str, mask: tuple[bool, ...]) -> str:
    return "".join(c for c, flag in zip(text, mask, strict=True) if flag)


def test_rule_version() -> None:
    assert JUDGEMENT_UNITS_RULE == "judgement-units-v2"


def test_a_task_with_only_the_first_sentence_has_an_empty_mask() -> None:
    assert judgement_regions(ONLY_FIRST_TEXT, BCB_RULE) == []
    assert not any(judgement_mask(ONLY_FIRST_TEXT, BCB_RULE))
    assert judgement_units(canonical(ONLY_FIRST, BCB_RULE), BCB_RULE) == 0


def test_the_mask_holds_the_prose_judgements_and_the_notes() -> None:
    assert covered(BCB_TEXT, judgement_mask(BCB_TEXT, BCB_RULE)) == (
        "It must not modify the input, e.g. by sorting it."
        "The result is scaled by 2.5 afterwards."
        "Note that: Notes: The title is 'Sum'."
        "Empty lists never fail."
    )
    assert judgement_units(canonical(BCB, BCB_RULE), BCB_RULE) == 4
    # LLMSecEval: the three sentences after the first one.
    assert judgement_units(canonical(LSE, LSE_RULE), LSE_RULE) == 3


def test_unit_regions_by_type() -> None:
    regions = unit_regions(BCB_TEXT, BCB_RULE)
    assert set(regions) == {"first", "prose", "notes"}
    assert [BCB_TEXT[a:b] for a, b in regions["first"]] == ["Compute the sum of a list."]
    assert len(regions["prose"]) == 2 and len(regions["notes"]) == 2


def test_agreement_report_by_unit_type(tmp_path: Path) -> None:
    sources = tmp_path / "sources.jsonl"
    write_sources([BCB, ONLY_FIRST], sources)
    first = [propose(BCB, BCB_RULE), propose(ONLY_FIRST, BCB_RULE)]
    # The second annotator moves one note from constraints to format.
    spans = list(first[0].spans)
    index = next(i for i, s in enumerate(spans) if s.phrase == "Empty lists never fail.")
    spans[index] = SpanAnnotation(
        "context.format", *(spans[index].start, spans[index].end), spans[index].phrase
    )
    second = [
        replace(first[0], spans=tuple(spans), annotator="studente"),
        replace(first[1], annotator="studente"),
    ]
    write_annotations(first, tmp_path / "a.jsonl")
    write_annotations(second, tmp_path / "b.jsonl")
    out = tmp_path / "agree" / "agreement.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "agreement",
            str(sources),
            str(tmp_path / "a.jsonl"),
            str(tmp_path / "b.jsonl"),
            "--rule",
            BCB_RULE,
            "--first-date",
            "2026-11-21",
            "--second-date",
            "2026-11-27",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.exception
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["judgement_units_rule"] == "judgement-units-v2"
    assert report["tasks_without_judgement_units"] == 1
    assert report["by_unit_type"]["first"]["char_kappa"] == 1.0
    assert report["by_unit_type"]["prose"]["char_kappa"] == 1.0
    assert report["by_unit_type"]["notes"]["char_kappa"] < 1.0
    assert report["judgement_units"]["char_kappa"] < 1.0
    assert report["interval_days"] == 6
