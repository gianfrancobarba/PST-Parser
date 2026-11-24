"""The parser gate reads the primary strict policy fixed by the C-22 rule.

auto-strict-v2 is accepted only when the C-22 rule, read on the v1 conversion of
the pilot tasks, makes it primary; a v1 conversion is refused when the
rule makes v2 primary. A final gate needs that document, listed in the freeze.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from test_gate_binding import TASKS, TEXTS, THRESHOLDS, convert, leaves, score, write_inputs
from typer.testing import CliRunner

from pstparser.empirical.auto import STRICT, STRICT_V2, auto_annotate, parser_gate
from pstparser.empirical.cli import app as empirical_app


def c22(share: float | None) -> dict[str, Any]:
    return {"format": "s-parse-v1", "policy": STRICT, "starting_code_share_of_failures": share}


def inputs(policy: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        {"index": i, "prompt": text, "target": "{}", "prediction": leaves(main_instruction=[text])}
        for i, text in enumerate(TEXTS)
    ]
    sources = [{"task_id": t, "text": x} for t, x in zip(TASKS, TEXTS, strict=True)]
    s_parse = auto_annotate(
        rows,
        {"tasks": TASKS},
        sources,
        policy=policy,
        inputs={"predictions_sha256": "p" * 64, "id_map_sha256": "m" * 64},
    ).s_parse
    score_report = {
        "total": 2,
        "field_f1": {"prompt.main_instruction": {"f1": 1.0, "support": 4}},
        "provenance": {"predictions_sha256": "p" * 64, "indices": [0, 1]},
    }
    return s_parse, score_report


def run(policy: str, rule: dict[str, Any] | None) -> dict[str, Any]:
    s_parse, score_report = inputs(policy)
    return parser_gate(
        s_parse,
        score_report,
        THRESHOLDS,
        id_map={"tasks": TASKS},
        id_map_sha256="m" * 64,
        sample_tasks=TASKS,
        c22=rule,
    )


def test_v2_is_gated_when_the_c22_rule_makes_it_primary() -> None:
    report = run(STRICT_V2, c22(0.8))
    assert report["status"] == "pass"
    assert report["policy"] == STRICT_V2
    assert report["c22_primary_policy"] == STRICT_V2


@pytest.mark.parametrize("rule", [None, c22(0.3), c22(None)])
def test_v2_without_the_rule_in_its_favour_is_refused(rule) -> None:
    with pytest.raises(ValueError, match="C-22"):
        run(STRICT_V2, rule)


def test_v1_is_refused_when_the_rule_makes_v2_primary() -> None:
    with pytest.raises(ValueError, match="C-22"):
        run(STRICT, c22(0.8))


def test_v1_is_gated_with_or_without_the_rule() -> None:
    assert run(STRICT, None)["policy"] == STRICT
    assert run(STRICT, c22(0.5))["c22_primary_policy"] == STRICT


def test_a_final_gate_needs_the_c22_document(tmp_path: Path, monkeypatch) -> None:
    from pstparser.empirical import cli

    paths = write_inputs(tmp_path / "in")
    results = score(paths, tmp_path / "score")
    s_parse = convert(paths, tmp_path / "hauto")
    root = tmp_path / "repo"
    s_gold = root / "protocol" / "samples" / cli.S_GOLD_FILE
    s_gold.parent.mkdir(parents=True)
    s_gold.write_bytes(paths["sample"].read_bytes())
    monkeypatch.setattr(cli, "_purpose", lambda purpose, path: {"purpose": purpose})
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(THRESHOLDS), encoding="utf-8")
    arguments = [
        "parser-gate",
        str(s_parse),
        str(results),
        "--thresholds",
        str(thresholds),
        "--id-map",
        str(paths["id_map"]),
        "--sample",
        str(paths["sample"]),
        "--out",
        str(tmp_path / "g" / "gate.json"),
        "--purpose",
        "final",
        "--freeze",
        str(root / "protocol" / "protocol-freeze.json"),
    ]
    result = CliRunner().invoke(empirical_app, arguments)
    assert isinstance(result.exception, ValueError)
    assert "--c22" in str(result.exception)
