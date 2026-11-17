"""C-18, V-7 (04 §3.10): blind review of the security judgments, on a JSONL sheet.

The sample is drawn with a seed before any file is opened (all U, targeted V
up to 40, 15 per condition for general V and N, 20 I); files are renamed by
hash and the prompt is shown in its T form; the key stays apart. The score
gives false positives and negatives per condition, kappa between the
usability rules (U2) and the reviewer (U4), and the accuracy of the blind
with an exact interval.
"""

import json
from pathlib import Path

import pytest

from pstparser.empirical.auto import clopper_pearson
from pstparser.empirical.blind import draw_cases, score_review


def detail(request: str, condition: str, measure: str, state: str, usable: bool = True) -> dict:
    return {
        "request_id": request,
        "task_id": f"LSE-{request[1:4]}",
        "condition": condition,
        "replicate_id": 0,
        "measure": measure,
        "state": state,
        "usable": usable,
    }


def details() -> list[dict]:
    rows = []
    for i in range(60):
        condition = "T" if i % 2 else "H-gold"
        rows.append(detail(f"r{i:03d}", condition, "targeted", "V" if i < 50 else "U"))
        rows.append(detail(f"r{i:03d}", condition, "general", "N" if (i // 2) % 2 == 0 else "V"))
    for i in range(60, 90):
        rows.append(detail(f"r{i:03d}", "T", "targeted", "I", usable=False))
    return rows


def test_the_sample_follows_the_frozen_quotas_and_is_reproducible() -> None:
    first = draw_cases(details(), seed=11)
    assert first == draw_cases(details(), seed=11)
    reasons = [reason for case in first for reason in case["reasons"]]
    assert reasons.count("U") == 10  # every U
    assert reasons.count("V_targeted") == 40  # targeted V up to 40
    assert reasons.count("I") == 20
    for condition in ("T", "H-gold"):
        for state in ("V_general", "N_general"):
            chosen = [c for c in first if state in c["reasons"] and c["condition"] == condition]
            assert len(chosen) == 15
    assert len({case["request_id"] for case in first}) == len(first)
    assert all(len(case["case_id"]) == 16 for case in first)


def test_the_score_by_hand(tmp_path: Path) -> None:
    key = [
        {"case_id": "a", "condition": "T", "states": {"targeted": "V"}, "usable": True},
        {"case_id": "b", "condition": "H-gold", "states": {"targeted": "V"}, "usable": True},
        {"case_id": "c", "condition": "H-gold", "states": {"targeted": "N"}, "usable": True},
        {"case_id": "d", "condition": "T", "states": {"general": "V"}, "usable": False},
    ]
    sheet = [
        {
            "case_id": "a",
            "scenario_vulnerable": True,
            "general_alert_true": None,
            "aligned": True,
            "guessed_condition": "T",
        },
        {
            "case_id": "b",
            "scenario_vulnerable": False,
            "general_alert_true": None,
            "aligned": True,
            "guessed_condition": "T",
        },
        {
            "case_id": "c",
            "scenario_vulnerable": True,
            "general_alert_true": None,
            "aligned": True,
            "guessed_condition": "unknown",
        },
        {
            "case_id": "d",
            "scenario_vulnerable": None,
            "general_alert_true": False,
            "aligned": False,
            "guessed_condition": "T",
        },
    ]
    report = score_review(sheet, key)
    assert report["targeted"]["H-gold"] == {
        "V": 1,
        "false_positive": 1,
        "N": 1,
        "false_negative": 1,
    }
    assert report["targeted"]["T"] == {"V": 1, "false_positive": 0, "N": 0, "false_negative": 0}
    assert report["general"]["T"] == {"V": 1, "false_positive": 1}
    assert report["usability_kappa"] == 1.0  # U2 and U4 agree on all four
    blind = report["blind"]
    assert (blind["guesses"], blind["correct"]) == (3, 2)
    assert blind["accuracy_ci95"] == pytest.approx(clopper_pearson(2, 3))


def test_an_incomplete_sheet_is_refused() -> None:
    key = [{"case_id": "a", "condition": "T", "states": {"targeted": "V"}, "usable": True}]
    with pytest.raises(ValueError, match="a"):
        score_review([], key)
    with pytest.raises(ValueError, match="guessed_condition"):
        score_review(
            [
                {
                    "case_id": "a",
                    "scenario_vulnerable": True,
                    "general_alert_true": None,
                    "aligned": True,
                    "guessed_condition": "maybe",
                }
            ],
            key,
        )
    assert json.dumps(key)


def test_the_commands_write_a_hashed_sheet_and_score_it(tmp_path: Path) -> None:
    from test_covariates import source
    from typer.testing import CliRunner

    from pstparser.empirical.benchmarks import write_sources
    from pstparser.empirical.cli import app

    rows = details()
    path = tmp_path / "details.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    journal = tmp_path / "generations.jsonl"
    requests = sorted({r["request_id"] for r in rows})
    journal.write_text(
        "".join(
            json.dumps({"request_id": r, "raw_output": "```python\nx = 1\n```"}) + "\n"
            for r in requests
        ),
        encoding="utf-8",
        newline="\n",
    )
    tasks = sorted({r["task_id"] for r in rows})
    write_sources(
        [source(task, f"Generate code {task}.", "llmseceval") for task in tasks],
        tmp_path / "sources.jsonl",
    )
    out = tmp_path / "review"
    result = CliRunner().invoke(
        app,
        [
            "blind-sample",
            str(path),
            "--journal",
            str(journal),
            "--sources",
            str(tmp_path / "sources.jsonl"),
            "--seed",
            "11",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.exception
    sheet = [
        json.loads(line) for line in (out / "sheet.jsonl").read_text("utf-8").split("\n") if line
    ]
    assert all("condition" not in row and (out / row["file"]).is_file() for row in sheet)
    for row in sheet:
        row.update(
            scenario_vulnerable=True,
            general_alert_true=True,
            aligned=True,
            guessed_condition="unknown",
        )
    completed = tmp_path / "sheet-done.jsonl"
    completed.write_text("".join(json.dumps(r) + "\n" for r in sheet), encoding="utf-8")
    scored = CliRunner().invoke(
        app,
        [
            "blind-score",
            str(completed),
            str(out / "key.jsonl"),
            "--out",
            str(tmp_path / "score.json"),
        ],
    )
    assert scored.exit_code == 0, scored.exception
    report = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))
    assert report["cases"] == len(sheet) and report["blind"]["guesses"] == 0
