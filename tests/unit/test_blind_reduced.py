"""Reduced blind review (plan of 18/11/2026, P-06, C-18): only the U and targeted V cases.

The reduced form is the part of the full draw drawn for U and V_targeted: same
seed, same identifiers, same order. The review can start with it and extend to
the full form if time allows, without a new draw.
"""

import json
from pathlib import Path

import pytest
from test_blind_review import details
from test_covariates import source
from typer.testing import CliRunner

from pstparser.empirical.benchmarks import write_sources
from pstparser.empirical.blind import FORMS, REDUCED_REASONS, draw_cases
from pstparser.empirical.cli import app


def test_the_reduced_form_is_the_u_and_targeted_v_part_of_the_full_draw() -> None:
    full = draw_cases(details(), seed=11)
    reduced = draw_cases(details(), seed=11, form="reduced")
    assert FORMS == ("full", "reduced") and REDUCED_REASONS == ("U", "V_targeted")
    expected = [
        {**case, "reasons": [r for r in case["reasons"] if r in REDUCED_REASONS]}
        for case in full
        if set(case["reasons"]) & set(REDUCED_REASONS)
    ]
    assert reduced == expected
    reasons = [reason for case in reduced for reason in case["reasons"]]
    assert reasons.count("U") == 10 and reasons.count("V_targeted") == 40
    assert set(reasons) == {"U", "V_targeted"}


def test_an_unknown_form_is_refused() -> None:
    with pytest.raises(ValueError, match="form"):
        draw_cases(details(), seed=11, form="half")


def test_the_command_records_the_form(tmp_path: Path) -> None:
    rows = details()
    path = tmp_path / "details.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    journal = tmp_path / "generations.jsonl"
    journal.write_text(
        "".join(
            json.dumps({"request_id": r, "raw_output": "```python\nx = 1\n```"}) + "\n"
            for r in sorted({r["request_id"] for r in rows})
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
    arguments = ["blind-sample", str(path), "--journal", str(journal)]
    arguments += ["--sources", str(tmp_path / "sources.jsonl"), "--seed", "11"]
    result = CliRunner().invoke(app, [*arguments, "--form", "reduced", "--out", str(out)])
    assert result.exit_code == 0, result.exception
    manifest = json.loads((out / "empirical-blind-sample_manifest.json").read_text("utf-8"))
    assert manifest["form"] == "reduced" and manifest["cases"] == 50
