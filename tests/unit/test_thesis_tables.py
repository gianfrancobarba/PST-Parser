"""Thesis tables and figures (M20): regenerated from run files, identical across executions."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    analyze_paired,
    write_outcomes,
)
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants
from pstparser.empirical.tables import (
    Provenance,
    Table,
    leaf_count,
    paired_tables,
    security_tables,
    to_csv,
    to_latex,
    tokens_tables,
    treatment_tables,
)

RUN_ID = "ds-test-01"
# Ten greedy pairs: both pass 4, T only 1, H-gold only 3, both fail 2 (b = 1, c = 3).
PAIRS = [(1, 1)] * 4 + [(1, 0)] + [(0, 1)] * 3 + [(0, 0)] * 2


def prompt(task: str, leaves: int) -> CanonicalPrompt:
    """A prompt with ``leaves`` sentences, each in a different leaf."""
    paths = ["main_instruction", "context.format", "context.constraints"][:leaves]
    sentences = [f"Sentence {i} of {task}." for i in range(leaves)]
    text = " ".join(sentences)
    spans, cursor = [], 0
    for path, sentence in zip(paths, sentences, strict=True):
        start = text.index(sentence, cursor)
        spans.append(SpanAnnotation(path, start, start + len(sentence), sentence))
        cursor = start + len(sentence)
    return CanonicalPrompt(task, text, tuple(spans), "gold-v2.0")


def variant_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(len(PAIRS)):
        canonical = prompt(f"BigCodeBench/{index}", (1, 2, 3)[index % 3])
        rows += [
            v.as_dict()
            for v in (*render_prompt_variants(canonical), render_envelope_variant(canonical))
        ]
    return rows


def journal_rows() -> list[dict[str, Any]]:
    tokens = {"BigCodeBench/0": (100, 250, 120), "BigCodeBench/1": (200, 300, 240)}
    rows = []
    for task, values in tokens.items():
        for condition, value in zip(("T", "H-gold", "E"), values, strict=True):
            rows.append(
                {
                    "request_id": f"{task}:{condition}:0",
                    "task_id": task,
                    "condition": condition,
                    "replicate_id": 0,
                    "input_tokens": value,
                    "output_tokens": value // 10,
                    "finish_reason": "length"
                    if (task, condition) == ("BigCodeBench/1", "E")
                    else "stop",
                    "session_id": "s1" if task.endswith("0") else "s2",
                    "gpu_name": "NVIDIA L4",
                }
            )
    return rows


def functional_report(analysis_id: str = "bcb-h1") -> dict[str, Any]:
    tasks = tuple(TaskSpec(f"BigCodeBench/{i}") for i in range(len(PAIRS)))
    records = [
        OutcomeRecord(
            analysis_id, "functional", task.task_id, condition, 0, "pass" if value else "fail"
        )
        for task, pair in zip(tasks, PAIRS, strict=True)
        for condition, value in zip(("T", "H-gold"), pair, strict=True)
    ]
    design = AnalysisDesign(analysis_id, "functional", tasks)
    return analyze_paired(design, records, bootstrap_samples=200).as_dict()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return path


def fake_run(root: Path) -> Path:
    run = root / "results" / "downstream" / RUN_ID
    run.mkdir(parents=True)
    (run / "run-identity.json").write_text(json.dumps({"run_id": RUN_ID}), encoding="utf-8")
    write_jsonl(run / "inputs" / "bcb.variants.jsonl", variant_rows())
    write_jsonl(run / "generation" / "bcb-greedy" / "generations.jsonl", journal_rows())
    report = run / "analysis" / "bcb-h1" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(functional_report()), encoding="utf-8")
    # A failed parser gate: H1 alone is confirmatory (C-11).
    (run / "parser" / "gate").mkdir(parents=True)
    (run / "parser" / "gate" / "gate.json").write_text(
        json.dumps(gate("descriptive")), encoding="utf-8"
    )
    security = [
        OutcomeRecord("sec-general", "security", f"LSE-{i:03d}", condition, 0, state)
        for i, (t_state, h_state) in enumerate(
            [("V", "N"), ("V", "V"), ("N", "I"), ("U", "missing")]
        )
        for condition, state in (("T", t_state), ("H-gold", h_state))
    ]
    target = run / "evaluation" / "llmseceval-sampled" / "codeql-01" / "outcomes.jsonl"
    target.parent.mkdir(parents=True)
    write_outcomes(
        [
            r
            if r.outcome != "missing"
            else OutcomeRecord(**{**r.__dict__, "reason": "codeql_unavailable"})
            for r in security
        ],
        target,
    )
    protocol = root / "protocol"
    (protocol / "samples").mkdir(parents=True)
    (protocol / "samples" / "bcb-final.json").write_text(
        json.dumps(
            {
                "benchmark": "bigcodebench",
                "kind": "final-sample",
                "seed": 7319,
                "hard_count": 1,
                "tasks": [{"task_id": "BigCodeBench/0"}, {"task_id": "BigCodeBench/1"}],
            }
        ),
        encoding="utf-8",
    )
    return run


PROVENANCE = Provenance(RUN_ID, "a" * 64, (("inputs/x.jsonl", "b" * 64),))


def test_leaf_count_counts_the_non_empty_leaves() -> None:
    rows = [r for r in variant_rows() if r["condition"] == "H-gold"]
    assert [leaf_count(r["text"]) for r in rows][:4] == [1, 2, 3, 1]


def test_treatment_table_by_hand() -> None:
    distribution, summary = treatment_tables(variant_rows())
    # Leaves 1, 2, 3, 1, 2, 3, 1, 2, 3, 1: four with 1, three with 2, three with 3.
    assert summary.rows == (("H-gold", 10, 1.9, 2.0, 0.4, 0.3),)
    assert distribution.rows == (
        ("H-gold", 1, 4, 0.4),
        ("H-gold", 2, 3, 0.3),
        ("H-gold", 3, 3, 0.3),
    )


def test_tokens_table_by_hand() -> None:
    (table,) = tokens_tables(journal_rows())
    rows = {row[0]: row for row in table.rows}
    assert rows["T"] == ("T", 2, 150.0, 150.0, 100, 200, 1.0, 1.0)
    # H-gold / T: 2.5 and 1.5; E / T: 1.2 and 1.2.
    assert rows["H-gold"][6:] == (2.0, 2.0)
    assert rows["E"][6] == pytest.approx(1.2) and rows["E"][7] == pytest.approx(1.2)


def test_paired_table_reads_the_report_values() -> None:
    main, sensitivity = paired_tables({"r": functional_report()})
    (row,) = main.rows
    values = dict(zip(main.columns, row, strict=True))
    assert (values["b"], values["c"], values["included_tasks"]) == (1, 3, 10)
    assert values["delta"] == pytest.approx(0.2)
    # Mid-p of (1, 3): 2 * (P(X <= 1) - P(X = 1) / 2) with X ~ Bin(4, 1/2) = 2 * (5 - 2) / 16.
    assert values["midp_p"] == pytest.approx(0.375)
    # Exact: 2 * P(X <= 1) = 10 / 16.
    assert values["exact_p"] == pytest.approx(0.625)
    assert values["tango_low"] < 0.2 < values["tango_high"]
    assert values["decision"] == "exploratory"
    assert len(sensitivity.rows) == 7


def report_with(analysis_id: str, p: float) -> dict[str, Any]:
    return {
        **functional_report(analysis_id),
        "midp_p": p,
    }


def h3_report(p: float) -> dict[str, Any]:
    """The S_parse estimate of H3: T vs H-auto, planned on the S_parse of the gate."""
    return {**report_with("bcb-h3-s-parse", p), "conditions": ["T", "H-auto"]}


def gate(h3: str) -> dict[str, Any]:
    return {"format": "parser-gate-v1", "h3": h3, "s_parse_size": len(PAIRS)}


@pytest.mark.parametrize(
    ("h1", "h3", "expected"),
    [
        (0.035, 0.5, ("H1 rejected", "H3 not_rejected")),  # H1 at 0.04
        (0.045, 0.5, ("H1 not_rejected", "H3 not_rejected")),
        (0.045, 0.005, ("H1 rejected", "H3 rejected")),  # H3 at 0.01, H1 recycled to 0.05
        (0.5, 0.045, ("H1 not_rejected", "H3 not_rejected")),
        (0.03, 0.045, ("H1 rejected", "H3 rejected")),  # H1 at 0.04, H3 recycled to 0.05
    ],
)
def test_weighted_holm_decisions_in_the_paired_table(
    h1: float, h3: float, expected: tuple[str, str]
) -> None:
    reports = {"a": report_with("bcb-h1", h1), "b": h3_report(h3)}
    family = {"H1": "bcb-h1", "H3": "bcb-h3-s-parse"}
    (main, _) = paired_tables(reports, family, gate("confirmatory"))
    decisions = tuple(row[-1] for row in main.rows)
    assert decisions == expected


def test_without_h3_the_family_is_h1_at_five_percent() -> None:
    reports = {"a": report_with("bcb-h1", 0.045)}
    (main, _) = paired_tables(reports, {"H1": "bcb-h1"}, gate("descriptive"))
    assert main.rows[0][-1] == "H1 rejected"


def test_the_confirmatory_family_refuses_unknown_names_and_reports() -> None:
    with pytest.raises(ValueError, match="H1"):
        paired_tables({"a": functional_report()}, {"H2": "bcb-h1"}, gate("descriptive"))
    with pytest.raises(ValueError, match="no report"):
        paired_tables({"a": functional_report()}, {"H1": "other"}, gate("descriptive"))


def test_security_table_counts_states_per_condition() -> None:
    records = [
        OutcomeRecord("g", "security", "LSE-000", "T", 0, "V"),
        OutcomeRecord("g", "security", "LSE-001", "T", 0, "N"),
        OutcomeRecord("g", "security", "LSE-002", "T", 0, "missing", reason="codeql_unavailable"),
        OutcomeRecord("g", "security", "LSE-000", "H-gold", 0, "I"),
    ]
    (table,) = security_tables(records)
    assert table.rows == (("g", "T", 1, 1, 0, 0, 1, 3, 0.5), ("g", "H-gold", 0, 0, 1, 0, 0, 1, 0.0))


def test_csv_and_latex_carry_the_provenance_and_escape_latex() -> None:
    table = Table("t", ("name_1", "value"), (("a_b & c", 0.123456), ("d", None)))
    csv_text = to_csv(table, PROVENANCE, "paired")
    assert csv_text.splitlines()[:4] == [
        "# thesis-tables-v1 kind=paired table=t",
        f"# run_id={RUN_ID}",
        f"# source_sha256={'a' * 64}",
        f"# input inputs/x.jsonl sha256={'b' * 64}",
    ]
    assert csv_text.splitlines()[4:] == ["name_1,value", "a_b & c,0.123456", "d,"]
    latex = to_latex(table, PROVENANCE, "paired")
    assert r"a\_b \& c & 0.1235 \\" in latex and r"d & -- \\" in latex
    assert f"% run_id={RUN_ID}" in latex and r"\begin{tabular}{lr}" in latex


def test_rows_must_match_the_header() -> None:
    with pytest.raises(ValueError, match="cells"):
        Table("t", ("a", "b"), (("x",),))


def run_thesis(run: Path, out: Path) -> Any:
    return CliRunner().invoke(
        app,
        [
            "thesis-tables",
            "--run",
            str(run),
            "--out",
            str(out),
            "--protocol",
            str(run.parents[2] / "protocol"),
            "--confirmatory",
            "H1=bcb-h1",
        ],
    )


def test_thesis_tables_are_identical_across_two_executions(tmp_path: Path, monkeypatch) -> None:
    run = fake_run(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    first = run_thesis(run, tmp_path / "first")
    second = run_thesis(run, tmp_path / "second")
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    names = sorted(path.name for path in (tmp_path / "first").iterdir())
    assert names == sorted(path.name for path in (tmp_path / "second").iterdir())
    for name in names:
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes(), name
    expected = {
        "samples.csv",
        "treatment.csv",
        "treatment-leaves.csv",
        "bcb-greedy-tokens.csv",
        "bcb-greedy-diagnostics.csv",
        "paired.csv",
        "paired-sensitivity.csv",
        "security.csv",
        "forest.pdf",
        "forest.png",
        "bcb-h1-effects.pdf",
        "bcb-h1-leaves.png",
        "bcb-greedy-tokens.pdf",
        "empirical-tables_manifest.json",
    }
    assert expected <= set(names)
    manifest = json.loads(
        (tmp_path / "first" / "empirical-tables_manifest.json").read_text("utf-8")
    )
    assert manifest["run_id"] == RUN_ID
    assert set(manifest["skipped"]) == {"freeze", "parser", "review"}
    assert "analysis/bcb-h1/report.json" in manifest["inputs"]
    paired = (tmp_path / "first" / "paired.csv").read_text("utf-8")
    assert "H1 rejected" not in paired and "H1 not_rejected" in paired  # mid-p 0.375
    assert f"# run_id={RUN_ID}" in paired
    assert str(tmp_path) not in paired


def test_an_existing_output_directory_is_refused(tmp_path: Path, monkeypatch) -> None:
    run = fake_run(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    (tmp_path / "out").mkdir()
    result = run_thesis(run, tmp_path / "out")
    assert isinstance(result.exception, FileExistsError)


def test_a_run_without_identity_is_refused(tmp_path: Path) -> None:
    run = fake_run(tmp_path)
    (run / "run-identity.json").unlink()
    result = run_thesis(run, tmp_path / "out")
    assert isinstance(result.exception, ValueError)
    assert not (tmp_path / "out").exists()


def test_the_tables_command_writes_one_kind(tmp_path: Path, monkeypatch) -> None:
    run = fake_run(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    journal = run / "generation" / "bcb-greedy" / "generations.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "tables",
            "tokens",
            "--run",
            str(run),
            "--out",
            str(tmp_path / "t"),
            "--journal",
            str(journal),
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == [
        "empirical-tables_manifest.json",
        "tokens.csv",
        "tokens.tex",
    ]
    missing = CliRunner().invoke(
        app, ["tables", "tokens", "--run", str(run), "--out", str(tmp_path / "u")]
    )
    assert isinstance(missing.exception, ValueError) and "--journal" in str(missing.exception)


def test_the_figures_command_draws_one_kind(tmp_path: Path, monkeypatch) -> None:
    run = fake_run(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    report = run / "analysis" / "bcb-h1" / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "figures",
            "forest",
            "--run",
            str(run),
            "--out",
            str(tmp_path / "f"),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "f" / "forest.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "f" / "forest.png").read_bytes().startswith(b"\x89PNG")


def test_the_protocol_family_decides_h1_alone_through_the_command(
    tmp_path: Path, monkeypatch
) -> None:
    run = fake_run(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    family = run.parents[2] / "protocol" / "analysis" / "confirmatory-family.json"
    family.parent.mkdir(parents=True, exist_ok=True)
    family.write_text(
        json.dumps(
            {
                "format": "confirmatory-family-v1",
                "family": ["H1"],
                "alpha": 0.05,
                "h3": "exploratory",
            }
        ),
        encoding="utf-8",
    )
    result = run_thesis(run, tmp_path / "out")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "out" / "empirical-tables_manifest.json").read_text("utf-8"))
    assert any(name.endswith("confirmatory-family.json") for name in manifest["inputs"])

    family.write_text(json.dumps({"format": "parser-gate-v1"}), encoding="utf-8")
    refused = run_thesis(run, tmp_path / "out-2")
    assert isinstance(refused.exception, ValueError)
    assert "confirmatory-family-v1" in str(refused.exception)


# Finding of the dress rehearsal (19/11/2026): the exploratory contrasts E - T and
# H-gold - E are analysed on the same core ledger as H1, so their reports share
# its analysis_id; a report is identified by its analysis_id and its conditions.


def test_reports_of_one_ledger_differ_by_their_conditions() -> None:
    h1 = report_with("bcb-h1", 0.035)
    e_t = {**report_with("bcb-h1", 0.5), "conditions": ["T", "E"]}
    (main, _) = paired_tables({"a": h1, "b": e_t}, {"H1": "bcb-h1"}, gate("descriptive"))
    rows = {(row[0], row[2], row[3]): row for row in main.rows}
    assert rows[("bcb-h1", "T", "H-gold")][-1] == "H1 rejected"
    assert rows[("bcb-h1", "T", "E")][-1] == "exploratory"


def test_two_reports_with_the_same_id_and_conditions_are_refused() -> None:
    with pytest.raises(ValueError, match="share"):
        paired_tables({"a": report_with("x", 0.1), "b": report_with("x", 0.2)})
