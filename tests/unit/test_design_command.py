"""C-14 (R-STA-04): analysis designs from frozen samples, thesis tables from final runs.

Before the correction the design was a hand-written JSON, a security design
without families silently ran a per-task sign-flip, and ``thesis-tables`` read
any run, pilot included.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import AnalysisDesign, OutcomeRecord, TaskSpec, analyze_paired
from pstparser.empirical.cli import app
from pstparser.repro.manifest import digest_file


def write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    return path


def bcb_sample(root: Path) -> Path:
    tasks = [{"position": i, "task_id": f"BigCodeBench/{n}"} for i, n in enumerate((458, 3, 77))]
    return write(root / "bcb-final.json", {"kind": "final-sample", "n": 3, "tasks": tasks})


def lse_map(root: Path) -> Path:
    records = [
        {"task_id": "LSE-000", "family_id": "119-BOF-1", "selections": ["S2"]},
        {"task_id": "LSE-001", "family_id": "119-BOF-1", "selections": ["S2"]},
        {"task_id": "LSE-002", "family_id": "20-IIV-1", "selections": ["S2"]},
        {"task_id": "LSE-003", "family_id": "20-IIV-1", "selections": []},
    ]
    return write(root / "llmseceval-ids.json", {"records": records})


def run_design(*arguments: str) -> object:
    return CliRunner().invoke(app, ["design", *arguments])


def test_a_functional_design_from_the_frozen_sample(tmp_path: Path) -> None:
    sample = bcb_sample(tmp_path)
    out = tmp_path / "d" / "design.json"
    result = run_design(
        str(sample),
        "--analysis-id",
        "bcb-h1",
        "--endpoint",
        "functional",
        "--conditions",
        "T,H-gold",
        "--out",
        str(out),
    )
    assert result.exit_code == 0, result.exception
    design = json.loads(out.read_text(encoding="utf-8"))
    assert design == {
        "analysis_id": "bcb-h1",
        "endpoint": "functional",
        "tasks": [
            {"task_id": t} for t in ("BigCodeBench/458", "BigCodeBench/3", "BigCodeBench/77")
        ],
        "replicate_ids": [0],
        "conditions": ["T", "H-gold"],
    }
    manifest = json.loads((out.parent / "empirical-design_manifest.json").read_text("utf-8"))
    assert manifest["sample_sha256"] == digest_file(sample)
    assert manifest["design_sha256"] == digest_file(out)


def test_a_security_design_carries_the_families_of_the_selection(tmp_path: Path) -> None:
    out = tmp_path / "s" / "design.json"
    result = run_design(
        str(lse_map(tmp_path)),
        "--analysis-id",
        "sec-general",
        "--endpoint",
        "security",
        "--conditions",
        "T,H-gold",
        "--selection",
        "S2",
        "--replicates",
        "0,1,2,3,4,5",
        "--out",
        str(out),
    )
    assert result.exit_code == 0, result.exception
    design = json.loads(out.read_text(encoding="utf-8"))
    assert design["tasks"] == [
        {"task_id": "LSE-000", "family_id": "119-BOF-1"},
        {"task_id": "LSE-001", "family_id": "119-BOF-1"},
        {"task_id": "LSE-002", "family_id": "20-IIV-1"},
    ]
    assert design["replicate_ids"] == [0, 1, 2, 3, 4, 5]


def test_a_functional_sample_cannot_make_a_security_design(tmp_path: Path) -> None:
    result = run_design(
        str(bcb_sample(tmp_path)),
        "--analysis-id",
        "x",
        "--endpoint",
        "security",
        "--conditions",
        "T,H-gold",
        "--out",
        str(tmp_path / "x" / "design.json"),
    )
    assert isinstance(result.exception, ValueError)


def test_a_security_analysis_without_families_is_refused() -> None:
    tasks = tuple(TaskSpec(f"LSE-{i:03d}") for i in range(3))
    records = [
        OutcomeRecord("sec", "security", task.task_id, condition, 0, "N")
        for task in tasks
        for condition in ("T", "H-gold")
    ]
    with pytest.raises(ValueError, match="famil"):
        analyze_paired(AnalysisDesign("sec", "security", tasks), records, bootstrap_samples=50)


def test_final_thesis_tables_refuse_an_exploratory_analysis(tmp_path: Path) -> None:
    from test_freeze_scope import make_repo

    freeze = make_repo(tmp_path / "repo", git=False)
    run = tmp_path / "run"
    write(run / "run-identity.json", {"run_id": "r"})
    write(
        run / "analysis" / "a" / "empirical-analysis_manifest.json",
        {"stage": "empirical-analysis", "purpose": "exploratory", "freeze_sha256": None},
    )
    result = CliRunner().invoke(
        app,
        [
            "thesis-tables",
            "--run",
            str(run),
            "--out",
            str(tmp_path / "t"),
            "--purpose",
            "final",
            "--freeze",
            str(freeze),
        ],
    )
    assert isinstance(result.exception, ValueError)
    assert "final" in str(result.exception)


# Finding of the dress rehearsal (19/11/2026): H3 is analysed on S_parse, whose
# document lists the tasks as strings; the design reads it through load_s_parse.


def s_parse_document(root: Path, tasks: list[str], tamper: bool = False) -> Path:
    from pstparser.empirical.auto import S_PARSE_FORMAT, tasks_digest

    digest = tasks_digest(tasks)
    return write(
        root / "s-parse.json",
        {
            "format": S_PARSE_FORMAT,
            "policy": "auto-strict-v1",
            "tasks": tasks,
            "tasks_sha256": "0" * 64 if tamper else digest,
        },
    )


def test_a_functional_design_on_s_parse_keeps_its_order(tmp_path: Path) -> None:
    tasks = ["BigCodeBench/77", "BigCodeBench/3"]
    out = tmp_path / "d" / "design.json"
    result = run_design(
        str(s_parse_document(tmp_path, tasks)),
        "--analysis-id",
        "final-bcb-s-parse",
        "--endpoint",
        "functional",
        "--conditions",
        "T,H-auto",
        "--out",
        str(out),
    )
    assert result.exit_code == 0, result.exception
    design = json.loads(out.read_text(encoding="utf-8"))
    assert design["tasks"] == [{"task_id": t} for t in tasks]
    assert design["conditions"] == ["T", "H-auto"]


def test_a_tampered_s_parse_gives_no_design(tmp_path: Path) -> None:
    out = tmp_path / "d" / "design.json"
    result = run_design(
        str(s_parse_document(tmp_path, ["BigCodeBench/3"], tamper=True)),
        "--analysis-id",
        "x",
        "--endpoint",
        "functional",
        "--conditions",
        "T,H-auto",
        "--out",
        str(out),
    )
    assert result.exit_code != 0 and not out.exists()


# Finding of the dress rehearsal (19/11/2026): the targeted measure judges only the
# families in the calibrated states, so its design keeps only their prompts.


def cards_file(root: Path) -> Path:
    path = root / "cards.yaml"
    path.write_text(
        "version: scenario-cards-v1\n"
        "query_map: qmap-v1\n"
        "families:\n"
        "  - {family_id: 119-BOF-1, cwe: CWE-119, target_rules: [r/a], state: calibrated}\n"
        "  - {family_id: 20-IIV-1, cwe: CWE-20, target_rules: [r/b],"
        " state: calibrated-with-false-positives}\n",
        encoding="utf-8",
    )
    return path


def security_design(root: Path, *extra: str) -> object:
    return run_design(
        str(lse_map(root)),
        "--analysis-id",
        "lse-targeted",
        "--endpoint",
        "security",
        "--conditions",
        "T,H-gold",
        "--selection",
        "S2",
        "--out",
        str(root / "d" / "design.json"),
        *extra,
    )


def test_the_targeted_design_keeps_the_calibrated_families(tmp_path: Path) -> None:
    result = security_design(tmp_path, "--cards", str(cards_file(tmp_path)))
    assert result.exit_code == 0, result.exception
    design = json.loads((tmp_path / "d" / "design.json").read_text(encoding="utf-8"))
    assert [t["task_id"] for t in design["tasks"]] == ["LSE-000", "LSE-001"]


def test_the_targeted_sensitivity_adds_the_families_with_false_positives(tmp_path: Path) -> None:
    result = security_design(tmp_path, "--cards", str(cards_file(tmp_path)), "--sensitivity")
    assert result.exit_code == 0, result.exception
    design = json.loads((tmp_path / "d" / "design.json").read_text(encoding="utf-8"))
    assert [t["task_id"] for t in design["tasks"]] == ["LSE-000", "LSE-001", "LSE-002"]


def test_the_sensitivity_needs_the_cards(tmp_path: Path) -> None:
    assert security_design(tmp_path, "--sensitivity").exit_code != 0
