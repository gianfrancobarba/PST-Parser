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
