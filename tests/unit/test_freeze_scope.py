"""The freeze covers every file of the repository and is required downstream.

Counterexamples of the independent review, inverted: a freeze used to verify
after changes to the gold, the notebook, the parser prompt, configurations and
``pyproject.toml``, and admitted unlisted names at any depth of ``protocol/``.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import OutcomeRecord, write_outcomes
from pstparser.empirical.cli import app
from pstparser.empirical.freeze import build_freeze, load_freeze, verify_freeze, write_freeze

META = {"protocol_version": "v", "frozen_by": "x"}
FILES = {
    "src/pstparser/a.py": "x = 1\n",
    "uv.lock": "lock\n",
    "pyproject.toml": "[project]\ndependencies=['transformers>=4.51.3,<4.58']\n",
    "protocol/freeze-spec.yaml": "spec\n",
    "protocol/samples/s.json": "{}\n",
    "data/annotated/downstream/bcb-final/annotations.jsonl": '{"status":"draft"}\n',
    "notebooks/downstream.ipynb": "{}\n",
    "prompts/system_pst.md": "parser system prompt\n",
    "configs/experiments/baseline.yaml": "max_new_tokens: 3072\n",
    ".gitignore": "outputs/\n",
}
ITEMS = ["protocol/freeze-spec.yaml", "protocol/samples/s.json"]


def make_repo(root: Path, git: bool) -> Path:
    for relative, text in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    if git:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    document = build_freeze(root, ITEMS, META, frozen_at="2026-11-17T00:00:00Z")
    out = write_freeze(document, root / "protocol" / "protocol-freeze.json")
    assert verify_freeze(out) == []
    return out


def found(out: Path) -> set[tuple[str, str]]:
    return {(m.kind, m.path) for m in verify_freeze(out)}


def test_the_freeze_lists_every_file_of_the_repository(tmp_path: Path) -> None:
    document = load_freeze(make_repo(tmp_path, git=False))
    assert document["format"] == "protocol-freeze-v2"
    assert set(document["repository"]["files"]) >= {
        "pyproject.toml",
        "data/annotated/downstream/bcb-final/annotations.jsonl",
        "notebooks/downstream.ipynb",
        "prompts/system_pst.md",
        "configs/experiments/baseline.yaml",
    }


def test_changes_outside_the_listed_items_are_found(tmp_path: Path) -> None:
    out = make_repo(tmp_path, git=False)
    changed = [
        "pyproject.toml",
        "data/annotated/downstream/bcb-final/annotations.jsonl",
        "notebooks/downstream.ipynb",
        "prompts/system_pst.md",
        "configs/experiments/baseline.yaml",
    ]
    for relative in changed:
        (tmp_path / relative).write_text("changed\n", encoding="utf-8")
    assert found(out) == {("changed", relative) for relative in changed}


def test_unlisted_names_are_allowed_only_at_the_root_of_protocol(tmp_path: Path) -> None:
    out = make_repo(tmp_path, git=False)
    refused = (
        "protocol/samples/deviations.md",
        "protocol/samples/protocol-freeze-old.json",
        "protocol/samples/FREEZE-notes.md",
        "protocol/samples/acks/thresholds.json",
    )
    for relative in (*refused, "protocol/deviations.md", "protocol/acks/reply.eml"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    assert found(out) == {("unlisted", relative) for relative in refused}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_new_file_of_a_git_checkout_is_found_but_results_and_ignored_files_are_not(
    tmp_path: Path,
) -> None:
    out = make_repo(tmp_path, git=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "verify_archive.py").write_text("print('new')\n", encoding="utf-8")
    (tmp_path / "results" / "run").mkdir(parents=True)
    (tmp_path / "results" / "run" / "report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert found(out) == {("unlisted", "scripts/verify_archive.py")}


def _analysis_inputs(root: Path) -> list[str]:
    records = [
        OutcomeRecord("core", "functional", f"BigCodeBench/{i}", condition, 0, outcome)
        for i in range(4)
        for condition, outcome in (("T", "fail"), ("H-gold", "pass" if i % 2 else "fail"))
    ]
    write_outcomes(records, root / "outcomes.jsonl")
    design = {
        "analysis_id": "core",
        "endpoint": "functional",
        "tasks": [{"task_id": f"BigCodeBench/{i}"} for i in range(4)],
    }
    (root / "design.json").write_text(json.dumps(design), encoding="utf-8")
    return ["analyze", str(root / "design.json"), str(root / "outcomes.jsonl")]


def test_a_final_analysis_requires_a_verified_freeze(tmp_path: Path) -> None:
    arguments = _analysis_inputs(tmp_path)
    final = ["--purpose", "final"]
    refused = CliRunner().invoke(app, [*arguments, str(tmp_path / "a" / "report.json"), *final])
    assert isinstance(refused.exception, ValueError)
    assert "freeze" in str(refused.exception)
    out = make_repo(tmp_path / "repo", git=False)
    with_freeze = [*final, "--freeze", str(out)]
    (tmp_path / "repo" / "pyproject.toml").write_text("changed\n", encoding="utf-8")
    stale = CliRunner().invoke(app, [*arguments, str(tmp_path / "b" / "r.json"), *with_freeze])
    assert isinstance(stale.exception, ValueError)
    assert "does not verify" in str(stale.exception)
    (tmp_path / "repo" / "pyproject.toml").write_text(
        FILES["pyproject.toml"], encoding="utf-8", newline="\n"
    )
    accepted = CliRunner().invoke(app, [*arguments, str(tmp_path / "c" / "r.json"), *with_freeze])
    assert accepted.exit_code == 0, accepted.exception
    manifest = json.loads((tmp_path / "c" / "empirical-analysis_manifest.json").read_text("utf-8"))
    assert manifest["purpose"] == "final"
    assert len(manifest["freeze_sha256"]) == 64
    exploratory = CliRunner().invoke(app, [*arguments, str(tmp_path / "d" / "r.json")])
    assert exploratory.exit_code == 0, exploratory.exception
    manifest = json.loads((tmp_path / "d" / "empirical-analysis_manifest.json").read_text("utf-8"))
    assert (manifest["purpose"], manifest["freeze_sha256"]) == ("exploratory", None)


@pytest.mark.parametrize("command", ["thesis-tables", "parser-gate", "render"])
def test_other_downstream_commands_refuse_final_without_a_freeze(command: str, tmp_path) -> None:
    arguments = {
        "thesis-tables": ["thesis-tables", "--run", str(tmp_path), "--out", str(tmp_path / "t")],
        "parser-gate": [
            "parser-gate",
            str(tmp_path / "s.json"),
            str(tmp_path / "score.json"),
            "--thresholds",
            str(tmp_path / "th.json"),
            "--id-map",
            str(tmp_path / "id-map.json"),
            "--sample",
            str(tmp_path / "sample.json"),
            "--out",
            str(tmp_path / "g" / "gate.json"),
        ],
        "render": ["render", str(tmp_path / "canonical.jsonl"), str(tmp_path / "v.jsonl")],
    }[command]
    result = CliRunner().invoke(app, [*arguments, "--purpose", "final"])
    assert isinstance(result.exception, ValueError)
    assert "freeze" in str(result.exception)
