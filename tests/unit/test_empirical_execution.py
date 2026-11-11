from pathlib import Path
from types import SimpleNamespace

import pytest

from pstparser.empirical.extraction import extract_python
from pstparser.empirical.sandbox import SandboxConfig, container_command, run_python_check
from pstparser.empirical.security import SecurityJudgment

IMAGE = "python@sha256:" + "a" * 64


@pytest.mark.parametrize(
    "raw,status",
    [
        ("", "empty"),
        ("# comment", "empty"),
        ("def bad(:", "syntax_error"),
        ("```python\ndef f():\n    return 1\n```", "ok"),
        ("def f():\n    return 1", "ok"),
        ("Here is code:\n```python\nx=1\n```", "invalid_fence"),
        ("```python\nx=1", "invalid_fence"),
        ("```\nx=1\n```\n```\ny=2\n```", "invalid_fence"),
    ],
)
def test_common_extraction_policy(raw: str, status: str) -> None:
    assert extract_python(raw).status == status


def test_security_precedence_retains_findings() -> None:
    result = SecurityJudgment(False, False, ("CWE-test",), "rubric-v1", "reviewed")
    assert result.state == "I" and result.outcome == 0
    assert result.confirmed_findings == ("CWE-test",)
    assert SecurityJudgment(True, False, ("finding",), "v1", "evidence").state == "V"
    assert SecurityJudgment(True, False, (), "v1", "not assessable").state == "U"
    assert SecurityJudgment(True, True, (), "v1", "controls completed").outcome == 1


def test_sandbox_requires_pinned_image() -> None:
    with pytest.raises(ValueError, match="pinned"):
        SandboxConfig("python:latest")


def test_string_boolean_cannot_produce_security_success() -> None:
    with pytest.raises(ValueError, match="booleans"):
        SecurityJudgment("false", True, (), "v1", "evidence")


def test_sandbox_command_has_no_host_execution_fallback(tmp_path: Path) -> None:
    command = container_command(SandboxConfig(IMAGE), tmp_path, "test-job")
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--pull") + 1] == "never"
    assert "--read-only" in command and "--cap-drop" in command
    assert command.count("--mount") == 1
    assert "readonly" in command[command.index("--mount") + 1]
    assert "--env" not in command


def test_execution_failure_classification_and_cleanup(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout="1\n" if command[1] == "wait" else "", stderr=""
        )

    monkeypatch.setattr("pstparser.empirical.sandbox.subprocess.run", run)
    result = run_python_check("x=1", "assert False", SandboxConfig(IMAGE), temporary_root=tmp_path)
    assert result.status == "failed"
    assert calls[-1][1:3] == ["rm", "--force"]
    assert list(tmp_path.iterdir()) == []


def test_unavailable_docker_is_not_a_failed_solution(monkeypatch, tmp_path: Path) -> None:
    def run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")

    monkeypatch.setattr("pstparser.empirical.sandbox.subprocess.run", run)
    result = run_python_check("x=1", "", SandboxConfig(IMAGE), temporary_root=tmp_path)
    assert result.status == "infrastructure_error"
