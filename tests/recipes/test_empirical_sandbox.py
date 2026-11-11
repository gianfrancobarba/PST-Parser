"""Opt-in checks against a real local Docker engine and an already installed image."""

import json
import os
from pathlib import Path

import pytest

from pstparser.empirical.sandbox import SandboxConfig, run_python_check

pytestmark = pytest.mark.slow


@pytest.fixture
def sandbox_config():
    image = os.environ.get("PST_SANDBOX_IMAGE")
    if not image:
        pytest.skip("set PST_SANDBOX_IMAGE to an already installed immutable Python image")
    return SandboxConfig(image, timeout_seconds=3)


@pytest.mark.parametrize(
    "code,status",
    [
        ("def add(a,b): return a+b", "passed"),
        ("def add(a,b): return a-b", "failed"),
        ("import sys; sys.exit(0)", "failed"),
        ("import os; os._exit(0)", "failed"),
        ("while True: pass", "timeout"),
    ],
)
def test_real_execution(code, status, sandbox_config, tmp_path: Path) -> None:
    result = run_python_check(
        code,
        "from candidate import add\nassert add(2,3)==5",
        sandbox_config,
        temporary_root=tmp_path,
    )
    assert result.status == status, result


def test_container_is_unprivileged_and_job_is_read_only(sandbox_config, tmp_path: Path) -> None:
    code = """import os
from pathlib import Path
assert os.getuid() == 65534
assert 'HF_TOKEN' not in os.environ
assert not Path('/var/run/docker.sock').exists()
try:
    Path('/work/forbidden').write_text('x')
except OSError:
    pass
else:
    raise AssertionError('job mount is writable')
"""
    result = run_python_check(code, "import candidate", sandbox_config, temporary_root=tmp_path)
    assert result.status == "passed", result


def test_artifact_pipeline_without_model_or_real_benchmark(sandbox_config, tmp_path: Path) -> None:
    """Exercise render/journal/calibration/check/analysis with explicit toy outputs."""
    from typer.testing import CliRunner

    from pstparser.empirical.cli import app
    from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
    from pstparser.empirical.journal import request_plan, run_journal
    from pstparser.empirical.rendering import render_prompt_variants

    variants = []
    for task in ("toy-a", "toy-b"):
        text = "Write add(a, b) returning the sum."
        variants.extend(
            render_prompt_variants(
                CanonicalPrompt(
                    task,
                    text,
                    (SpanAnnotation("main_instruction", 0, len(text), text),),
                    "toy-v1",
                )
            )
        )
    requests = request_plan(variants, [0], 42)

    def synthetic_completion(request):
        wrong = request["task_id"] == "toy-b" and request["condition"] == "T"
        return {
            "raw_output": "def add(a,b): return a-b" if wrong else "def add(a,b): return a+b",
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "eos",
        }

    generated = tmp_path / "generated"
    run_journal(
        requests, {"purpose": "synthetic integration fixture"}, generated, synthetic_completion
    )
    oracles = tmp_path / "oracles.jsonl"
    oracles.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task,
                    "version": "toy-v1",
                    "harness": "from candidate import add\nassert add(2,3)==5",
                    "positive_control": "def add(a,b): return a+b",
                    "negative_control": "def add(a,b): return a-b",
                }
            )
            for task in ("toy-a", "toy-b")
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    checked = tmp_path / "checked"
    result = runner.invoke(
        app,
        [
            "check",
            str(generated / "generations.jsonl"),
            str(oracles),
            str(checked),
            sandbox_config.image,
            "toy-study",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert (checked / "empirical-check_manifest.json").exists()
    design = tmp_path / "design.json"
    design.write_text(
        json.dumps(
            {
                "analysis_id": "toy-study",
                "endpoint": "functional",
                "tasks": [{"task_id": task} for task in ("toy-a", "toy-b")],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "analysis" / "report.json"
    result = runner.invoke(
        app, ["analyze", str(design), str(checked / "outcomes.jsonl"), str(report)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(report.read_text())["delta"] == 0.5
    assert (report.parent / "empirical-analysis_manifest.json").exists()
