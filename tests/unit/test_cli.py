"""Command line behaviour that does not need a model."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.cli.main import app
from pstparser.data.predictions import PredictionRecord, write_predictions

CONFIG = Path("tests/assets/configs/valid.yaml")


@pytest.fixture
def historical(tmp_path: Path) -> Path:
    """A predictions file standing where historical results are kept."""
    path = tmp_path / "results" / "predictions.jsonl"
    record = PredictionRecord(index=0, prompt="Write a poem.", target="{}", prediction="{}")
    write_predictions([record], path)
    return path


@pytest.mark.parametrize(
    ("command", "produced"),
    [("score", "results.json"), ("align", "alignments.jsonl")],
)
def test_default_destination_is_under_the_run_root(
    command: str, produced: str, historical: Path, tmp_path: Path
) -> None:
    runs = tmp_path / "outputs"
    before = sorted(historical.parent.iterdir())

    result = CliRunner().invoke(
        app,
        [
            command,
            "--config",
            str(CONFIG),
            "--predictions",
            str(historical),
            "--set",
            f"training.output_dir={runs.as_posix()}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(historical.parent.iterdir()) == before
    (stage_dir,) = runs.iterdir()
    assert stage_dir.name.startswith(f"{command}-")
    assert (stage_dir / produced).is_file()
    assert (stage_dir / f"{command}_manifest.json").is_file()


@pytest.mark.parametrize("command", ["score", "align"])
def test_occupied_run_directory_is_a_clean_error(
    command: str, historical: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "taken"
    run_dir.mkdir()
    (run_dir / f"{command}_manifest.json").write_text("{}", encoding="utf-8")
    arguments = [
        command,
        "--config",
        str(CONFIG),
        "--predictions",
        str(historical),
        "--run-dir",
        str(run_dir),
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
