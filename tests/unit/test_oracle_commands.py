"""Commands of the two oracles: BigCodeBench harness and CodeQL, with fake containers."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import test_bigcodebench_adapter as bcb
from typer.testing import CliRunner

from pstparser.empirical import bigcodebench
from pstparser.empirical.analysis import read_outcomes
from pstparser.empirical.bigcodebench import write_harness_dataset
from pstparser.empirical.cli import app
from pstparser.repro.manifest import digest_directory


@pytest.fixture
def parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return bcb.pinned_parquet(tmp_path, monkeypatch)


@pytest.fixture
def nltk_dir(tmp_path: Path) -> Path:
    return bcb.nltk_data(tmp_path)


# ------------------------------------------------------------------ check-bigcodebench


def command(
    tmp_path: Path,
    parquet: Path,
    nltk_dir: Path,
    sample_tasks: list[str],
    excluded: list[str],
    **calibrated_with: Any,
) -> list[str]:
    reference = tmp_path / "calibration-data" / "BigCodeBench.jsonl"
    harness = {
        "image": bcb.IMAGE,
        "dataset_sha256": write_harness_dataset(parquet, bcb.REVISION, bcb.SPLIT, reference),
        "nltk_sha256": digest_directory(nltk_dir),
        "parallel": 8,
        "calibrated": True,
        "network": "none",
        **calibrated_with,
    }
    journal = tmp_path / "generations.jsonl"
    journal.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bcb.planned()),
        encoding="utf-8",
    )
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps({"tasks": [{"task_id": t} for t in sample_tasks]}), encoding="utf-8"
    )
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(
        json.dumps({"harness": harness, "excluded": [{"task_id": task} for task in excluded]}),
        encoding="utf-8",
    )
    return [
        "check-bigcodebench",
        str(journal),
        "--parquet",
        str(parquet),
        "--revision",
        bcb.REVISION,
        "--split",
        bcb.SPLIT,
        "--image",
        bcb.IMAGE,
        "--nltk-dir",
        str(nltk_dir),
        "--sample",
        str(sample),
        "--exclusions",
        str(exclusions),
        "--directory",
        str(tmp_path / "check"),
        "--analysis-id",
        "pilot-01",
        "--order-seed",
        "7",
    ]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> bcb.FakeHarness:
    harness = bcb.FakeHarness()
    monkeypatch.setattr(bigcodebench, "run_harness", harness)
    monkeypatch.setattr(bigcodebench, "sanitize_in_image", bcb.fake_sanitize)
    monkeypatch.setattr(bigcodebench, "harness_version", lambda image: "0.2.4")
    monkeypatch.setattr(bigcodebench, "docker_resources", lambda: {"cpus": 28})
    return harness


def test_command_writes_the_ledger_the_details_and_the_manifest(
    parquet: Path, nltk_dir: Path, tmp_path: Path, offline: bcb.FakeHarness
) -> None:
    arguments = command(tmp_path, parquet, nltk_dir, bcb.TASKS, [])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "9 generations, 7 executed, 1 discordant" in result.output
    assert "warning: more than 1%" in result.output
    directory = tmp_path / "check"
    outcomes = read_outcomes(directory / "outcomes.jsonl")
    assert len(outcomes) == 9
    assert sum(o.outcome == "pass" for o in outcomes) == 6
    details = [
        json.loads(line)
        for line in (directory / "details.jsonl").read_text("utf-8").split("\n")
        if line
    ]
    assert {d["extraction_policy"] for d in details} == {"bigcodebench-sanitize"}
    manifest = json.loads((directory / "empirical-check-bcb_manifest.json").read_text("utf-8"))
    assert manifest["harness"]["image"] == bcb.IMAGE and manifest["harness_version"] == "0.2.4"
    assert manifest["extraction_policy"] == "bigcodebench-sanitize"
    assert manifest["rules"] == ["json-response-v1", "prompt-as-data-v1"]
    assert manifest["order_seed"] == 7 and manifest["flaky_tasks"] == [bcb.TASKS[1]]
    assert (
        manifest["outcomes_sha256"]
        == hashlib.sha256((directory / "outcomes.jsonl").read_bytes()).hexdigest()
    )
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


@pytest.mark.parametrize(
    "sample_tasks,excluded,message",
    [
        (bcb.TASKS[:2], [], "do not cover exactly the tasks of the sample"),
        ([*bcb.TASKS, "BigCodeBench/9"], [], "do not cover exactly the tasks of the sample"),
        (bcb.TASKS, [bcb.TASKS[1]], "generations for excluded tasks"),
    ],
)
def test_command_refuses_generations_outside_the_frozen_sample(
    sample_tasks: list[str],
    excluded: list[str],
    message: str,
    parquet: Path,
    nltk_dir: Path,
    tmp_path: Path,
    offline: bcb.FakeHarness,
) -> None:
    result = CliRunner().invoke(app, command(tmp_path, parquet, nltk_dir, sample_tasks, excluded))

    assert result.exit_code != 0
    assert message in str(result.exception)
    assert offline.calls == [] and not (tmp_path / "check").exists()


@pytest.mark.parametrize(
    "calibrated_with,differing",
    [
        ({"parallel": 4}, "['parallel']"),
        ({"image": bcb.IMAGE.replace("a" * 64, "b" * 64)}, "['image']"),
        (
            {"nltk_sha256": "0" * 64, "dataset_sha256": "0" * 64},
            "['dataset_sha256', 'nltk_sha256']",
        ),
        ({"calibrated": False}, "['calibrated']"),
    ],
)
def test_command_refuses_settings_other_than_those_of_the_calibration(
    calibrated_with: dict[str, Any],
    differing: str,
    parquet: Path,
    nltk_dir: Path,
    tmp_path: Path,
    offline: bcb.FakeHarness,
) -> None:
    arguments = command(tmp_path, parquet, nltk_dir, bcb.TASKS, [], **calibrated_with)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert f"differ from the calibration of the exclusions: {differing}" in str(result.exception)
    assert offline.calls == [] and not (tmp_path / "check" / "outcomes.jsonl").exists()


@pytest.mark.parametrize("damage", ["drop", "repeat"])
def test_command_refuses_an_incomplete_or_repeated_journal(
    damage: str, parquet: Path, nltk_dir: Path, tmp_path: Path, offline: bcb.FakeHarness
) -> None:
    arguments = command(tmp_path, parquet, nltk_dir, bcb.TASKS, [])
    journal = Path(arguments[1])
    lines = [line for line in journal.read_text(encoding="utf-8").split("\n") if line]
    # One condition of one task is missing, or one request appears twice.
    lines = lines[:-1] if damage == "drop" else [*lines, lines[0]]
    journal.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert "each condition and replicate exactly once" in str(result.exception)
    assert offline.calls == [] and not (tmp_path / "check").exists()


# ------------------------------------------------- check-security, calibrate-security
