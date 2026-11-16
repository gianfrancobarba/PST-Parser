"""Opt-in calibration of BigCodeBench with the real harness image and Docker.

Nothing is downloaded: the image must already be installed and the dataset
files must be local copies of the pinned ones. Set:

- ``PST_BCB_IMAGE``: the harness image as ``name@sha256:<digest>``;
- ``PST_BCB_PARQUET`` and ``PST_BCB_HARD_PARQUET``: the pinned parquet files;
- ``PST_BCB_NLTK``: the local ``nltk`` data directory;
- ``PST_BCB_FULL=1``: also run the complete calibration (about half an hour).
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.benchmarks import load_bigcodebench_hard_ids
from pstparser.empirical.bigcodebench import (
    HarnessConfig,
    calibrate,
    task_number,
    write_harness_dataset,
)
from pstparser.empirical.cli import app
from pstparser.repro.manifest import digest_directory

pytestmark = pytest.mark.slow

REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
HARD_REVISION = "298d2cc7b96612e15e47313c3603ee124cee0c1f"
SPLIT = "v0.1.4"
# Excluded in every calibration made so far. Reference fails offline: 101, 176,
# 314, 590, 1005, 1012. Tests that pass a program doing nothing: 59, 272.
ALWAYS_EXCLUDED = {59, 101, 176, 272, 314, 590, 1005, 1012}


def _setting(variable: str) -> str:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable}; see the module documentation")
    return value


@pytest.fixture(scope="module")
def parquet() -> Path:
    return Path(_setting("PST_BCB_PARQUET"))


@pytest.fixture(scope="module")
def hard_parquet() -> Path:
    return Path(_setting("PST_BCB_HARD_PARQUET"))


@pytest.fixture(scope="module")
def nltk_dir() -> Path:
    return Path(_setting("PST_BCB_NLTK"))


@pytest.fixture(scope="module")
def image() -> str:
    return _setting("PST_BCB_IMAGE")


def test_harness_dataset_is_the_file_the_earlier_runs_used(parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "BigCodeBench-v0.1.4.jsonl"

    digest = write_harness_dataset(parquet, REVISION, SPLIT, out)

    assert digest == "f6704a124f4edcec9d12cc7912f41f3d4207f51b87f091d136e3436f7e25149f"
    # The harness keys its ground-truth cache on the MD5 of this file.
    assert hashlib.md5(out.read_bytes()).hexdigest() == "7da7e5ed503d450b3bffa6f4a497b166"


def test_first_ten_hard_tasks_calibrate_and_repeat(
    parquet: Path, hard_parquet: Path, nltk_dir: Path, image: str, tmp_path: Path
) -> None:
    hard = load_bigcodebench_hard_ids(hard_parquet, HARD_REVISION, SPLIT)
    first_ten = sorted(hard, key=task_number)[:10]
    dataset = tmp_path / "data" / "BigCodeBench-v0.1.4.jsonl"
    digest = write_harness_dataset(parquet, REVISION, SPLIT, dataset)
    nltk_digest = digest_directory(nltk_dir)
    assert nltk_digest is not None
    config = HarnessConfig(image, dataset, digest, nltk_dir, nltk_digest, parallel=8)

    report = calibrate(config, hard, tmp_path / "runs", task_ids=first_ten)

    runs = {run["name"]: run for run in report["runs"]}
    # References pass in both runs, which therefore agree; both controls fail.
    assert runs["reference-a"]["failed"] == runs["reference-b"]["failed"] == []
    assert runs["samples-reference"]["statuses"] == {"pass": 10, "fail": 0, "timeout": 0}
    assert runs["samples-trivial"]["statuses"]["pass"] == 0
    assert runs["samples-empty"]["statuses"]["pass"] == 0
    assert report["excluded"] == [] and report["unexplained"] == []
    assert (report["tasks"], report["hard"], report["eligible_hard"]) == (10, 10, 10)


def test_full_calibration_excludes_the_stable_cases_and_only_adds_rule_e1(
    parquet: Path, hard_parquet: Path, nltk_dir: Path, image: str, tmp_path: Path
) -> None:
    if os.environ.get("PST_BCB_FULL") != "1":
        pytest.skip("set PST_BCB_FULL=1 to run the complete calibration")
    out = tmp_path / "protocol" / "bcb-exclusions.json"
    directory = tmp_path / "calibration"

    result = CliRunner().invoke(
        app,
        [
            "calibrate-bigcodebench",
            str(parquet),
            "--revision",
            REVISION,
            "--hard-parquet",
            str(hard_parquet),
            "--hard-revision",
            HARD_REVISION,
            "--image",
            image,
            "--nltk-dir",
            str(nltk_dir),
            "--directory",
            str(directory),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    rules = {task_number(row["task_id"]): row["rules"] for row in payload["excluded"]}
    assert set(rules) >= ALWAYS_EXCLUDED
    assert rules[59] == rules[272] == ["E2", "E3"]
    # A few references fail only now and then (205, 363 and 823 so far): one
    # calibration may add such tasks, always and only through rule E1. The
    # frozen exclusions are the union of the calibrations.
    assert all(rules[task] == ["E1"] for task in set(rules) - ALWAYS_EXCLUDED)
    assert (payload["tasks"], payload["hard"]) == (1140, 148)
    assert payload["eligible"] == 1140 - payload["excluded_count"]
    assert payload["unexplained"] == []
    assert payload["scope"] == "all"
    manifest = json.loads(
        (directory / "empirical-calibrate-bcb_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["exclusions_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
