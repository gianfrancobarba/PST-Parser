"""Official harness configuration and task calibration, without Docker."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from pstparser.empirical import benchmarks, bigcodebench
from pstparser.empirical.benchmarks import DatasetPin, load_bigcodebench_hard_ids
from pstparser.empirical.bigcodebench import (
    HarnessConfig,
    HarnessRun,
    calibrate,
    control_samples,
    eligibility,
    harness_command,
    read_harness_dataset,
    reference_failures,
    run_harness,
    write_harness_dataset,
    write_samples,
)
from pstparser.empirical.cli import app
from pstparser.repro.manifest import digest_directory

IMAGE = "bigcodebench/bigcodebench-evaluate@sha256:" + "a" * 64
REVISION = "1" * 40
HARD_REVISION = "2" * 40
SPLIT = "v9.9.9"
TASKS = [f"BigCodeBench/{number}" for number in range(8)]
HARD = ["BigCodeBench/1", "BigCodeBench/3", "BigCodeBench/5"]
NETWORK_TRACE = (
    "Traceback (most recent call last):\n  File '__test__.py', line 9, in test_get\n"
    "urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>"
)
ASSERTION_TRACE = "Traceback (most recent call last):\nAssertionError: None != 'dns.google'"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_row(task_id: str) -> dict[str, str]:
    number = task_id.split("/")[1]
    return {
        "task_id": task_id,
        "complete_prompt": f"def task_func():\n    '''doc {number}'''\n",
        "instruct_prompt": f"Solve task {number}.",
        "canonical_solution": f"    return {number}\n",
        "code_prompt": f"import os\nLIMIT = {number}\ndef task_func():\n",
        "test": f"class TestCases: value = {number}\n",
        "entry_point": "task_func",
        "doc_struct": "{}",
        "libs": "['os']",
    }


ROWS = [dataset_row(task) for task in TASKS]


@pytest.fixture
def parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "full.parquet"
    # Columns out of order: the harness dataset must still follow its own order.
    shuffled = [{key: row[key] for key in sorted(row, reverse=True)} for row in ROWS]
    pq.write_table(pa.Table.from_pylist(shuffled), path)
    monkeypatch.setattr(
        benchmarks,
        "BIGCODEBENCH_PINS",
        {(REVISION, SPLIT): DatasetPin(sha256_bytes(path.read_bytes()), len(ROWS))},
    )
    monkeypatch.setattr(bigcodebench, "BIGCODEBENCH_PINS", benchmarks.BIGCODEBENCH_PINS)
    return path


@pytest.fixture
def hard_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "hard.parquet"
    pq.write_table(pa.Table.from_pylist([dataset_row(task) for task in HARD]), path)
    monkeypatch.setattr(
        benchmarks,
        "BIGCODEBENCH_HARD_PINS",
        {(HARD_REVISION, SPLIT): DatasetPin(sha256_bytes(path.read_bytes()), len(HARD))},
    )
    return path


@pytest.fixture
def nltk_dir(tmp_path: Path) -> Path:
    path = tmp_path / "nltk_data"
    (path / "corpora").mkdir(parents=True)
    (path / "corpora" / "stopwords.zip").write_bytes(b"zip")
    return path


@pytest.fixture
def config(parquet: Path, nltk_dir: Path, tmp_path: Path) -> HarnessConfig:
    dataset = tmp_path / "data" / "BigCodeBench.jsonl"
    digest = write_harness_dataset(parquet, REVISION, SPLIT, dataset)
    nltk_digest = digest_directory(nltk_dir)
    assert nltk_digest is not None
    return HarnessConfig(IMAGE, dataset, digest, nltk_dir, nltk_digest, parallel=8)


# ------------------------------------------------------------------------------ inputs


def test_hard_identifiers_come_from_the_pinned_file(hard_parquet: Path) -> None:
    assert load_bigcodebench_hard_ids(hard_parquet, HARD_REVISION, SPLIT) == frozenset(HARD)
    with pytest.raises(ValueError, match="no pinned"):
        load_bigcodebench_hard_ids(hard_parquet, REVISION, SPLIT)


def test_the_hard_dataset_is_pinned_to_the_protocol_revision() -> None:
    assert {
        ("298d2cc7b96612e15e47313c3603ee124cee0c1f", "v0.1.4"): DatasetPin(
            "73a4270b43feb81abefad7bb1b592937c768ea45c3263bedc1cabd30a9a6ce79", 148
        )
    } == benchmarks.BIGCODEBENCH_HARD_PINS


def test_harness_dataset_keeps_every_field_in_the_harness_order(
    parquet: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out" / "dataset.jsonl"

    digest = write_harness_dataset(parquet, REVISION, SPLIT, out)

    raw = out.read_bytes()
    assert digest == sha256_bytes(raw)
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines[-1] == "" and len(lines) == len(ROWS) + 1
    assert [json.loads(line) for line in lines[:-1]] == ROWS
    assert list(json.loads(lines[0])) == [
        "task_id",
        "complete_prompt",
        "instruct_prompt",
        "canonical_solution",
        "code_prompt",
        "test",
        "entry_point",
        "doc_struct",
        "libs",
    ]
    assert read_harness_dataset(out) == ROWS
    with pytest.raises(FileExistsError):
        write_harness_dataset(parquet, REVISION, SPLIT, out)


def test_harness_dataset_refuses_an_unpinned_file(parquet: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no pinned"):
        write_harness_dataset(parquet, "3" * 40, SPLIT, tmp_path / "a.jsonl")
    other = tmp_path / "other.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS[:2]), other)
    with pytest.raises(ValueError, match="not the pinned"):
        write_harness_dataset(other, REVISION, SPLIT, tmp_path / "b.jsonl")
    assert not (tmp_path / "a.jsonl").exists() and not (tmp_path / "b.jsonl").exists()


def test_control_programs() -> None:
    row = ROWS[3]

    assert control_samples([row], "reference") == [
        {
            "task_id": "BigCodeBench/3",
            "solution": "import os\nLIMIT = 3\ndef task_func():\n    return 3\n",
        }
    ]
    assert control_samples([row], "trivial") == [
        {
            "task_id": "BigCodeBench/3",
            "solution": "import os\nLIMIT = 3\ndef task_func():\n    return None\n",
        }
    ]
    assert control_samples([row], "empty") == [{"task_id": "BigCodeBench/3", "solution": ""}]
    with pytest.raises(ValueError, match="unknown control"):
        control_samples([row], "other")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- harness


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"image": "bigcodebench/bigcodebench-evaluate:latest"}, "pinned"),
        ({"image": "sha256:" + "a" * 64}, "pinned"),
        ({"network": "bridge"}, "without network"),
        ({"parallel": 0}, "parallel"),
        ({"parallel": -1}, "parallel"),
    ],
)
def test_harness_refuses_a_tag_a_network_and_free_parallelism(
    changes: dict[str, Any], message: str, tmp_path: Path
) -> None:
    arguments: dict[str, Any] = {
        "image": IMAGE,
        "dataset_jsonl": tmp_path / "d.jsonl",
        "dataset_sha256": "0" * 64,
        "nltk_dir": tmp_path,
        "nltk_sha256": "0" * 64,
        "parallel": 8,
    }
    with pytest.raises(ValueError, match=message):
        HarnessConfig(**{**arguments, **changes})


def test_harness_command_is_offline_pinned_and_read_only(
    config: HarnessConfig, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"

    command = harness_command(config, run_dir)

    assert command == [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--mount",
        f"type=bind,source={run_dir.resolve()},target=/app",
        "--mount",
        f"type=bind,source={config.dataset_jsonl.resolve().parent},target=/data,readonly",
        "--mount",
        f"type=bind,source={config.nltk_dir.resolve()},"
        "target=/home/bigcodebenchuser/nltk_data,readonly",
        "--env",
        "BIGCODEBENCH_OVERRIDE_PATH=/data/BigCodeBench.jsonl",
        IMAGE,
        "--execution",
        "local",
        "--split",
        "instruct",
        "--subset",
        "full",
        "--samples",
        "samples.jsonl",
        "--parallel",
        "8",
        "--calibrated",
        "True",
    ]


def test_harness_command_options(config: HarnessConfig, tmp_path: Path) -> None:
    selective = harness_command(config, tmp_path, task_ids=TASKS[:2], no_gt=True)
    reference = harness_command(config, tmp_path, check_gt_only=True)

    assert selective[-3:] == ["--selective_evaluate", "BigCodeBench/0,BigCodeBench/1", "--no_gt"]
    assert reference[-1] == "--check_gt_only"
    for command in (selective, reference):
        assert not any(argument.startswith("--pass_k") for argument in command)
    with pytest.raises(ValueError, match="task_ids"):
        harness_command(config, tmp_path, task_ids=[])
    with pytest.raises(ValueError, match="task_ids"):
        harness_command(config, tmp_path, task_ids=["BigCodeBench/1,BigCodeBench/2"])


class FakeDocker:
    """Stand-in for ``subprocess.run``: records the command and writes a log."""

    def __init__(self, returncode: int = 0) -> None:
        """Start with the exit code the container will report."""
        self.returncode = returncode
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **options: Any) -> Any:
        self.calls.append({"command": command, **options})
        options["stdout"].write(b"harness output\n")
        return type("Completed", (), {"returncode": self.returncode})()


def test_run_harness_writes_the_log_and_returns_the_exit_code(
    config: HarnessConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = FakeDocker(returncode=3)
    monkeypatch.setattr(bigcodebench.subprocess, "run", docker)
    run_dir = tmp_path / "run"
    write_samples([{"task_id": TASKS[0], "solution": ""}], run_dir)

    run = run_harness(config, run_dir, no_gt=True)

    assert run.exit_code == 3
    assert run.log.read_bytes() == b"harness output\n"
    assert list(run.command) == harness_command(config, run_dir, no_gt=True)
    assert len(docker.calls) == 1
    assert docker.calls[0]["stdin"] == bigcodebench.subprocess.DEVNULL


def test_run_harness_refuses_a_directory_with_earlier_results(
    config: HarnessConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = FakeDocker()
    monkeypatch.setattr(bigcodebench.subprocess, "run", docker)
    run_dir = tmp_path / "run"
    write_samples([{"task_id": TASKS[0], "solution": ""}], run_dir)
    (run_dir / "samples_eval_results.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must hold exactly"):
        run_harness(config, run_dir)
    with pytest.raises(ValueError, match="must hold exactly"):
        run_harness(config, run_dir, check_gt_only=True)
    with pytest.raises(ValueError, match="must hold exactly"):
        run_harness(config, tmp_path / "absent")
    assert docker.calls == []


@pytest.mark.parametrize("target", ["dataset", "nltk"])
def test_run_harness_refuses_inputs_that_changed_since_they_were_hashed(
    target: str, config: HarnessConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = FakeDocker()
    monkeypatch.setattr(bigcodebench.subprocess, "run", docker)
    if target == "dataset":
        with config.dataset_jsonl.open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
    else:
        (config.nltk_dir / "corpora" / "extra.zip").write_bytes(b"x")

    with pytest.raises(ValueError, match="not the recorded"):
        run_harness(config, tmp_path / "run", check_gt_only=True)
    assert docker.calls == []


def test_write_samples_never_reuses_a_directory(tmp_path: Path) -> None:
    write_samples([{"task_id": TASKS[0], "solution": "x = 1\n"}], tmp_path / "run")

    assert (tmp_path / "run" / "samples.jsonl").read_bytes() == (
        b'{"task_id": "BigCodeBench/0", "solution": "x = 1\\n"}\n'
    )
    with pytest.raises(FileExistsError):
        write_samples([], tmp_path / "run")


# ------------------------------------------------------------------------ reference log


def test_reference_failures_are_read_from_the_log() -> None:
    log = (
        "100%|##########| 8/8 [00:05<00:00]\nExpected outputs computed in 5.77s\n"
        "Groundtruth pass rate: 0.750\n"
        "Failed tasks: ['BigCodeBench/5', 'BigCodeBench/2']\n"
    )

    assert reference_failures(log, 8) == ["BigCodeBench/2", "BigCodeBench/5"]
    assert reference_failures("Groundtruth pass rate: 1.000\n", 8) == []


@pytest.mark.parametrize(
    "log,message",
    [
        ("Reading samples...\n", "did not report"),
        ("Groundtruth pass rate: 0.750\nGroundtruth pass rate: 0.750\n", "did not report"),
        ("Groundtruth pass rate: 0.750\nFailed tasks: ['BigCodeBench/5']\n", "does not match"),
        ("Groundtruth pass rate: 1.000\nFailed tasks: ['BigCodeBench/5']\n", "does not match"),
        ("Groundtruth pass rate: 0.750\n", "does not match"),
        (
            "Groundtruth pass rate: 0.750\nFailed tasks: ['BigCodeBench/5', 'BigCodeBench/5']\n",
            "malformed",
        ),
    ],
)
def test_incomplete_or_inconsistent_reference_log_is_refused(log: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reference_failures(log, 8)


# -------------------------------------------------------------------------- eligibility


def results(statuses: dict[str, str], details: dict[str, dict[str, str]] | None = None) -> Any:
    return {
        task: {
            "task_id": task,
            "status": statuses.get(task, "fail"),
            "details": (details or {}).get(task, {}),
        }
        for task in TASKS
    }


def passing(*failed: str, **other: str) -> dict[str, str]:
    statuses = {task: "pass" for task in TASKS}
    statuses.update({task: "fail" for task in failed})
    statuses.update({f"BigCodeBench/{key[1:]}": value for key, value in other.items()})
    return statuses


def test_rules_e1_to_e4() -> None:
    report = eligibility(
        TASKS,
        HARD,
        # Task 1 fails in both reference runs, task 2 only in the second.
        reference_runs=[["BigCodeBench/1"], ["BigCodeBench/1", "BigCodeBench/2"]],
        reference_samples=results(
            passing("BigCodeBench/1", t2="timeout"),
            {"BigCodeBench/1": {"test_get": NETWORK_TRACE}},
        ),
        # Task 4 passes with the trivial program only; task 6 with both.
        trivial=results({"BigCodeBench/4": "pass", "BigCodeBench/6": "pass"}),
        empty=results({"BigCodeBench/6": "pass", "BigCodeBench/2": "timeout"}),
    )

    assert report["tasks"] == 8
    assert report["hard"] == 3
    assert report["excluded_count"] == 4
    assert report["eligible"] == 4
    # Hard holds tasks 1, 3 and 5; only task 1 is excluded.
    assert report["eligible_hard"] == 2
    assert report["unexplained"] == []
    assert [(row["task_id"], row["rules"], row["reason"]) for row in report["excluded"]] == [
        ("BigCodeBench/1", ["E1", "E4"], "reference_fails_offline"),
        ("BigCodeBench/2", ["E1"], "reference_fails_offline"),
        ("BigCodeBench/4", ["E2"], "trivial_program_passes"),
        ("BigCodeBench/6", ["E2", "E3"], "trivial_program_passes"),
    ]
    first = report["excluded"][0]
    assert first["hard"] is True
    assert first["evidence"] == {
        "reference_runs_failed": [0, 1],
        "reference_sample_status": "fail",
        "trivial_status": "fail",
        "empty_status": "fail",
        "network_message": "urlopen error",
    }
    assert report["excluded"][1]["evidence"]["reference_runs_failed"] == [1]
    assert report["excluded"][1]["evidence"]["network_message"] is None
    assert report["rules"] == {
        "E1": "reference_fails_offline",
        "E2": "trivial_program_passes",
        "E3": "empty_program_passes",
        "E4": "network_dependency",
    }


def test_network_traceback_excludes_even_when_both_reference_runs_pass() -> None:
    report = eligibility(
        TASKS,
        HARD,
        reference_runs=[[], []],
        reference_samples=results(
            passing("BigCodeBench/3"), {"BigCodeBench/3": {"test_a": NETWORK_TRACE}}
        ),
        trivial=results({}),
        empty=results({}),
    )

    assert [(row["task_id"], row["rules"], row["reason"]) for row in report["excluded"]] == [
        ("BigCodeBench/3", ["E4"], "network_dependency")
    ]
    assert report["eligible_hard"] == 2


def test_a_passing_reference_sample_is_never_read_as_network_evidence() -> None:
    report = eligibility(
        TASKS,
        HARD,
        reference_runs=[[], []],
        reference_samples=results(passing(), {"BigCodeBench/3": {"test_a": NETWORK_TRACE}}),
        trivial=results({}),
        empty=results({}),
    )

    assert report["excluded"] == [] and report["eligible"] == 8


def test_reference_sample_failure_outside_the_rules_is_reported_not_excluded() -> None:
    report = eligibility(
        TASKS,
        HARD,
        reference_runs=[[], []],
        reference_samples=results(
            passing("BigCodeBench/7"), {"BigCodeBench/7": {"test_a": ASSERTION_TRACE}}
        ),
        trivial=results({}),
        empty=results({}),
    )

    assert report["excluded"] == []
    assert report["unexplained"] == ["BigCodeBench/7"]


def test_eligibility_refuses_an_incomplete_calibration() -> None:
    complete = results(passing())
    partial = {task: row for task, row in complete.items() if task != TASKS[0]}

    with pytest.raises(ValueError, match="two independent reference runs"):
        eligibility(TASKS, HARD, [[]], complete, results({}), results({}))
    with pytest.raises(ValueError, match="trivial run does not cover"):
        eligibility(TASKS, HARD, [[], []], complete, partial, results({}))
    with pytest.raises(ValueError, match="unknown task"):
        eligibility(TASKS, HARD, [[], ["BigCodeBench/99"]], complete, results({}), results({}))
    with pytest.raises(ValueError, match="Hard task"):
        eligibility(TASKS, ["BigCodeBench/99"], [[], []], complete, results({}), results({}))


# -------------------------------------------------------------------------- calibration


class FakeHarness:
    """Play the five calibration runs from scripted results, without Docker."""

    def __init__(
        self,
        reference_logs: list[str],
        statuses: dict[str, dict[str, str]],
        details: dict[str, dict[str, str]] | None = None,
        tamper: str | None = None,
        exit_code: int = 0,
    ) -> None:
        """Script the reference logs and the status of every control sample."""
        self.reference_logs = list(reference_logs)
        self.statuses = statuses
        self.details = details or {}
        self.tamper = tamper
        self.exit_code = exit_code
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, config: HarnessConfig, run_dir: Path, **options: Any) -> HarnessRun:
        self.calls.append((run_dir.name, options))
        if options.get("check_gt_only"):
            run_dir.mkdir(parents=True)
            (run_dir / "log.txt").write_text(self.reference_logs.pop(0), encoding="utf-8")
        else:
            kind = run_dir.name.removeprefix("samples-")
            code = {row["task_id"]: row["code_prompt"] for row in ROWS}
            lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").split("\n")
            evaluated = {}
            for sample in (json.loads(line) for line in lines if line):
                task = sample["task_id"]
                ran = code[task] + "\n    pass\n" + sample["solution"]
                if self.tamper == kind and task == TASKS[2]:
                    ran += "# changed"
                evaluated[task] = [
                    {
                        "task_id": task,
                        "solution": ran,
                        "status": self.statuses[kind].get(task, "fail"),
                        "details": self.details.get(task, {}) if kind == "reference" else {},
                    }
                ]
            (run_dir / "samples_eval_results.json").write_text(
                json.dumps({"date": "2026-11-15 10:00", "eval": evaluated}), encoding="utf-8"
            )
            (run_dir / "log.txt").write_text("evaluated\n", encoding="utf-8")
        return HarnessRun(
            run_dir, ("docker", "run", IMAGE, "--execution", "local"), self.exit_code, 1.5
        )


def scripted() -> FakeHarness:
    failing = "Groundtruth pass rate: 0.875\nFailed tasks: ['BigCodeBench/1']\n"
    return FakeHarness(
        [failing, failing],
        {
            "reference": passing("BigCodeBench/1"),
            "trivial": {"BigCodeBench/6": "pass"},
            "empty": {"BigCodeBench/6": "pass"},
        },
        {"BigCodeBench/1": {"test_get": NETWORK_TRACE}},
    )


def test_calibration_runs_five_times_and_applies_the_rules(
    config: HarnessConfig, tmp_path: Path
) -> None:
    harness = scripted()

    report = calibrate(config, HARD, tmp_path / "runs", runner=harness)

    assert [name for name, _ in harness.calls] == [
        "reference-a",
        "reference-b",
        "samples-reference",
        "samples-trivial",
        "samples-empty",
    ]
    assert [options for _, options in harness.calls] == [
        {"task_ids": None, "check_gt_only": True},
        {"task_ids": None, "check_gt_only": True},
        {"task_ids": None, "no_gt": True},
        {"task_ids": None, "no_gt": True},
        {"task_ids": None, "no_gt": True},
    ]
    assert report["scope"] == "all"
    assert (report["tasks"], report["excluded_count"], report["eligible"]) == (8, 2, 6)
    assert report["eligible_hard"] == 2
    assert [(row["task_id"], row["rules"]) for row in report["excluded"]] == [
        ("BigCodeBench/1", ["E1", "E4"]),
        ("BigCodeBench/6", ["E2", "E3"]),
    ]
    assert report["harness"]["image"] == IMAGE
    assert report["harness"]["parallel"] == 8
    assert report["harness"]["network"] == "none"
    assert [run["name"] for run in report["runs"]] == [name for name, _ in harness.calls]
    assert report["runs"][0]["failed"] == ["BigCodeBench/1"]
    assert report["runs"][3]["statuses"] == {"pass": 1, "fail": 7, "timeout": 0}
    assert report["runs"][2]["arguments"] == ["--execution", "local"]
    submitted = (tmp_path / "runs" / "samples-trivial" / "samples.jsonl").read_bytes()
    assert report["runs"][3]["samples_sha256"] == sha256_bytes(submitted)


def test_calibration_on_a_subset_is_marked_as_such(config: HarnessConfig, tmp_path: Path) -> None:
    subset = ["BigCodeBench/1", "BigCodeBench/6", "BigCodeBench/3"]
    failing = "Groundtruth pass rate: 0.667\nFailed tasks: ['BigCodeBench/1']\n"
    harness = scripted()
    harness.reference_logs = [failing, failing]

    report = calibrate(config, HARD, tmp_path / "runs", task_ids=subset, runner=harness)

    assert report["scope"] == "subset"
    assert (report["tasks"], report["hard"], report["eligible"]) == (3, 2, 1)
    assert all(options["task_ids"] == subset for _, options in harness.calls)
    with pytest.raises(ValueError, match="distinct tasks"):
        calibrate(config, HARD, tmp_path / "other", task_ids=["BigCodeBench/99"], runner=harness)


def test_calibration_stops_on_a_failed_container(config: HarnessConfig, tmp_path: Path) -> None:
    harness = scripted()
    harness.exit_code = 137

    with pytest.raises(RuntimeError, match="reference-a exited with 137"):
        calibrate(config, HARD, tmp_path / "runs", runner=harness)
    assert len(harness.calls) == 1


def test_calibration_refuses_results_for_another_program(
    config: HarnessConfig, tmp_path: Path
) -> None:
    harness = scripted()
    harness.tamper = "trivial"

    with pytest.raises(ValueError, match=r"samples-trivial.*BigCodeBench/2"):
        calibrate(config, HARD, tmp_path / "runs", runner=harness)


def command_arguments(parquet: Path, hard: Path, nltk: Path, tmp_path: Path) -> list[str]:
    return [
        "calibrate-bigcodebench",
        str(parquet),
        "--revision",
        REVISION,
        "--split",
        SPLIT,
        "--hard-parquet",
        str(hard),
        "--hard-revision",
        HARD_REVISION,
        "--image",
        IMAGE,
        "--nltk-dir",
        str(nltk),
        "--directory",
        str(tmp_path / "calibration"),
        "--out",
        str(tmp_path / "protocol" / "bcb-exclusions.json"),
    ]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> FakeHarness:
    harness = scripted()
    monkeypatch.setattr(bigcodebench, "run_harness", harness)
    monkeypatch.setattr(bigcodebench, "harness_version", lambda image: "0.2.4")
    monkeypatch.setattr(bigcodebench, "docker_resources", lambda: {"cpus": 28})
    return harness


def test_command_writes_the_exclusions_with_their_hash(
    parquet: Path, hard_parquet: Path, nltk_dir: Path, tmp_path: Path, offline: FakeHarness
) -> None:
    arguments = command_arguments(parquet, hard_parquet, nltk_dir, tmp_path)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "2 excluded, 6 eligible, 2 eligible Hard" in result.output
    out = tmp_path / "protocol" / "bcb-exclusions.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["scope"] == "all"
    assert payload["dataset_revision"] == REVISION
    assert payload["dataset_file_sha256"] == sha256_bytes(parquet.read_bytes())
    assert payload["hard_file_sha256"] == sha256_bytes(hard_parquet.read_bytes())
    assert payload["harness_version"] == "0.2.4"
    assert payload["docker"] == {"cpus": 28}
    assert [row["task_id"] for row in payload["excluded"]] == ["BigCodeBench/1", "BigCodeBench/6"]
    directory = tmp_path / "calibration"
    assert (directory / "calibration.json").read_bytes() == out.read_bytes()
    manifest = json.loads(
        (directory / "empirical-calibrate-bcb_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["conclusive"] is True
    assert manifest["exclusions_sha256"] == sha256_bytes(out.read_bytes())
    dataset = directory / "data" / f"BigCodeBench-{SPLIT}.jsonl"
    assert manifest["harness"]["dataset_sha256"] == sha256_bytes(dataset.read_bytes())
    # Neither the exclusions nor the run directory are ever overwritten.
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


def test_command_writes_no_exclusions_for_a_subset(
    parquet: Path, hard_parquet: Path, nltk_dir: Path, tmp_path: Path, offline: FakeHarness
) -> None:
    offline.reference_logs = ["Groundtruth pass rate: 1.000\n", "Groundtruth pass rate: 1.000\n"]
    arguments = [
        *command_arguments(parquet, hard_parquet, nltk_dir, tmp_path),
        "--task-ids",
        "BigCodeBench/0,BigCodeBench/3",
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "protocol").exists()
    payload = json.loads((tmp_path / "calibration" / "calibration.json").read_text("utf-8"))
    assert payload["scope"] == "subset" and payload["tasks"] == 2
    manifest = json.loads(
        (tmp_path / "calibration" / "empirical-calibrate-bcb_manifest.json").read_text("utf-8")
    )
    assert manifest["conclusive"] is False and manifest["exclusions_sha256"] is None


def test_command_fails_and_writes_no_exclusions_on_an_unexplained_failure(
    parquet: Path, hard_parquet: Path, nltk_dir: Path, tmp_path: Path, offline: FakeHarness
) -> None:
    offline.statuses["reference"] = passing("BigCodeBench/1", "BigCodeBench/4")

    result = CliRunner().invoke(app, command_arguments(parquet, hard_parquet, nltk_dir, tmp_path))

    assert result.exit_code != 0
    assert "BigCodeBench/4" in str(result.exception)
    assert not (tmp_path / "protocol").exists()
    payload = json.loads((tmp_path / "calibration" / "calibration.json").read_text("utf-8"))
    assert payload["unexplained"] == ["BigCodeBench/4"]
