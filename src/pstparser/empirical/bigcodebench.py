"""Run the official BigCodeBench harness in its pinned image and calibrate the tasks.

The harness is the oracle of functional correctness. It runs offline, from an
image pinned by digest, on a dataset file and on ``nltk`` data identified by
hash. Eligibility of a task depends on harness objects only (reference
solution, tests, trivial programs), never on any model output, so the rules
apply to every condition alike.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pstparser.empirical.analysis import Outcome, OutcomeRecord
from pstparser.empirical.benchmarks import BIGCODEBENCH_PINS
from pstparser.empirical.extraction import extraction_statuses
from pstparser.repro.manifest import digest_directory, digest_file

#: Columns of the dataset file read by the harness, in the order they are written.
DATASET_FIELDS: Final = (
    "task_id",
    "complete_prompt",
    "instruct_prompt",
    "canonical_solution",
    "code_prompt",
    "test",
    "entry_point",
    "doc_struct",
    "libs",
)
SAMPLES_FILE: Final = "samples.jsonl"
RESULTS_FILE: Final = "samples_eval_results.json"
LOG_FILE: Final = "log.txt"
_NLTK_TARGET: Final = "/home/bigcodebenchuser/nltk_data"

#: Exclusion rules and the enumerated reason each one records.
RULES: Final[dict[str, str]] = {
    "E1": "reference_fails_offline",
    "E2": "trivial_program_passes",
    "E3": "empty_program_passes",
    "E4": "network_dependency",
}
ControlKind = Literal["reference", "trivial", "empty"]

#: Messages of a failed attempt to reach the network, as they appear in a traceback.
_NETWORK_EVIDENCE: Final = re.compile(
    r"Temporary failure in name resolution|NameResolutionError|gaierror|"
    r"Network is unreachable|urlopen error|Max retries exceeded|"
    r"Failed to establish a new connection|ConnectionError",
)
_IMAGE: Final = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_PASS_RATE: Final = re.compile(r"Groundtruth pass rate: ([0-9.]+)")
_FAILED_TASKS: Final = re.compile(r"Failed tasks: (\[[^\]]*\])")


@dataclass(frozen=True)
class HarnessConfig:
    """Frozen execution settings of the official harness."""

    image: str
    dataset_jsonl: Path
    dataset_sha256: str
    nltk_dir: Path
    nltk_sha256: str
    parallel: int
    calibrated: bool = True
    network: str = "none"

    def __post_init__(self) -> None:
        """Reject a mutable image tag, any network and an unset parallelism."""
        if not _IMAGE.fullmatch(self.image):
            raise ValueError("the harness image must be pinned as name@sha256:<64 hex digits>")
        if self.network != "none":
            raise ValueError("the harness runs without network")
        if type(self.parallel) is not int or self.parallel < 1:
            raise ValueError("parallel must be a fixed positive number of workers")

    def verify(self) -> None:
        """Check the mounted inputs against their recorded hashes."""
        if digest_file(self.dataset_jsonl) != self.dataset_sha256:
            raise ValueError(f"{self.dataset_jsonl}: not the recorded harness dataset")
        if digest_directory(self.nltk_dir) != self.nltk_sha256:
            raise ValueError(f"{self.nltk_dir}: not the recorded nltk data")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, with file names instead of local paths."""
        record = asdict(self)
        record["dataset_jsonl"] = self.dataset_jsonl.name
        record["nltk_dir"] = self.nltk_dir.name
        return record


@dataclass(frozen=True)
class HarnessRun:
    """One finished container run and where its files are."""

    directory: Path
    command: tuple[str, ...]
    exit_code: int
    elapsed_seconds: float

    @property
    def log(self) -> Path:
        """Combined standard output and error of the container."""
        return self.directory / LOG_FILE

    @property
    def results(self) -> Path:
        """Per-sample results written by the harness."""
        return self.directory / RESULTS_FILE


def write_harness_dataset(parquet: Path, dataset_revision: str, split: str, out: Path) -> str:
    """Convert the pinned parquet into the JSON Lines file the harness loads.

    This file holds tests and reference solutions: it is oracle material, kept
    apart from the canonical sources. Returns its SHA-256.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pin = BIGCODEBENCH_PINS.get((dataset_revision, split))
    if pin is None:
        raise ValueError(f"no pinned dataset file for BigCodeBench {dataset_revision} {split}")
    data = Path(parquet).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin.file_sha256:
        raise ValueError(f"{parquet}: not the pinned dataset file")
    rows = pq.read_table(pa.BufferReader(data), columns=list(DATASET_FIELDS)).to_pylist()
    if len(rows) != pin.records:
        raise ValueError(f"{parquet}: {len(rows)} records, expected {pin.records}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if not all(isinstance(row[name], str) for name in DATASET_FIELDS):
                raise ValueError(f"{row['task_id']}: non-text dataset field")
            handle.write(
                json.dumps({name: row[name] for name in DATASET_FIELDS}, ensure_ascii=False) + "\n"
            )
    digest = digest_file(out)
    assert digest is not None
    return digest


def read_harness_dataset(dataset_jsonl: Path) -> list[dict[str, str]]:
    """Read the harness dataset; lines are split on the line feed only."""
    lines = Path(dataset_jsonl).read_bytes().decode("utf-8").split("\n")
    return [json.loads(line) for line in lines if line]


def control_samples(
    dataset: Sequence[Mapping[str, str]], kind: ControlKind
) -> list[dict[str, str]]:
    """Build one calibration program per task.

    ``reference`` is the reference solution submitted as a model would submit
    it; ``trivial`` keeps the starting code and returns nothing; ``empty`` is
    the empty program.
    """
    if kind == "reference":
        return [
            {"task_id": row["task_id"], "solution": row["code_prompt"] + row["canonical_solution"]}
            for row in dataset
        ]
    if kind == "trivial":
        return [
            {"task_id": row["task_id"], "solution": row["code_prompt"] + "    return None\n"}
            for row in dataset
        ]
    if kind == "empty":
        return [{"task_id": row["task_id"], "solution": ""} for row in dataset]
    raise ValueError(f"unknown control {kind!r}")


def write_samples(rows: Sequence[Mapping[str, Any]], directory: Path) -> Path:
    """Write the samples of one run into a new run directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / SAMPLES_FILE
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def harness_command(
    config: HarnessConfig,
    run_dir: Path,
    *,
    task_ids: Sequence[str] | None = None,
    check_gt_only: bool = False,
    no_gt: bool = False,
) -> list[str]:
    """Build the container command: no network, no pull, inputs mounted read-only.

    ``--pass_k`` is never passed: the pinned harness misreads a single value.
    """
    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        config.network,
        "--mount",
        f"type=bind,source={Path(run_dir).resolve()},target=/app",
        "--mount",
        f"type=bind,source={config.dataset_jsonl.resolve().parent},target=/data,readonly",
        "--mount",
        f"type=bind,source={config.nltk_dir.resolve()},target={_NLTK_TARGET},readonly",
        "--env",
        f"BIGCODEBENCH_OVERRIDE_PATH=/data/{config.dataset_jsonl.name}",
        config.image,
        "--execution",
        "local",
        "--split",
        "instruct",
        "--subset",
        "full",
        "--samples",
        SAMPLES_FILE,
        "--parallel",
        str(config.parallel),
        "--calibrated",
        str(config.calibrated),
    ]
    if task_ids is not None:
        if not task_ids or any("," in task or not task.strip() for task in task_ids):
            raise ValueError("task_ids must be complete, non-empty task identifiers")
        command += ["--selective_evaluate", ",".join(task_ids)]
    if check_gt_only:
        command.append("--check_gt_only")
    if no_gt:
        command.append("--no_gt")
    return command


def run_harness(
    config: HarnessConfig,
    run_dir: Path,
    *,
    task_ids: Sequence[str] | None = None,
    check_gt_only: bool = False,
    no_gt: bool = False,
    timeout_seconds: int = 4 * 60 * 60,
) -> HarnessRun:
    """Run the harness once in a run directory that holds nothing but its samples.

    The harness reloads an existing result file instead of evaluating, so a
    directory with anything else in it is refused. A reference-only run needs
    no samples and gets an empty directory. The exit code is returned, not
    interpreted: a non-zero value is an infrastructure failure for the caller.
    """
    run_dir = Path(run_dir)
    expected = set() if check_gt_only else {SAMPLES_FILE}
    if check_gt_only:
        run_dir.mkdir(parents=True, exist_ok=True)
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != expected:
        raise ValueError(f"{run_dir}: a run directory must hold exactly {sorted(expected)}")
    config.verify()
    command = harness_command(
        config, run_dir, task_ids=task_ids, check_gt_only=check_gt_only, no_gt=no_gt
    )
    started = time.monotonic()
    with (run_dir / LOG_FILE).open("xb") as log:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    return HarnessRun(
        directory=run_dir,
        command=tuple(command),
        exit_code=completed.returncode,
        elapsed_seconds=round(time.monotonic() - started, 1),
    )


def reference_failures(log_text: str, tasks: int) -> list[str]:
    """Read the tasks whose reference solution failed from a reference-only log.

    Raises:
        ValueError: If the run did not finish, or the reported pass rate does
            not match the list of failed tasks.
    """
    rates = _PASS_RATE.findall(log_text)
    if len(rates) != 1:
        raise ValueError("the reference run did not report one ground-truth pass rate")
    listed = _FAILED_TASKS.findall(log_text)
    if len(listed) > 1:
        raise ValueError("the reference run reported failed tasks more than once")
    failed = list(ast.literal_eval(listed[0])) if listed else []
    if len(set(failed)) != len(failed) or not all(isinstance(task, str) for task in failed):
        raise ValueError("malformed list of failed tasks")
    if abs(float(rates[0]) - (tasks - len(failed)) / tasks) > 0.0005:
        raise ValueError("the reported pass rate does not match the failed tasks")
    return sorted(failed, key=task_number)


def task_number(task_id: str) -> int:
    """Return the numeric part of a ``BigCodeBench/<n>`` identifier."""
    prefix, _, number = task_id.partition("/")
    if prefix != "BigCodeBench" or not number.isdigit():
        raise ValueError(f"unexpected task identifier {task_id!r}")
    return int(number)


def read_single_results(path: Path, task_ids: Collection[str]) -> dict[str, dict[str, Any]]:
    """Read a result file that must hold exactly one sample for every task."""
    payload = json.loads(Path(path).read_bytes().decode("utf-8"))
    results = payload["eval"]
    if set(results) != set(task_ids):
        raise ValueError(f"{path}: results do not cover exactly the calibrated tasks")
    single: dict[str, dict[str, Any]] = {}
    for task_id, rows in results.items():
        if len(rows) != 1 or rows[0]["status"] not in {"pass", "fail", "timeout"}:
            raise ValueError(f"{path}: {task_id} needs exactly one result with a known status")
        single[task_id] = rows[0]
    return single


def _network_evidence(result: Mapping[str, Any]) -> str | None:
    """Return the first network failure message in the tracebacks of a result."""
    for trace in (result.get("details") or {}).values():
        found = _NETWORK_EVIDENCE.search(str(trace))
        if found:
            return found.group(0)
    return None


def eligibility(
    task_ids: Sequence[str],
    hard_ids: Collection[str],
    reference_runs: Sequence[Collection[str]],
    reference_samples: Mapping[str, Mapping[str, Any]],
    trivial: Mapping[str, Mapping[str, Any]],
    empty: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply rules E1 to E4 to the calibration runs.

    - E1: the reference solution fails in at least one of the reference runs;
    - E2: the trivial program passes;
    - E3: the empty program passes;
    - E4: the reference solution, submitted as a sample, fails with a traceback
      that shows an attempt to reach the network.

    A reference sample that fails for a task no rule excludes is reported as
    ``unexplained``: the calibration is then not conclusive.
    """
    if len(reference_runs) < 2:
        raise ValueError("rule E1 needs at least two independent reference runs")
    known = set(task_ids)
    if len(known) != len(task_ids):
        raise ValueError("duplicate task")
    for failed in reference_runs:
        if not set(failed) <= known:
            raise ValueError("a reference run names an unknown task")
    for label, results in (
        ("reference", reference_samples),
        ("trivial", trivial),
        ("empty", empty),
    ):
        if set(results) != known:
            raise ValueError(f"the {label} run does not cover exactly the calibrated tasks")
    if not set(hard_ids) <= known:
        raise ValueError("a Hard task is not in the dataset")
    excluded: list[dict[str, Any]] = []
    unexplained: list[str] = []
    for task_id in sorted(known, key=task_number):
        failing_runs = [index for index, failed in enumerate(reference_runs) if task_id in failed]
        sample = reference_samples[task_id]
        evidence = _network_evidence(sample) if sample["status"] != "pass" else None
        rules = [
            rule
            for rule, applies in (
                ("E1", bool(failing_runs)),
                ("E2", trivial[task_id]["status"] == "pass"),
                ("E3", empty[task_id]["status"] == "pass"),
                ("E4", evidence is not None),
            )
            if applies
        ]
        if rules:
            excluded.append(
                {
                    "task_id": task_id,
                    "hard": task_id in hard_ids,
                    "rules": rules,
                    "reason": RULES[rules[0]],
                    "evidence": {
                        "reference_runs_failed": failing_runs,
                        "reference_sample_status": sample["status"],
                        "trivial_status": trivial[task_id]["status"],
                        "empty_status": empty[task_id]["status"],
                        "network_message": evidence,
                    },
                }
            )
        elif sample["status"] != "pass":
            unexplained.append(task_id)
    excluded_ids = {row["task_id"] for row in excluded}
    return {
        "rules": dict(RULES),
        "tasks": len(known),
        "hard": len(set(hard_ids)),
        "excluded": excluded,
        "excluded_count": len(excluded),
        "eligible": len(known) - len(excluded),
        "eligible_hard": len(set(hard_ids) - excluded_ids),
        "unexplained": unexplained,
    }


def _require_success(name: str, run: HarnessRun) -> None:
    if run.exit_code != 0:
        raise RuntimeError(f"harness run {name} exited with {run.exit_code}; see {run.log}")


def calibrate(
    config: HarnessConfig,
    hard_ids: Collection[str],
    directory: Path,
    *,
    task_ids: Sequence[str] | None = None,
    runner: Callable[..., HarnessRun] | None = None,
) -> dict[str, Any]:
    """Run the five calibration runs and apply the eligibility rules.

    Two reference-only runs, then the reference solutions, the trivial programs
    and the empty programs submitted as samples. Every run gets a new directory
    under ``directory``. A failed container is an infrastructure error and
    stops the calibration: nothing is retried or repaired here.
    """
    execute = runner if runner is not None else run_harness
    dataset = read_harness_dataset(config.dataset_jsonl)
    by_id = {row["task_id"]: row for row in dataset}
    selected = list(task_ids) if task_ids is not None else list(by_id)
    if len(set(selected)) != len(selected) or not set(selected) <= set(by_id):
        raise ValueError("task_ids must be distinct tasks of the dataset")
    selective = selected if task_ids is not None else None
    directory = Path(directory)
    runs: list[dict[str, Any]] = []

    def record(name: str, run: HarnessRun, **extra: Any) -> None:
        runs.append(
            {
                "name": name,
                "exit_code": run.exit_code,
                "elapsed_seconds": run.elapsed_seconds,
                "arguments": list(run.command[run.command.index(config.image) + 1 :]),
                "log_sha256": digest_file(run.log),
                **extra,
            }
        )

    reference_runs: list[list[str]] = []
    for name in ("reference-a", "reference-b"):
        run = execute(config, directory / name, task_ids=selective, check_gt_only=True)
        _require_success(name, run)
        log_text = run.log.read_bytes().decode("utf-8", errors="replace")
        failed = reference_failures(log_text, len(selected))
        reference_runs.append(failed)
        record(name, run, failed=failed)
    samples: dict[str, dict[str, dict[str, Any]]] = {}
    kinds: tuple[ControlKind, ...] = ("reference", "trivial", "empty")
    for kind in kinds:
        name = f"samples-{kind}"
        submitted = control_samples([by_id[task] for task in selected], kind)
        path = write_samples(submitted, directory / name)
        run = execute(config, directory / name, task_ids=selective, no_gt=True)
        _require_success(name, run)
        results = read_single_results(run.results, selected)
        for row in submitted:
            # The harness reports the program it ran: it must be the submitted one.
            task = row["task_id"]
            ran = by_id[task]["code_prompt"] + "\n    pass\n" + row["solution"]
            if results[task]["solution"] != (ran if config.calibrated else row["solution"]):
                raise ValueError(f"{name}: the harness ran another program for {task}")
        samples[kind] = results
        record(
            name,
            run,
            samples_sha256=digest_file(path),
            results_sha256=digest_file(run.results),
            statuses={
                status: sum(1 for row in results.values() if row["status"] == status)
                for status in ("pass", "fail", "timeout")
            },
        )
    report = eligibility(
        selected,
        set(hard_ids) & set(selected),
        reference_runs,
        samples["reference"],
        samples["trivial"],
        samples["empty"],
    )
    return {
        "scope": "all" if task_ids is None else "subset",
        **report,
        "harness": config.as_dict(),
        "runs": runs,
    }


def harness_version(image: str) -> str:
    """Read the version of the ``bigcodebench`` package installed in the image."""
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--entrypoint",
            "python3",
            image,
            "-c",
            "from importlib.metadata import version; print(version('bigcodebench'))",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"cannot read the harness version: {completed.stderr[-500:]}")
    return completed.stdout.strip()


def docker_resources() -> dict[str, Any]:
    """Record the engine version, processors and memory the containers can use."""
    completed = subprocess.run(
        ["docker", "info", "--format", "{{json .}}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker info failed: {completed.stderr[-500:]}")
    info = json.loads(completed.stdout)
    return {
        "server_version": info.get("ServerVersion"),
        "cpus": info.get("NCPU"),
        "memory_bytes": info.get("MemTotal"),
        "operating_system": info.get("OperatingSystem"),
        "architecture": info.get("Architecture"),
    }


#: Fields that must coincide for calibrations to describe the same oracle.
_SAME_ORACLE: Final = (
    "format_version",
    "benchmark",
    "dataset_revision",
    "split",
    "dataset_file_sha256",
    "hard_revision",
    "hard_file_sha256",
    "tasks",
    "hard",
    "harness",
    "harness_version",
    "rules",
)


def merge_calibrations(calibrations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine complete calibrations: a task excluded by any of them is excluded.

    The reference solution of a few tasks fails only now and then. Rule E1
    excludes a task whose reference fails in at least one reference run, so the
    rule is applied to every run made in the final configuration: the frozen
    exclusions are the union over the calibrations, each task keeping the rules
    and the evidence of every calibration that excluded it. Nothing is ever
    removed by adding a calibration.

    Raises:
        ValueError: If a calibration is partial or inconclusive, or the
            calibrations do not share dataset, Hard subset and harness settings.
    """
    if len(calibrations) < 2:
        raise ValueError("merging needs at least two calibrations")
    first = calibrations[0]
    for calibration in calibrations:
        if calibration.get("scope") != "all" or calibration.get("unexplained") != []:
            raise ValueError("only complete, conclusive calibrations can be merged")
        for name in _SAME_ORACLE:
            if calibration.get(name) != first.get(name) or name not in calibration:
                raise ValueError(f"the calibrations differ in {name!r}")
    merged: dict[str, dict[str, Any]] = {}
    for index, calibration in enumerate(calibrations):
        for row in calibration["excluded"]:
            entry = merged.setdefault(
                row["task_id"],
                {"task_id": row["task_id"], "hard": row["hard"], "rules": [], "evidence": []},
            )
            if entry["hard"] != row["hard"]:
                raise ValueError(f"{row['task_id']}: calibrations disagree on the Hard label")
            entry["rules"] = sorted(set(entry["rules"]) | set(row["rules"]))
            entry["evidence"].append(
                {"calibration": index, "rules": row["rules"], **row["evidence"]}
            )
    excluded = []
    for task_id in sorted(merged, key=task_number):
        entry = merged[task_id]
        excluded.append({**entry, "reason": RULES[entry["rules"][0]]})
        entry_calibrations = {item["calibration"] for item in entry["evidence"]}
        excluded[-1]["excluded_in"] = sorted(entry_calibrations)
    hard_excluded = sum(1 for row in excluded if row["hard"])
    return {
        **{name: first[name] for name in _SAME_ORACLE},
        "scope": "all",
        "merge_rule": "union-of-calibrations-v1",
        "calibrations": len(calibrations),
        "reference_runs": sum(
            1
            for calibration in calibrations
            for run in calibration["runs"]
            if run["name"].startswith("reference-")
        ),
        "excluded": excluded,
        "excluded_count": len(excluded),
        "eligible": first["tasks"] - len(excluded),
        "eligible_hard": first["hard"] - hard_excluded,
        "unexplained": [],
        "per_calibration": [
            {
                "excluded": [row["task_id"] for row in calibration["excluded"]],
                "docker": calibration.get("docker"),
                "runs": calibration["runs"],
            }
            for calibration in calibrations
        ],
    }


# --------------------------------------------------------------------------- evaluation

SANITIZE_POLICY: Final = "bigcodebench-sanitize"
PROMPT_AS_DATA_RULE: Final = "prompt-as-data-v1"
JSON_RESPONSE_RULE: Final = "json-response-v1"
RAW_FILE: Final = "raw.jsonl"
SANITIZED_FILE: Final = "sanitized.jsonl"
PASS_AT_K_FILE: Final = "samples_pass_at_k.json"

#: Program run inside the harness image: the official sanitize, called directly.
SANITIZE_DRIVER: Final = (
    "import json\n"
    "from bigcodebench.sanitize import sanitize\n"
    "with open('/app/raw.jsonl', encoding='utf-8', newline='\\n') as source, "
    "open('/app/sanitized.jsonl', 'x', encoding='utf-8', newline='\\n') as target:\n"
    "    for line in source:\n"
    "        row = json.loads(line)\n"
    "        code = sanitize(row['raw_output'], row['entry_point'])\n"
    "        record = {'request_id': row['request_id'], 'sanitized': code}\n"
    "        target.write(json.dumps(record, ensure_ascii=False) + '\\n')\n"
)

#: Keys of the hierarchical prompt and of the envelope control.
_PROMPT_KEYS: Final = ("main_instruction", "phrase", "position", "prompt")
_PST_KEYS_IN_TEXT: Final = re.compile(r'"(main_instruction|context|position|phrase)"\s*:')
_KEY_IN_TEXT: Final = re.compile(
    r'"(main_instruction|phrase|position|prompt)"\s*:|\[\s*["\'](main_instruction|phrase|position|prompt)["\']\s*\]'
)
_SYNTAX: Final = re.compile(r"SyntaxError|IndentationError|invalid syntax|unexpected indent")
_IMPORT: Final = re.compile(r"No module named|cannot import")


@dataclass(frozen=True)
class SampleResult:
    """What one harness run reported for one exported program."""

    request_id: str
    status: str
    details: Mapping[str, str]


@dataclass(frozen=True)
class Resolved:
    """Final status of one program over the runs, with the flaky flag."""

    request_id: str
    status: str
    flaky: bool
    statuses: tuple[str, ...]
    details: Mapping[str, str]


def sanitize_in_image(
    config: HarnessConfig,
    generations: Sequence[Mapping[str, Any]],
    entry_points: Mapping[str, str],
    work_dir: Path,
) -> dict[str, str]:
    """Extract code with the official ``sanitize`` of the pinned image.

    The function of the harness is called as it is, offline, on the raw output
    and the entry point of the task; nothing is reimplemented or repaired here.
    Returns the sanitized code by request.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    with (work_dir / RAW_FILE).open("x", encoding="utf-8", newline="\n") as handle:
        for generation in generations:
            row = {
                "request_id": generation["request_id"],
                "raw_output": generation["raw_output"],
                "entry_point": entry_points[generation["task_id"]],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--mount",
        f"type=bind,source={work_dir.resolve()},target=/app",
        "--entrypoint",
        "python3",
        config.image,
        "-c",
        SANITIZE_DRIVER,
    ]
    with (work_dir / LOG_FILE).open("xb") as log:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60 * 60,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"sanitize exited with {completed.returncode}; see {work_dir / LOG_FILE}"
        )
    lines = (work_dir / SANITIZED_FILE).read_bytes().decode("utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line]
    expected = [generation["request_id"] for generation in generations]
    if [row["request_id"] for row in rows] != expected:
        raise ValueError("sanitize did not return exactly the submitted outputs, in order")
    return {row["request_id"]: row["sanitized"] for row in rows}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _order_key(seed: int, label: str, request_id: str) -> str:
    return _sha256(f"{seed}|{label}|{request_id}")


def export_samples(
    generations: Sequence[Mapping[str, Any]],
    sanitized: Mapping[str, str],
    *,
    order_seed: int,
    run_label: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Mix every condition in one sample file, in a seeded order of its own.

    All conditions of a task are judged in the same run, under the same load.
    The order of the lines is ``SHA-256(seed | run label | request)``, so two
    runs see two different orders. An output cut at its token budget and an
    empty extraction are not executable: they are left out and stay failures.

    Returns the lines of the sample file and, for each line, the mapping
    ``(line, task, position among the samples of the task)`` to the request.
    """
    by_id = {generation["request_id"]: generation for generation in generations}
    if len(by_id) != len(generations) or set(sanitized) != set(by_id):
        raise ValueError("generations and sanitized outputs must cover the same requests once")
    executable = [
        request
        for request, generation in by_id.items()
        if generation["finish_reason"] != "length" and sanitized[request].strip()
    ]
    ordered = sorted(
        executable, key=lambda request: (_order_key(order_seed, run_label, request), request)
    )
    rows: list[dict[str, str]] = []
    mapping: list[dict[str, Any]] = []
    position: dict[str, int] = {}
    for line, request in enumerate(ordered):
        generation = by_id[request]
        task = generation["task_id"]
        rows.append({"task_id": task, "solution": sanitized[request]})
        mapping.append(
            {
                "line": line,
                "task_id": task,
                "position": position.get(task, 0),
                "condition": generation["condition"],
                "replicate_id": generation["replicate_id"],
                "request_id": request,
                "solution_sha256": _sha256(sanitized[request]),
            }
        )
        position[task] = position.get(task, 0) + 1
    return rows, mapping


def parse_results(
    results_json: Path,
    mapping: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, str]],
    code_prompts: Mapping[str, str],
    *,
    calibrated: bool = True,
) -> list[SampleResult]:
    """Reassociate harness results with the exported programs, fail-closed.

    A result is taken by task and position and accepted only if the program the
    harness reports is the exported one (with the starting code in front when
    calibrated). A row the harness did not report is ``missing``, never a
    failure.

    Raises:
        ValueError: If the harness reports another program for a position.
    """
    reported = json.loads(Path(results_json).read_bytes().decode("utf-8"))["eval"]
    results: list[SampleResult] = []
    for row in mapping:
        task, request = row["task_id"], row["request_id"]
        entries = reported.get(task, [])
        if row["position"] >= len(entries):
            results.append(SampleResult(request, "missing", {"reason": "row_absent"}))
            continue
        entry = entries[row["position"]]
        solution = samples[row["line"]]["solution"]
        ran = code_prompts[task] + "\n    pass\n" + solution if calibrated else solution
        if _sha256(solution) != row["solution_sha256"] or entry["solution"] != ran:
            raise ValueError(f"{task} position {row['position']}: the harness ran another program")
        if entry["status"] not in {"pass", "fail", "timeout"}:
            raise ValueError(f"{task}: unknown harness status {entry['status']!r}")
        details = {str(key): str(value) for key, value in (entry.get("details") or {}).items()}
        results.append(SampleResult(request, entry["status"], details))
    return results


def _passed(status: str) -> bool:
    return status == "pass"


def compare_runs(first: Sequence[SampleResult], second: Sequence[SampleResult]) -> list[str]:
    """Requests whose pass/fail outcome differs between two runs of the same programs."""
    left = {result.request_id: result for result in first}
    right = {result.request_id: result for result in second}
    if set(left) != set(right) or len(left) != len(first) or len(right) != len(second):
        raise ValueError("the two runs do not hold the same programs")
    return sorted(
        request
        for request in left
        if "missing" not in {left[request].status, right[request].status}
        and _passed(left[request].status) != _passed(right[request].status)
    )


def resolve_majority(
    first: Sequence[SampleResult],
    second: Sequence[SampleResult],
    third: Sequence[SampleResult] | None = None,
) -> list[Resolved]:
    """Settle each program: agreement of two runs, or the majority of three.

    A timeout counts as a failure. Where the first two runs disagree a third
    run is required and the program is flagged ``flaky``. A program missing
    from a run it was part of stays ``missing``.

    Raises:
        ValueError: If two runs disagree and no third run settles the program.
    """
    discordant = set(compare_runs(first, second))
    extra = {result.request_id: result for result in third or ()}
    if not set(extra) >= discordant:
        raise ValueError("a third run is required for the programs on which two runs disagree")
    second_by_id = {result.request_id: result for result in second}
    resolved: list[Resolved] = []
    for one in first:
        runs = [one, second_by_id[one.request_id]]
        if one.request_id in discordant:
            runs.append(extra[one.request_id])
        statuses = tuple(run.status for run in runs)
        if "missing" in statuses:
            status, shown = "missing", {"reason": "row_absent"}
        else:
            passes = sum(1 for value in statuses if _passed(value))
            if passes * 2 > len(statuses):
                status, shown = "pass", {}
            else:
                # Details of the earliest run in which the program did not pass.
                failed = next(run for run in runs if not _passed(run.status))
                status, shown = failed.status, dict(failed.details)
        resolved.append(
            Resolved(one.request_id, status, one.request_id in discordant, statuses, shown)
        )
    return resolved


def _defines(code: str, name: str) -> bool:
    """Whether the code defines ``name`` at top level.

    Text that does not parse (prose, or code with a syntax error) is searched
    for a definition line instead, so that a syntax error in a real attempt is
    not mistaken for missing code.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        pattern = rf"^(?:async\s+)?def\s+{re.escape(name)}\s*\(|^class\s+{re.escape(name)}\b"
        return re.search(pattern, code, re.MULTILINE) is not None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                return True
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return True
    return False


def is_json_response(raw_output: str) -> bool:
    """Rule ``json-response-v1``: the answer is, or repeats, a JSON prompt.

    True when the raw output starts with ``{`` or ``[``, or holds at least two
    different keys of the hierarchical prompt in key position.
    """
    if raw_output.lstrip()[:1] in {"{", "["}:
        return True
    return len(set(_PST_KEYS_IN_TEXT.findall(raw_output))) >= 2


def is_prompt_as_data(code: str) -> bool:
    """Rule ``prompt-as-data-v1``: the program handles the prompt as its input data.

    True when the extracted code uses a key of the hierarchical prompt or of
    the envelope (``main_instruction``, ``phrase``, ``position``, ``prompt``)
    as a dictionary key or as a subscript, or passes text holding such a key to
    a ``loads`` call. Code that does not parse is searched as text.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return _KEY_IN_TEXT.search(code) is not None
    for node in ast.walk(tree):
        keys: list[ast.expr | None] = []
        if isinstance(node, ast.Dict):
            keys = list(node.keys)
        elif isinstance(node, ast.Subscript):
            keys = [node.slice]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute | ast.Name)
            and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
            == "loads"
        ):
            for argument in node.args:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and _KEY_IN_TEXT.search(argument.value)
                ):
                    return True
        for key in keys:
            if isinstance(key, ast.Constant) and key.value in _PROMPT_KEYS:
                return True
    return False


def categorize(
    generation: Mapping[str, Any],
    sanitized: str,
    entry_point: str,
    status: str,
    details: Mapping[str, str],
) -> str:
    """Name the outcome of one program: ``pass``, or the first failure category.

    The categories are exclusive and tried in a fixed order: form of the answer
    first (``truncated``, empty extraction as ``no_code``, ``json_response``,
    ``prompt_as_data``, code without the entry point as ``no_code``), then what
    the harness observed (``timeout``, ``syntax_or_compile_error``,
    ``import_error``, ``module_level_exception``, ``assertion_failure``,
    ``test_exception``). A timeout and a cut output are failures.
    """
    if generation["finish_reason"] == "length":
        return "truncated"
    if not sanitized.strip():
        return "no_code"
    if status == "pass":
        return "pass"
    # An answer in JSON never defines the entry point: it is counted as a JSON
    # answer, not as missing code, so that the two can be told apart.
    if is_json_response(generation["raw_output"]):
        return "json_response"
    if is_prompt_as_data(sanitized):
        return "prompt_as_data"
    if not _defines(sanitized, entry_point):
        return "no_code"
    if status == "timeout":
        return "timeout"
    if "ALL" in details:
        message = details["ALL"]
        if _SYNTAX.search(message):
            return "syntax_or_compile_error"
        if _IMPORT.search(message):
            return "import_error"
        return "module_level_exception"
    traces = [trace for trace in details.values() if trace.strip()]
    if not traces:
        return "fail_no_detail"
    if all("AssertionError" in trace.strip().splitlines()[-1] for trace in traces):
        return "assertion_failure"
    return "test_exception"


def to_outcomes(
    generations: Sequence[Mapping[str, Any]],
    sanitized: Mapping[str, str],
    entry_points: Mapping[str, str],
    resolved: Sequence[Resolved],
    analysis_id: str,
) -> tuple[list[OutcomeRecord], list[dict[str, Any]]]:
    """Turn the settled runs into one outcome per planned generation.

    Every generation gets a record: executed programs from the runs, a cut
    output or an empty extraction as failures that were never executed. A
    program the harness did not report is ``missing`` with its reason.

    Raises:
        ValueError: If an executable program has no settled result, or a
            result refers to a program that was not executable.
    """
    by_request = {item.request_id: item for item in resolved}
    outcomes: list[OutcomeRecord] = []
    details: list[dict[str, Any]] = []
    used: set[str] = set()
    for generation in generations:
        request = generation["request_id"]
        code = sanitized[request]
        entry_point = entry_points[generation["task_id"]]
        executable = generation["finish_reason"] != "length" and bool(code.strip())
        settled = by_request.get(request)
        if executable != (settled is not None):
            raise ValueError(f"{request}: settled results do not match the executable programs")
        if settled is None:
            category = categorize(generation, code, entry_point, "fail", {})
            outcome: Outcome = "fail"
        elif settled.status == "missing":
            used.add(request)
            category, outcome = settled.details["reason"], "missing"
        else:
            used.add(request)
            category = categorize(generation, code, entry_point, settled.status, settled.details)
            outcome = "pass" if settled.status == "pass" else "fail"
        outcomes.append(
            OutcomeRecord(
                analysis_id=analysis_id,
                endpoint="functional",
                task_id=generation["task_id"],
                condition=generation["condition"],
                replicate_id=generation["replicate_id"],
                outcome=outcome,
                reason=category,
            )
        )
        details.append(
            {
                "request_id": request,
                "task_id": generation["task_id"],
                "condition": generation["condition"],
                "replicate_id": generation["replicate_id"],
                "extraction_policy": SANITIZE_POLICY,
                "extraction_statuses": extraction_statuses(generation["raw_output"]),
                "sanitized_sha256": _sha256(code),
                "sanitized_lines": sum(1 for line in code.split("\n") if line.strip()),
                "executed": executable,
                "outcome": outcome,
                "reason": category,
                "run_statuses": list(settled.statuses) if settled is not None else [],
                "flaky": settled.flaky if settled is not None else False,
                "finish_reason": generation["finish_reason"],
            }
        )
    if used != set(by_request):
        raise ValueError("a settled result refers to a generation that was not planned")
    return outcomes, details


def harness_prompt_hashes(config: HarnessConfig) -> dict[str, str]:
    """SHA-256 of every ``instruct_prompt`` as the harness in the image loads it."""
    config.verify()
    driver = (
        "import hashlib, json\n"
        "from bigcodebench.data import get_bigcodebench\n"
        "print(json.dumps({task: hashlib.sha256(row['instruct_prompt'].encode('utf-8')).hexdigest()"
        " for task, row in get_bigcodebench().items()}))\n"
    )
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--mount",
            f"type=bind,source={config.dataset_jsonl.resolve().parent},target=/data,readonly",
            "--env",
            f"BIGCODEBENCH_OVERRIDE_PATH=/data/{config.dataset_jsonl.name}",
            "--entrypoint",
            "python3",
            config.image,
            "-c",
            driver,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot read the prompts in the image: {completed.stderr[-500:]}")
    hashes: dict[str, str] = json.loads(completed.stdout.strip().splitlines()[-1])
    return hashes


def evaluate_generations(
    config: HarnessConfig,
    generations: Sequence[Mapping[str, Any]],
    directory: Path,
    *,
    analysis_id: str,
    order_seed: int,
    runner: Callable[..., HarnessRun] | None = None,
    sanitizer: Callable[..., dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Judge every generation with the official harness: two runs, a third on disagreement.

    Returns the outcomes, the per-generation details and a summary of the
    runs. A container that exits with an error stops the evaluation: it is an
    infrastructure failure, not an outcome.
    """
    execute = runner if runner is not None else run_harness
    extract_code = sanitizer if sanitizer is not None else sanitize_in_image
    dataset = {row["task_id"]: row for row in read_harness_dataset(config.dataset_jsonl)}
    unknown = sorted({g["task_id"] for g in generations} - set(dataset))
    if unknown or not generations:
        raise ValueError(f"generations for tasks outside the dataset: {unknown[:5]}")
    entry_points = {task: row["entry_point"] for task, row in dataset.items()}
    code_prompts = {task: row["code_prompt"] for task, row in dataset.items()}
    directory = Path(directory)
    sanitized = extract_code(config, generations, entry_points, directory / "sanitize")
    runs: list[dict[str, Any]] = []

    def one_run(label: str, subset: Sequence[Mapping[str, Any]]) -> list[SampleResult]:
        rows, mapping = export_samples(
            subset,
            {g["request_id"]: sanitized[g["request_id"]] for g in subset},
            order_seed=order_seed,
            run_label=label,
        )
        if not rows:
            return []
        run_dir = directory / label
        path = write_samples(rows, run_dir)
        with (run_dir.parent / f"{label}.mapping.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            for row in mapping:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tasks = sorted({row["task_id"] for row in rows}, key=task_number)
        run = execute(config, run_dir, task_ids=tasks)
        _require_success(label, run)
        results = parse_results(
            run.results, mapping, rows, code_prompts, calibrated=config.calibrated
        )
        summary = run.directory / PASS_AT_K_FILE
        reference = json.loads(summary.read_bytes().decode("utf-8")) if summary.is_file() else {}
        runs.append(
            {
                "name": label,
                "programs": len(rows),
                "tasks": len(tasks),
                "elapsed_seconds": run.elapsed_seconds,
                "samples_sha256": digest_file(path),
                "results_sha256": digest_file(run.results),
                "log_sha256": digest_file(run.log),
                "reference_pass_rate": reference.get("gt_pass_rate"),
                "reference_failed": sorted(reference.get("failed_tasks", []), key=task_number),
                "statuses": {
                    status: sum(1 for result in results if result.status == status)
                    for status in ("pass", "fail", "timeout", "missing")
                },
            }
        )
        return results

    first = one_run("run-a", generations)
    second = one_run("run-b", generations)
    discordant = set(compare_runs(first, second))
    third: list[SampleResult] | None = None
    if discordant:
        by_id = {g["request_id"]: g for g in generations}
        flaky_tasks = {by_id[request]["task_id"] for request in discordant}
        # Every condition of a task with a disagreement is run again together.
        third = one_run("run-c", [g for g in generations if g["task_id"] in flaky_tasks])
    resolved = resolve_majority(first, second, third)
    outcomes, details = to_outcomes(generations, sanitized, entry_points, resolved, analysis_id)
    executed = len(first)
    return {
        "outcomes": outcomes,
        "details": details,
        "summary": {
            "extraction_policy": SANITIZE_POLICY,
            "rules": [JSON_RESPONSE_RULE, PROMPT_AS_DATA_RULE],
            "order_seed": order_seed,
            "generations": len(generations),
            "executed": executed,
            "discordant": len(discordant),
            "discordance_rate": len(discordant) / executed if executed else 0.0,
            "discordance_above_1_percent": bool(executed) and len(discordant) > 0.01 * executed,
            "flaky_tasks": sorted({d["task_id"] for d in details if d["flaky"]}, key=task_number),
            "runs": runs,
        },
    }
