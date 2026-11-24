"""BigCodeBench adapter: export, result parsing, majority rule, failure categories."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pstparser.empirical import benchmarks, bigcodebench
from pstparser.empirical.benchmarks import DatasetPin
from pstparser.empirical.bigcodebench import (
    HarnessConfig,
    HarnessRun,
    SampleResult,
    categorize,
    compare_runs,
    evaluate_generations,
    export_samples,
    is_json_response,
    is_prompt_as_data,
    parse_results,
    resolve_majority,
    to_outcomes,
    write_harness_dataset,
)
from pstparser.repro.manifest import digest_directory

IMAGE = "bigcodebench/bigcodebench-evaluate@sha256:" + "a" * 64
REVISION = "1" * 40
SPLIT = "v9.9.9"
TASKS = ["BigCodeBench/0", "BigCodeBench/1", "BigCodeBench/2"]
GOOD = "def task_func():\n    return 1\n"
ASSERTION = (
    "Traceback (most recent call last):\n  File 't.py', line 3, in test_a\nAssertionError: 1 != 2"
)
KEY_ERROR = "Traceback (most recent call last):\n  File 't.py', line 3, in test_b\nKeyError: 'x'"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_row(task_id: str) -> dict[str, str]:
    number = task_id.split("/")[1]
    return {
        "task_id": task_id,
        "complete_prompt": "def task_func():\n    '''doc'''\n",
        "instruct_prompt": f"Solve task {number}.",
        "canonical_solution": f"    return {number}\n",
        "code_prompt": f"import os\nLIMIT = {number}\ndef task_func():\n",
        "test": "class TestCases: pass\n",
        "entry_point": "task_func",
        "doc_struct": "{}",
        "libs": "[]",
    }


CODE_PROMPTS = {task: dataset_row(task)["code_prompt"] for task in TASKS}
ENTRY = dict.fromkeys(TASKS, "task_func")


def generation(task: str, condition: str, raw: str, finish: str = "eos") -> dict[str, Any]:
    return {
        "request_id": sha256(f"{task}|{condition}"),
        "task_id": task,
        "condition": condition,
        "replicate_id": 0,
        "raw_output": raw,
        "finish_reason": finish,
    }


def fence(code: str) -> str:
    return f"Here it is:\n```python\n{code}```\n"


def strip_fence(raw: str) -> str:
    """Stand-in for the official sanitize: the content of the first fenced block."""
    if "```python\n" not in raw:
        return raw if raw.lstrip().startswith(("def ", "import ", "{")) else ""
    return raw.split("```python\n", 1)[1].split("```", 1)[0]


GENERATIONS = [
    generation(task, condition, fence(GOOD.replace("1", str(index))))
    for index, task in enumerate(TASKS)
    for condition in ("T", "H-gold", "E")
]
SANITIZED = {g["request_id"]: strip_fence(g["raw_output"]) for g in GENERATIONS}


# ------------------------------------------------------------------------------- export


def test_export_mixes_the_conditions_and_maps_every_line_back() -> None:
    rows, mapping = export_samples(GENERATIONS, SANITIZED, order_seed=5, run_label="run-a")

    assert len(rows) == len(mapping) == 9
    by_id = {g["request_id"]: g for g in GENERATIONS}
    seen: dict[str, int] = {}
    for line, (row, entry) in enumerate(zip(rows, mapping, strict=True)):
        original = by_id[entry["request_id"]]
        assert entry["line"] == line
        assert (entry["task_id"], entry["condition"], entry["replicate_id"]) == (
            original["task_id"],
            original["condition"],
            0,
        )
        assert row == {"task_id": original["task_id"], "solution": SANITIZED[entry["request_id"]]}
        assert entry["solution_sha256"] == sha256(row["solution"])
        # Position counts the earlier lines of the same task, as the harness does.
        assert entry["position"] == seen.get(entry["task_id"], 0)
        seen[entry["task_id"]] = entry["position"] + 1
    assert {entry["request_id"] for entry in mapping} == set(by_id)
    # Conditions are not grouped: the order is not the order of the journal.
    assert [entry["request_id"] for entry in mapping] != [g["request_id"] for g in GENERATIONS]


def test_export_order_depends_on_the_seed_and_on_the_run() -> None:
    def order(seed: int, label: str) -> list[str]:
        _, mapping = export_samples(GENERATIONS, SANITIZED, order_seed=seed, run_label=label)
        return [entry["request_id"] for entry in mapping]

    assert order(5, "run-a") == order(5, "run-a")
    assert order(5, "run-a") != order(5, "run-b")
    assert order(5, "run-a") != order(6, "run-a")
    assert sorted(order(5, "run-a")) == sorted(order(6, "run-b"))


def test_cut_outputs_and_empty_extractions_are_not_exported() -> None:
    cut = generation(TASKS[0], "T", fence(GOOD), finish="length")
    empty = generation(TASKS[0], "H-gold", "I cannot write this.")
    kept = generation(TASKS[0], "E", fence(GOOD))
    generations = [cut, empty, kept]
    sanitized = {g["request_id"]: strip_fence(g["raw_output"]) for g in generations}

    rows, mapping = export_samples(generations, sanitized, order_seed=1, run_label="run-a")

    assert rows == [{"task_id": TASKS[0], "solution": GOOD}]
    assert [entry["request_id"] for entry in mapping] == [kept["request_id"]]
    with pytest.raises(ValueError, match="cover the same requests"):
        export_samples(generations, {kept["request_id"]: GOOD}, order_seed=1, run_label="run-a")


# ------------------------------------------------------------------------------ results


def harness_json(
    rows: list[dict[str, str]], statuses: dict[tuple[str, int], tuple[str, dict[str, str]]]
) -> dict[str, Any]:
    """Build the result file the harness would write for these sample lines."""
    evaluated: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entries = evaluated.setdefault(row["task_id"], [])
        status, details = statuses.get((row["task_id"], len(entries)), ("pass", {}))
        entries.append(
            {
                "task_id": row["task_id"],
                "solution": CODE_PROMPTS[row["task_id"]] + "\n    pass\n" + row["solution"],
                "status": status,
                "details": details,
            }
        )
    return {"date": "2026-11-16 10:00", "eval": evaluated}


def write_results(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "samples_eval_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_results_are_reassociated_by_task_and_position(tmp_path: Path) -> None:
    rows, mapping = export_samples(GENERATIONS, SANITIZED, order_seed=5, run_label="run-a")
    statuses = {
        (TASKS[0], 1): ("fail", {"test_a": ASSERTION}),
        (TASKS[2], 0): ("timeout", {}),
    }
    path = write_results(tmp_path, harness_json(rows, statuses))

    results = parse_results(path, mapping, rows, CODE_PROMPTS)

    expected = {
        entry["request_id"]: statuses.get((entry["task_id"], entry["position"]), ("pass", {}))
        for entry in mapping
    }
    assert [(r.request_id, r.status, dict(r.details)) for r in results] == [
        (entry["request_id"], *expected[entry["request_id"]]) for entry in mapping
    ]
    assert sorted(r.status for r in results) == ["fail", *["pass"] * 7, "timeout"]


def test_a_solution_swapped_between_two_positions_is_refused(tmp_path: Path) -> None:
    rows, mapping = export_samples(GENERATIONS, SANITIZED, order_seed=5, run_label="run-a")
    payload = harness_json(rows, {})
    # Different tasks hold different programs: swap two entries across tasks.
    first, second = payload["eval"][TASKS[0]][0], payload["eval"][TASKS[1]][0]
    first["solution"], second["solution"] = second["solution"], first["solution"]

    with pytest.raises(ValueError, match="the harness ran another program"):
        parse_results(write_results(tmp_path, payload), mapping, rows, CODE_PROMPTS)


def test_uncalibrated_results_hold_the_bare_program(tmp_path: Path) -> None:
    rows, mapping = export_samples(GENERATIONS, SANITIZED, order_seed=5, run_label="run-a")
    payload = harness_json(rows, {})
    for entries in payload["eval"].values():
        for entry, row in zip(
            entries, [r for r in rows if r["task_id"] == entries[0]["task_id"]], strict=True
        ):
            entry["solution"] = row["solution"]
    path = write_results(tmp_path, payload)

    assert len(parse_results(path, mapping, rows, CODE_PROMPTS, calibrated=False)) == 9
    with pytest.raises(ValueError, match="another program"):
        parse_results(path, mapping, rows, CODE_PROMPTS, calibrated=True)


def test_a_row_the_harness_did_not_report_is_missing_not_failed(tmp_path: Path) -> None:
    rows, mapping = export_samples(GENERATIONS, SANITIZED, order_seed=5, run_label="run-a")
    payload = harness_json(rows, {})
    payload["eval"][TASKS[1]].pop()
    del payload["eval"][TASKS[2]]

    results = parse_results(write_results(tmp_path, payload), mapping, rows, CODE_PROMPTS)

    missing = {r.request_id for r in results if r.status == "missing"}
    expected = {
        entry["request_id"]
        for entry in mapping
        if entry["task_id"] == TASKS[2] or (entry["task_id"] == TASKS[1] and entry["position"] == 2)
    }
    assert missing == expected and len(missing) == 4
    assert all(
        dict(r.details) == {"reason": "row_absent"} for r in results if r.status == "missing"
    )
    assert {r.status for r in results} == {"pass", "missing"}


# ----------------------------------------------------------------------------- majority


def run(*statuses: str) -> list[SampleResult]:
    details = {
        "fail": {"test_a": ASSERTION},
        "timeout": {},
        "pass": {},
        "missing": {"reason": "row_absent"},
    }
    return [
        SampleResult(f"r{index}", status, details[status]) for index, status in enumerate(statuses)
    ]


def test_agreement_needs_no_third_run() -> None:
    first, second = run("pass", "fail", "timeout"), run("pass", "timeout", "fail")

    assert compare_runs(first, second) == []
    resolved = resolve_majority(first, second)
    assert [(r.status, r.flaky, r.statuses) for r in resolved] == [
        ("pass", False, ("pass", "pass")),
        ("fail", False, ("fail", "timeout")),
        ("timeout", False, ("timeout", "fail")),
    ]


def test_disagreement_is_settled_by_the_majority_of_three_and_flagged() -> None:
    first, second = run("pass", "fail", "pass"), run("fail", "pass", "pass")
    third = [SampleResult("r0", "pass", {}), SampleResult("r1", "timeout", {})]

    assert compare_runs(first, second) == ["r0", "r1"]
    resolved = resolve_majority(first, second, third)

    assert [(r.request_id, r.status, r.flaky, r.statuses) for r in resolved] == [
        ("r0", "pass", True, ("pass", "fail", "pass")),
        ("r1", "fail", True, ("fail", "pass", "timeout")),
        ("r2", "pass", False, ("pass", "pass")),
    ]
    # The details are those of the earliest run that did not pass.
    assert dict(resolved[1].details) == {"test_a": ASSERTION}
    assert dict(resolved[0].details) == {}


def test_disagreement_without_a_third_run_is_an_error() -> None:
    first, second = run("pass", "fail"), run("fail", "fail")

    with pytest.raises(ValueError, match="third run is required"):
        resolve_majority(first, second)
    with pytest.raises(ValueError, match="third run is required"):
        resolve_majority(first, second, [SampleResult("r1", "pass", {})])
    with pytest.raises(ValueError, match="same programs"):
        compare_runs(first, run("pass"))


def test_a_missing_row_is_not_a_disagreement_but_needs_a_third_run() -> None:
    first, second = run("pass", "missing"), run("missing", "fail")

    assert compare_runs(first, second) == []
    # One run missing calls a third run; missing again, it stays missing.
    third = run("missing", "missing")
    assert [(r.status, dict(r.details)) for r in resolve_majority(first, second, third)] == [
        ("missing", {"reason": "row_absent"}),
        ("missing", {"reason": "row_absent"}),
    ]


# --------------------------------------------------------------------------- categories

PST_ECHO = '{\n  "prompt": {\n    "main_instruction": [{"position": 0, "phrase": "Sum."}]\n  }\n}'


@pytest.mark.parametrize(
    "raw,finish,code,status,details,category",
    [
        (fence(GOOD), "eos", GOOD, "pass", {}, "pass"),
        (fence(GOOD), "length", GOOD, "pass", {}, "truncated"),
        ("Sorry.", "eos", "", "fail", {}, "no_code"),
        # Prose that the extraction returns as it is: no definition of the entry point.
        (
            "I cannot help.",
            "eos",
            "I cannot help.",
            "fail",
            {"ALL": "SyntaxError: invalid syntax"},
            "no_code",
        ),
        (
            fence("def helper():\n    return 1\n"),
            "eos",
            "def helper():\n    return 1\n",
            "fail",
            {"test_a": ASSERTION},
            "no_code",
        ),
        (PST_ECHO, "eos", PST_ECHO, "fail", {"ALL": "NameError"}, "json_response"),
        ("[1, 2]", "eos", "task_func = [1, 2]", "fail", {"test_a": KEY_ERROR}, "json_response"),
        (
            fence("def task_func():\n    data = {'phrase': 1, 'position': 0}\n    return data\n"),
            "eos",
            "def task_func():\n    data = {'phrase': 1, 'position': 0}\n    return data\n",
            "fail",
            {"test_a": ASSERTION},
            "prompt_as_data",
        ),
        (fence(GOOD), "eos", GOOD, "timeout", {}, "timeout"),
        (
            fence(GOOD),
            "eos",
            "def task_func(:\n",
            "fail",
            {"ALL": "SyntaxError: invalid syntax"},
            "syntax_or_compile_error",
        ),
        (
            fence(GOOD),
            "eos",
            GOOD,
            "fail",
            {"ALL": "ModuleNotFoundError: No module named 'x'"},
            "import_error",
        ),
        (
            fence(GOOD),
            "eos",
            GOOD,
            "fail",
            {"ALL": "ZeroDivisionError: division by zero"},
            "module_level_exception",
        ),
        (
            fence(GOOD),
            "eos",
            GOOD,
            "fail",
            {"test_a": ASSERTION, "test_c": ASSERTION},
            "assertion_failure",
        ),
        (
            fence(GOOD),
            "eos",
            GOOD,
            "fail",
            {"test_a": ASSERTION, "test_b": KEY_ERROR},
            "test_exception",
        ),
        (fence(GOOD), "eos", GOOD, "fail", {}, "fail_no_detail"),
    ],
)
def test_failure_categories(
    raw: str, finish: str, code: str, status: str, details: dict[str, str], category: str
) -> None:
    produced = generation(TASKS[0], "T", raw, finish)

    assert categorize(produced, code, "task_func", status, details) == category


def test_a_passing_program_is_never_given_a_failure_category() -> None:
    echo = generation(TASKS[0], "H-gold", PST_ECHO)

    assert (
        categorize(echo, "def task_func():\n    return {'prompt': 1}\n", "task_func", "pass", {})
        == "pass"
    )


@pytest.mark.parametrize(
    "code,expected",
    [
        ("def task_func(d):\n    return d['main_instruction']\n", True),
        ('def task_func(d):\n    return d["prompt"]["context"]\n', True),
        ("X = {'position': 0, 'phrase': 'a'}\n", True),
        ('import json\nDATA = json.loads(\'{"prompt": "Sum two numbers"}\')\n', True),
        ("from json import loads\nDATA = loads('{\"main_instruction\": 1}')\n", True),
        ("import json\ndef task_func(s):\n    return json.loads(s)['prompt']\n", True),
        # v2: ordinary uses of the same keys are not the prompt.
        ("from json import loads\nDATA = loads('{\"phrase\": 1}')\n", False),
        ("def task_func(df):\n    return df['position'].mean()\n", False),
        ("def task_func(t):\n    return {'prompt': t, 'max_tokens': 5}\n", False),
        # The same words as ordinary names, values or text are not the prompt.
        (
            "def task_func(prompt, position):\n    phrase = prompt + str(position)\n"
            "    return phrase\n",
            False,
        ),
        ("def task_func():\n    return 'phrase'\n", False),
        ("import json\ndef task_func(s):\n    return json.loads(s)\n", False),
        ("X = {'name': 'prompt'}\n", False),
        # Code that does not parse is searched as text.
        ("def task_func(:\n    return d['main_instruction']\n", True),
        ("def task_func(:\n    return d['phrase']\n", False),
        ("def task_func(:\n    return phrase\n", False),
    ],
)
def test_prompt_as_data_rule(code: str, expected: bool) -> None:
    assert is_prompt_as_data(code) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (PST_ECHO, True),
        ('  \n{"code": "def f(): pass"}', True),
        ("[1, 2, 3]", True),
        (
            'The request has "main_instruction": [...] and "phrase": "Sum".\n```python\nx = 1\n```',
            False,
        ),
        # v2: a JSON echo of either prompt counts, a bracketed heading does not.
        ('Input: {"prompt": "Sum."}\n```python\nx = 1\n```', True),
        ("[Solution]\n```python\nx = 1\n```", False),
        (fence(GOOD), False),
        ('It reads the "context": given above.\n```python\nx = 1\n```', False),
        ("def task_func():\n    return {'a': [1]}\n", False),
    ],
)
def test_json_response_rule(raw: str, expected: bool) -> None:
    assert is_json_response(raw) is expected


# ----------------------------------------------------------------------------- outcomes


def test_every_planned_generation_gets_one_outcome() -> None:
    passing = generation(TASKS[0], "T", fence(GOOD))
    failing = generation(TASKS[0], "H-gold", fence(GOOD))
    cut = generation(TASKS[1], "T", fence(GOOD), finish="length")
    refusal = generation(TASKS[1], "H-gold", "No.")
    absent = generation(TASKS[2], "T", fence(GOOD))
    generations = [passing, failing, cut, refusal, absent]
    sanitized = {g["request_id"]: strip_fence(g["raw_output"]) for g in generations}
    resolved = resolve_majority(
        [
            SampleResult(passing["request_id"], "pass", {}),
            SampleResult(failing["request_id"], "timeout", {}),
            SampleResult(absent["request_id"], "missing", {"reason": "row_absent"}),
        ],
        [
            SampleResult(passing["request_id"], "pass", {}),
            SampleResult(failing["request_id"], "fail", {"test_a": ASSERTION}),
            SampleResult(absent["request_id"], "pass", {}),
        ],
        # One run missing: a third run settles it; missing again, one valid outcome.
        [SampleResult(absent["request_id"], "missing", {"reason": "row_absent"})],
    )

    outcomes, details = to_outcomes(generations, sanitized, ENTRY, resolved, "pilot-01")

    assert [(o.task_id, o.condition, o.outcome, o.reason) for o in outcomes] == [
        (TASKS[0], "T", "pass", "pass"),
        (TASKS[0], "H-gold", "fail", "timeout"),
        (TASKS[1], "T", "fail", "truncated"),
        (TASKS[1], "H-gold", "fail", "no_code"),
        (TASKS[2], "T", "missing", "row_absent"),
    ]
    assert {o.endpoint for o in outcomes} == {"functional"}
    assert {o.analysis_id for o in outcomes} == {"pilot-01"}
    assert [d["executed"] for d in details] == [True, True, False, False, True]
    assert details[0]["extraction_policy"] == "bigcodebench-sanitize"
    assert details[0]["extraction_statuses"] == {
        "python-single-block-v1": "invalid_fence",
        "python-fenced-blocks-v2": "ok",
        "python-longest-block-v2": "ok",
    }
    assert details[1]["run_statuses"] == ["timeout", "fail"] and details[1]["flaky"] is False
    assert details[2]["run_statuses"] == [] and details[2]["finish_reason"] == "length"
    assert details[0]["sanitized_sha256"] == sha256(GOOD) and details[0]["sanitized_lines"] == 2


def test_outcomes_refuse_results_that_do_not_match_the_plan() -> None:
    one = generation(TASKS[0], "T", fence(GOOD))
    sanitized = {one["request_id"]: GOOD}

    with pytest.raises(ValueError, match="do not match the executable programs"):
        to_outcomes([one], sanitized, ENTRY, [], "a")
    stray = resolve_majority(
        [SampleResult(one["request_id"], "pass", {}), SampleResult("other", "pass", {})],
        [SampleResult(one["request_id"], "pass", {}), SampleResult("other", "pass", {})],
    )
    with pytest.raises(ValueError, match="was not planned"):
        to_outcomes([one], sanitized, ENTRY, stray, "a")


# --------------------------------------------------------------------------- end to end


def pinned_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a three-task dataset file and pin it as the dataset of the tests."""
    path = tmp_path / "full.parquet"
    pq.write_table(pa.Table.from_pylist([dataset_row(task) for task in TASKS]), path)
    pins = {(REVISION, SPLIT): DatasetPin(hashlib.sha256(path.read_bytes()).hexdigest(), 3)}
    monkeypatch.setattr(benchmarks, "BIGCODEBENCH_PINS", pins)
    monkeypatch.setattr(bigcodebench, "BIGCODEBENCH_PINS", pins)
    return path


@pytest.fixture
def parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return pinned_parquet(tmp_path, monkeypatch)


def nltk_data(tmp_path: Path) -> Path:
    path = tmp_path / "nltk_data"
    path.mkdir()
    (path / "stopwords.zip").write_bytes(b"zip")
    return path


@pytest.fixture
def nltk_dir(tmp_path: Path) -> Path:
    return nltk_data(tmp_path)


@pytest.fixture
def config(parquet: Path, nltk_dir: Path, tmp_path: Path) -> HarnessConfig:
    dataset = tmp_path / "data" / "BigCodeBench.jsonl"
    digest = write_harness_dataset(parquet, REVISION, SPLIT, dataset)
    nltk_digest = digest_directory(nltk_dir)
    assert nltk_digest is not None
    return HarnessConfig(IMAGE, dataset, digest, nltk_dir, nltk_digest, parallel=8)


class FakeHarness:
    """Judge a program by a marker in its text; a flaky one changes between runs."""

    def __init__(self) -> None:
        """Start with no run made."""
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, config: HarnessConfig, run_dir: Path, **options: Any) -> HarnessRun:
        self.calls.append((run_dir.name, list(options["task_ids"])))
        lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").split("\n")
        rows = [json.loads(line) for line in lines if line]
        statuses: dict[tuple[str, int], tuple[str, dict[str, str]]] = {}
        position: dict[str, int] = {}
        for row in rows:
            key = (row["task_id"], position.get(row["task_id"], 0))
            position[row["task_id"]] = key[1] + 1
            if "WRONG" in row["solution"]:
                statuses[key] = ("fail", {"test_a": ASSERTION})
            elif "FLAKY" in row["solution"] and run_dir.name == "run-b":
                statuses[key] = ("timeout", {})
        (run_dir / "samples_eval_results.json").write_text(
            json.dumps(harness_json(rows, statuses)), encoding="utf-8"
        )
        (run_dir / "samples_pass_at_k.json").write_text(
            json.dumps({"gt_pass_rate": 1.0, "failed_tasks": []}), encoding="utf-8"
        )
        (run_dir / "log.txt").write_text("done\n", encoding="utf-8")
        return HarnessRun(run_dir, ("docker", "run", IMAGE), 0, 2.0)


def fake_sanitize(
    config: HarnessConfig, generations: Any, entry_points: Any, work_dir: Path
) -> dict[str, str]:
    work_dir.mkdir(parents=True)
    return {g["request_id"]: strip_fence(g["raw_output"]) for g in generations}


def planned() -> list[dict[str, Any]]:
    return [
        generation(TASKS[0], "T", fence(GOOD)),
        generation(TASKS[0], "H-gold", fence(GOOD + "# WRONG\n")),
        generation(TASKS[0], "E", fence(GOOD + "# other\n")),
        generation(TASKS[1], "T", fence(GOOD + "# FLAKY\n")),
        generation(TASKS[1], "H-gold", fence(GOOD + "# b\n")),
        generation(TASKS[1], "E", "Nothing useful."),
        generation(TASKS[2], "T", fence(GOOD + "# c\n")),
        generation(TASKS[2], "H-gold", fence(GOOD + "# d\n"), finish="length"),
        generation(TASKS[2], "E", fence(GOOD + "# e\n")),
    ]


def test_two_runs_then_a_third_only_for_the_tasks_that_disagree(
    config: HarnessConfig, tmp_path: Path
) -> None:
    harness = FakeHarness()

    # One discordant program out of seven is more than 1%: accepted explicitly.
    result = evaluate_generations(
        config,
        planned(),
        tmp_path / "check",
        analysis_id="pilot-01",
        order_seed=7,
        runner=harness,
        sanitizer=fake_sanitize,
        accept_discordance=True,
    )

    # The third run holds every condition of the one task that disagreed.
    assert harness.calls == [("run-a", TASKS), ("run-b", TASKS), ("run-c", [TASKS[1]])]
    reasons = [(o.task_id, o.condition, o.outcome, o.reason) for o in result["outcomes"]]
    assert reasons == [
        (TASKS[0], "T", "pass", "pass"),
        (TASKS[0], "H-gold", "fail", "assertion_failure"),
        (TASKS[0], "E", "pass", "pass"),
        (TASKS[1], "T", "pass", "pass"),
        (TASKS[1], "H-gold", "pass", "pass"),
        (TASKS[1], "E", "fail", "no_code"),
        (TASKS[2], "T", "pass", "pass"),
        (TASKS[2], "H-gold", "fail", "truncated"),
        (TASKS[2], "E", "pass", "pass"),
    ]
    flaky = [d for d in result["details"] if d["flaky"]]
    assert [(d["task_id"], d["condition"], d["run_statuses"]) for d in flaky] == [
        (TASKS[1], "T", ["pass", "timeout", "pass"])
    ]
    summary = result["summary"]
    assert (summary["generations"], summary["executed"], summary["discordant"]) == (9, 7, 1)
    assert summary["discordance_rate"] == pytest.approx(1 / 7)
    assert summary["discordance_above_1_percent"] is True
    assert summary["discordance_accepted"] is True
    assert summary["flaky_tasks"] == [TASKS[1]]
    assert [run["name"] for run in summary["runs"]] == ["run-a", "run-b", "run-c"]
    assert [run["programs"] for run in summary["runs"]] == [7, 7, 2]
    assert summary["runs"][0]["statuses"] == {"pass": 6, "fail": 1, "timeout": 0, "missing": 0}
    # Two runs see the same programs in two different orders.
    orders = [
        (tmp_path / "check" / f"{name}.mapping.jsonl").read_text(encoding="utf-8")
        for name in ("run-a", "run-b")
    ]
    assert orders[0] != orders[1]


def test_no_third_run_when_the_two_runs_agree(config: HarnessConfig, tmp_path: Path) -> None:
    harness = FakeHarness()
    stable = [g for g in planned() if "FLAKY" not in g["raw_output"]]

    result = evaluate_generations(
        config,
        stable,
        tmp_path / "check",
        analysis_id="pilot-01",
        order_seed=7,
        runner=harness,
        sanitizer=fake_sanitize,
    )

    assert [name for name, _ in harness.calls] == ["run-a", "run-b"]
    assert result["summary"]["discordant"] == 0
    assert result["summary"]["discordance_above_1_percent"] is False
    assert not (tmp_path / "check" / "run-c").exists()


def test_generations_for_an_unknown_task_are_refused(config: HarnessConfig, tmp_path: Path) -> None:
    stray = [generation("BigCodeBench/99", "T", fence(GOOD))]

    with pytest.raises(ValueError, match="outside the dataset"):
        evaluate_generations(
            config,
            stray,
            tmp_path / "check",
            analysis_id="a",
            order_seed=1,
            runner=FakeHarness(),
            sanitizer=fake_sanitize,
        )


# ----------------------------------------------------------------- infrastructure retries


class BrokenHarness(FakeHarness):
    """A harness whose container exits with an error the first ``failures`` times of a run."""

    def __init__(self, failures: dict[str, int]) -> None:
        """Remember how many attempts of each run fail."""
        super().__init__()
        self.failures = dict(failures)

    def __call__(self, config: HarnessConfig, run_dir: Path, **options: Any) -> HarnessRun:
        label = run_dir.name.split("-attempt-")[0]
        if self.failures.get(label, 0) > 0:
            self.failures[label] -= 1
            self.calls.append((run_dir.name, list(options["task_ids"])))
            (run_dir / "log.txt").write_text("killed\n", encoding="utf-8")
            return HarnessRun(run_dir, ("docker", "run", IMAGE), 137, 0.5)
        return super().__call__(config, run_dir, **options)


def stable() -> list[dict[str, Any]]:
    return [g for g in planned() if "FLAKY" not in g["raw_output"]]


def test_a_failed_container_is_retried_in_a_new_directory_and_logged(
    config: HarnessConfig, tmp_path: Path
) -> None:
    harness = BrokenHarness({"run-a": 2})

    result = evaluate_generations(
        config,
        stable(),
        tmp_path / "check",
        analysis_id="a",
        order_seed=7,
        runner=harness,
        sanitizer=fake_sanitize,
    )

    assert [name for name, _ in harness.calls] == [
        "run-a",
        "run-a-attempt-2",
        "run-a-attempt-3",
        "run-b",
    ]
    assert [(a["run"], a["attempt"], a["status"]) for a in result["attempts"]] == [
        ("run-a", 1, "infrastructure_error"),
        ("run-a", 2, "infrastructure_error"),
        ("run-a", 3, "ok"),
        ("run-b", 1, "ok"),
    ]
    assert result["attempts"][0]["detail"] == "exit code 137"
    assert result["attempts"][2]["directory"] == "run-a-attempt-3"
    # The same programs are given to every attempt, and the outcomes are those of a clean run.
    samples = [
        (tmp_path / "check" / name / "samples.jsonl").read_bytes()
        for name in ("run-a", "run-a-attempt-2", "run-a-attempt-3")
    ]
    assert samples[0] == samples[1] == samples[2]
    assert all(o.outcome != "missing" for o in result["outcomes"])
    assert result["summary"]["runs"][0]["attempts"] == 3
    assert result["summary"]["retried_runs"] == 2


def test_three_failed_attempts_are_followed_by_runs_task_by_task(
    config: HarnessConfig, tmp_path: Path
) -> None:
    # An exhausted run costs only the tasks whose own batch fails;
    # here every batch of run-b succeeds, so no program is missing.
    harness = BrokenHarness({"run-b": 3})

    result = evaluate_generations(
        config,
        stable(),
        tmp_path / "check",
        analysis_id="a",
        order_seed=7,
        runner=harness,
        sanitizer=fake_sanitize,
    )

    executed = {d["request_id"] for d in result["details"] if d["executed"]}
    for outcome, detail in zip(result["outcomes"], result["details"], strict=True):
        if detail["request_id"] in executed:
            assert outcome.outcome != "missing"
        else:  # never executed: still observed failures of the model
            assert outcome.outcome == "fail"
    assert [a["status"] for a in result["attempts"] if a["run"] == "run-b"] == [
        "infrastructure_error"
    ] * 3
    batches = [a["run"] for a in result["attempts"] if a["run"].startswith("run-b-task-")]
    assert batches and all(a["status"] == "ok" for a in result["attempts"] if a["run"] in batches)
    assert result["summary"]["runs"][1]["batches_by_task"] == len(batches)


def test_failures_of_the_programs_are_never_retried(config: HarnessConfig, tmp_path: Path) -> None:
    harness = FakeHarness()

    result = evaluate_generations(
        config,
        stable(),
        tmp_path / "check",
        analysis_id="a",
        order_seed=7,
        runner=harness,
        sanitizer=fake_sanitize,
    )

    assert any(o.outcome == "fail" and o.reason == "assertion_failure" for o in result["outcomes"])
    assert [(a["run"], a["attempt"]) for a in result["attempts"]] == [("run-a", 1), ("run-b", 1)]
