"""Opt-in end-to-end checks of the BigCodeBench adapter with the real harness image.

Same settings as the calibration recipe: ``PST_BCB_IMAGE``, ``PST_BCB_PARQUET``
and ``PST_BCB_NLTK``. Nothing is downloaded and the containers run offline.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from pstparser.empirical.benchmarks import load_bigcodebench
from pstparser.empirical.bigcodebench import (
    HarnessConfig,
    evaluate_generations,
    harness_prompt_hashes,
    read_harness_dataset,
    sanitize_in_image,
    write_harness_dataset,
)
from pstparser.repro.manifest import digest_directory

pytestmark = pytest.mark.slow

REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
SPLIT = "v0.1.4"
# Trial sample: 20 tasks drawn earlier with seed 20261115, plus three tasks
# whose reference needs the network (101, 590, 1012).
TRIAL = [51, 94, 161, 194, 307, 325, 330, 363, 382, 393, 491, 616, 689, 720, 926, 974]
TRIAL += [1013, 1051, 1057, 1113, 101, 590, 1012]
NETWORK = {101, 590, 1012}
# The reference of 363 hangs in most runs, not in all of them.
UNSTABLE = {363}


def _setting(variable: str) -> str:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable}; see the calibration recipe")
    return value


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> HarnessConfig:
    parquet = Path(_setting("PST_BCB_PARQUET"))
    nltk_dir = Path(_setting("PST_BCB_NLTK"))
    image = _setting("PST_BCB_IMAGE")
    dataset = tmp_path_factory.mktemp("data") / "BigCodeBench-v0.1.4.jsonl"
    digest = write_harness_dataset(parquet, REVISION, SPLIT, dataset)
    nltk_digest = digest_directory(nltk_dir)
    assert nltk_digest is not None
    return HarnessConfig(image, dataset, digest, nltk_dir, nltk_digest, parallel=8)


def generation(task: str, condition: str, raw: str) -> dict[str, Any]:
    identifier = hashlib.sha256(f"{task}|{condition}".encode()).hexdigest()
    return {
        "request_id": identifier,
        "task_id": task,
        "condition": condition,
        "replicate_id": 0,
        "raw_output": raw,
        "finish_reason": "eos",
    }


def fenced(code: str) -> str:
    """A fenced block as a model writes it: the closing fence on a line of its own.

    Reference solutions do not end with a line feed; a fence glued to the last
    line would make that line invalid, and the sanitize would drop it.
    """
    return "```python\n" + code.rstrip("\n") + "\n```"


def test_official_sanitize_on_the_shapes_an_answer_can_take(
    config: HarnessConfig, tmp_path: Path
) -> None:
    row = next(
        r for r in read_harness_dataset(config.dataset_jsonl) if r["task_id"].endswith("/51")
    )
    solution = row["code_prompt"] + row["canonical_solution"]
    usage = "print(task_func)\n"
    shapes = {
        "plain": solution,
        "prose": f"Here is the solution:\n\n{fenced(solution)}\n\nIt should work.",
        "two-blocks": f"{fenced(solution)}\nUsage:\n{fenced(usage)}\n",
        "json": json.dumps({"code": solution}),
        "refusal": "I cannot help with that.",
    }
    generations = [generation(row["task_id"], name, raw) for name, raw in shapes.items()]

    sanitized = sanitize_in_image(
        config, generations, {row["task_id"]: row["entry_point"]}, tmp_path / "sanitize"
    )

    by_shape = {g["condition"]: sanitized[g["request_id"]] for g in generations}
    # Prose, fences and the usage example of the second block are dropped; the
    # solution comes back whole, last line included.
    for name in ("plain", "prose", "two-blocks"):
        assert by_shape[name].strip("\n") == solution.strip("\n"), name
    # A JSON answer is a valid Python expression: it comes back as such.
    assert by_shape["json"] == shapes["json"]
    # Prose alone is not repaired into code.
    assert "def task_func" not in by_shape["refusal"]


def test_trial_sample_end_to_end(config: HarnessConfig, tmp_path: Path) -> None:
    dataset = {row["task_id"]: row for row in read_harness_dataset(config.dataset_jsonl)}
    generations = []
    for number in TRIAL:
        task = f"BigCodeBench/{number}"
        row = dataset[task]
        reference = row["code_prompt"] + row["canonical_solution"]
        trivial = row["code_prompt"] + "    return None\n"
        generations += [
            generation(task, "reference", f"Solution:\n{fenced(reference)}\nDone."),
            generation(task, "trivial", fenced(trivial)),
            generation(task, "empty", ""),
        ]

    result = evaluate_generations(
        config, generations, tmp_path / "check", analysis_id="trial", order_seed=20261116
    )

    passed: dict[str, set[int]] = {"reference": set(), "trivial": set(), "empty": set()}
    reasons: dict[str, set[str]] = {"reference": set(), "trivial": set(), "empty": set()}
    for outcome in result["outcomes"]:
        number = int(outcome.task_id.split("/")[1])
        reasons[outcome.condition].add(outcome.reason)
        if outcome.outcome == "pass":
            passed[outcome.condition].add(number)
        assert outcome.outcome in {"pass", "fail"}
    assert len(result["outcomes"]) == 69
    # References: 19 of 23 pass; the three network tasks fail; 363 usually hangs.
    assert set(TRIAL) - NETWORK - UNSTABLE <= passed["reference"]
    assert not passed["reference"] & NETWORK
    assert len(passed["reference"] - UNSTABLE) == 19
    # Neither control passes any task; the empty answer is never executed.
    assert passed["trivial"] == set() and passed["empty"] == set()
    assert reasons["empty"] == {"no_code"}
    executed = [d for d in result["details"] if d["executed"]]
    assert len(executed) == 46 and result["summary"]["executed"] == 46
    assert {d["extraction_policy"] for d in result["details"]} == {"bigcodebench-sanitize"}
    # The second run agrees, except possibly on the unstable task.
    flaky = {int(task.split("/")[1]) for task in result["summary"]["flaky_tasks"]}
    assert flaky <= UNSTABLE


def test_prompts_in_the_image_are_the_imported_ones(config: HarnessConfig) -> None:
    records = load_bigcodebench(Path(_setting("PST_BCB_PARQUET")), REVISION, SPLIT)

    in_image = harness_prompt_hashes(config)

    assert len(in_image) == 1140
    assert in_image == {record.task_id: record.source_sha256 for record in records}
