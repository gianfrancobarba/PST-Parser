"""Exclusions frozen as the union of several complete calibrations."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pstparser.empirical.bigcodebench import RULES, merge_calibrations
from pstparser.empirical.cli import app

HARNESS = {"image": "name@sha256:" + "a" * 64, "parallel": 8, "calibrated": True}


def excluded(number: int, rules: list[str], hard: bool = False, **evidence: Any) -> dict[str, Any]:
    return {
        "task_id": f"BigCodeBench/{number}",
        "hard": hard,
        "rules": rules,
        "reason": RULES[rules[0]],
        "evidence": {"reference_runs_failed": [0, 1], "network_message": None, **evidence},
    }


def calibration(rows: list[dict[str, Any]], **changes: Any) -> dict[str, Any]:
    document = {
        "format_version": 1,
        "benchmark": "bigcodebench",
        "dataset_revision": "1" * 40,
        "split": "v0.1.4",
        "dataset_file_sha256": "d" * 64,
        "hard_revision": "2" * 40,
        "hard_file_sha256": "e" * 64,
        "scope": "all",
        "rules": dict(RULES),
        "tasks": 40,
        "hard": 6,
        "excluded": rows,
        "excluded_count": len(rows),
        "eligible": 40 - len(rows),
        "eligible_hard": 6 - sum(1 for row in rows if row["hard"]),
        "unexplained": [],
        "harness": dict(HARNESS),
        "harness_version": "0.2.4",
        "docker": {"cpus": 28},
        "runs": [{"name": "reference-a"}, {"name": "reference-b"}, {"name": "samples-reference"}],
    }
    document.update(changes)
    return document


# The first calibration excludes 3 (Hard), 9 and 12; the second 3, 5 and 12.
FIRST = calibration(
    [
        excluded(3, ["E1", "E4"], hard=True, network_message="gaierror"),
        excluded(9, ["E1"], reference_runs_failed=[1]),
        excluded(12, ["E2", "E3"], reference_runs_failed=[]),
    ]
)
SECOND = calibration(
    [
        excluded(3, ["E1", "E4"], hard=True, network_message="gaierror"),
        excluded(5, ["E1"], reference_runs_failed=[0]),
        excluded(12, ["E2"], reference_runs_failed=[]),
    ]
)


def test_a_task_excluded_by_any_calibration_is_excluded() -> None:
    merged = merge_calibrations([FIRST, SECOND])

    assert [row["task_id"] for row in merged["excluded"]] == [
        "BigCodeBench/3",
        "BigCodeBench/5",
        "BigCodeBench/9",
        "BigCodeBench/12",
    ]
    assert (merged["excluded_count"], merged["eligible"], merged["eligible_hard"]) == (4, 36, 5)
    assert (merged["tasks"], merged["hard"]) == (40, 6)
    assert merged["scope"] == "all" and merged["unexplained"] == []
    assert merged["merge_rule"] == "union-of-calibrations-v1"
    assert (merged["calibrations"], merged["reference_runs"]) == (2, 4)
    by_task = {row["task_id"]: row for row in merged["excluded"]}
    assert by_task["BigCodeBench/5"]["excluded_in"] == [1]
    assert by_task["BigCodeBench/9"]["excluded_in"] == [0]
    assert by_task["BigCodeBench/3"]["excluded_in"] == [0, 1]
    # Rules are the union; the reason follows the first rule in E1..E4 order.
    assert by_task["BigCodeBench/12"]["rules"] == ["E2", "E3"]
    assert by_task["BigCodeBench/12"]["reason"] == "trivial_program_passes"
    assert by_task["BigCodeBench/3"]["reason"] == "reference_fails_offline"
    assert by_task["BigCodeBench/9"]["evidence"] == [
        {"calibration": 0, "rules": ["E1"], "reference_runs_failed": [1], "network_message": None}
    ]
    assert len(by_task["BigCodeBench/3"]["evidence"]) == 2
    assert merged["per_calibration"][0]["excluded"] == [
        "BigCodeBench/3",
        "BigCodeBench/9",
        "BigCodeBench/12",
    ]
    assert merged["harness"] == HARNESS


def test_adding_a_calibration_never_removes_an_exclusion() -> None:
    third = calibration([])

    two = merge_calibrations([FIRST, SECOND])
    three = merge_calibrations([FIRST, SECOND, third])

    assert [row["task_id"] for row in three["excluded"]] == [
        row["task_id"] for row in two["excluded"]
    ]
    assert three["reference_runs"] == 6
    assert merge_calibrations([SECOND, FIRST])["excluded_count"] == two["excluded_count"]


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"scope": "subset"}, "complete, conclusive"),
        ({"unexplained": ["BigCodeBench/7"]}, "complete, conclusive"),
        ({"dataset_file_sha256": "f" * 64}, "dataset_file_sha256"),
        ({"hard_file_sha256": "f" * 64}, "hard_file_sha256"),
        ({"harness": {**HARNESS, "parallel": 4}}, "harness"),
        ({"harness": {**HARNESS, "image": "name@sha256:" + "b" * 64}}, "harness"),
        ({"harness_version": "0.2.5"}, "harness_version"),
        ({"tasks": 41}, "tasks"),
    ],
)
def test_calibrations_of_another_oracle_or_incomplete_are_refused(
    changes: dict[str, Any], message: str
) -> None:
    other = copy.deepcopy(SECOND)
    other.update(changes)

    with pytest.raises(ValueError, match=message):
        merge_calibrations([FIRST, other])


def test_merging_needs_two_calibrations_and_a_consistent_hard_label() -> None:
    with pytest.raises(ValueError, match="at least two"):
        merge_calibrations([FIRST])
    other = calibration([excluded(3, ["E1"], hard=False)])
    with pytest.raises(ValueError, match="Hard label"):
        merge_calibrations([FIRST, other])


def test_command_writes_the_merged_exclusions_with_the_hashes_of_their_inputs(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(FIRST), encoding="utf-8")
    second.write_text(json.dumps(SECOND), encoding="utf-8")
    out = tmp_path / "oracles" / "bcb-exclusions.json"
    arguments = ["merge-calibrations", str(first), str(second), "--out", str(out)]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "4 excluded, 36 eligible, 5 eligible Hard" in result.output
    document = json.loads(out.read_text(encoding="utf-8"))
    digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)]
    assert document["calibration_sha256"] == digests
    assert document["excluded_count"] == 4
    # The fields the sampling command checks are all there.
    assert document["format_version"] == 1 and document["scope"] == "all"
    assert document["unexplained"] == [] and document["tasks"] == 40
    assert {row["reason"] for row in document["excluded"]} <= set(RULES.values())
    manifest = json.loads(
        (out.parent / "empirical-merge-calibrations_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["exclusions_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["calibration_sha256"] == digests
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


def test_command_refuses_the_same_calibration_twice(tmp_path: Path) -> None:
    first, copy_of_first = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(FIRST), encoding="utf-8")
    copy_of_first.write_text(json.dumps(FIRST), encoding="utf-8")

    for paths in ([first, first], [first, copy_of_first]):
        result = CliRunner().invoke(
            app, ["merge-calibrations", *map(str, paths), "--out", str(tmp_path / "o" / "x.json")]
        )
        assert result.exit_code != 0
        assert not (tmp_path / "o").exists()
