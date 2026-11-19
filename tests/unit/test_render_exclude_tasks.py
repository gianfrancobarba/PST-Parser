"""render --exclude-tasks: the tasks outside S_parse, for the second journal of a session.

Plan of 18/11/2026, P-04 and C-26: one session has a journal with H-auto on
S_parse and a journal without H-auto on the other tasks of the sample. The
second variant file is the gold rendering of exactly the tasks outside S_parse.
"""

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pstparser.empirical.auto import S_PARSE_FORMAT, tasks_digest
from pstparser.empirical.cli import app

TASKS = ["BigCodeBench/1", "BigCodeBench/4", "BigCodeBench/9"]


def write_canonical(path: Path) -> Path:
    rows = []
    for task in TASKS:
        text = f"Write the function of {task}."
        rows.append(
            {
                "task_id": task,
                "text": text,
                "annotation_version": "gold-v2.0",
                "annotations": [
                    {"path": "main_instruction", "start": 0, "end": len(text), "phrase": text}
                ],
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def write_s_parse(path: Path, tasks: list[str], **override: Any) -> Path:
    document = {
        "format": S_PARSE_FORMAT,
        "policy": "auto-strict-v1",
        "tasks": tasks,
        "tasks_sha256": tasks_digest(tasks),
        **override,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def render(*args: str) -> Any:
    return CliRunner().invoke(app, ["render", *args])


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_exclude_tasks_renders_exactly_the_tasks_outside_s_parse(tmp_path: Path) -> None:
    source = write_canonical(tmp_path / "canonical.jsonl")
    s_parse = write_s_parse(tmp_path / "s_parse.json", ["BigCodeBench/4", "BigCodeBench/1"])
    out = tmp_path / "rest.jsonl"
    result = render(str(source), str(out), "--exclude-tasks", str(s_parse), "--envelope")
    assert result.exit_code == 0, result.output
    rendered = rows(out)
    assert {r["task_id"] for r in rendered} == {"BigCodeBench/9"}
    assert [r["condition"] for r in rendered] == ["T", "H-gold", "E"]
    record = json.loads(Path(f"{out}.render.json").read_text("utf-8"))
    assert record["excluded_tasks_sha256"] == tasks_digest(["BigCodeBench/4", "BigCodeBench/1"])


def test_tasks_and_exclude_tasks_partition_the_source(tmp_path: Path) -> None:
    source = write_canonical(tmp_path / "canonical.jsonl")
    s_parse = write_s_parse(tmp_path / "s_parse.json", ["BigCodeBench/1", "BigCodeBench/4"])
    inside, outside = tmp_path / "inside.jsonl", tmp_path / "outside.jsonl"
    assert render(str(source), str(inside), "--tasks", str(s_parse)).exit_code == 0
    assert render(str(source), str(outside), "--exclude-tasks", str(s_parse)).exit_code == 0
    first = {r["task_id"] for r in rows(inside)}
    second = {r["task_id"] for r in rows(outside)}
    assert first.isdisjoint(second) and first | second == set(TASKS)
    record = json.loads(Path(f"{inside}.render.json").read_text("utf-8"))
    assert record["excluded_tasks_sha256"] is None


def test_exclude_tasks_refuses_a_task_absent_from_the_source(tmp_path: Path) -> None:
    source = write_canonical(tmp_path / "canonical.jsonl")
    s_parse = write_s_parse(tmp_path / "s_parse.json", ["BigCodeBench/1", "BigCodeBench/77"])
    result = render(str(source), str(tmp_path / "out.jsonl"), "--exclude-tasks", str(s_parse))
    assert result.exit_code != 0
    assert "absent from the source" in str(result.exception)
    assert not (tmp_path / "out.jsonl").exists()


def test_exclude_tasks_refuses_a_tampered_list_and_both_options(tmp_path: Path) -> None:
    source = write_canonical(tmp_path / "canonical.jsonl")
    tampered = write_s_parse(tmp_path / "bad.json", ["BigCodeBench/1"], tasks_sha256="0" * 64)
    result = render(str(source), str(tmp_path / "a.jsonl"), "--exclude-tasks", str(tampered))
    assert result.exit_code != 0 and "hash" in str(result.exception)
    s_parse = write_s_parse(tmp_path / "s_parse.json", ["BigCodeBench/1"])
    result = render(
        str(source),
        str(tmp_path / "b.jsonl"),
        "--tasks",
        str(s_parse),
        "--exclude-tasks",
        str(s_parse),
    )
    assert result.exit_code != 0 and "not both" in str(result.exception)


def test_exclude_tasks_refuses_an_empty_rendering(tmp_path: Path) -> None:
    source = write_canonical(tmp_path / "canonical.jsonl")
    s_parse = write_s_parse(tmp_path / "s_parse.json", list(TASKS))
    result = render(str(source), str(tmp_path / "out.jsonl"), "--exclude-tasks", str(s_parse))
    assert result.exit_code != 0 and "empty corpus" in str(result.exception)
