"""Benchmark import: the source text is the benchmark text, with declared changes only."""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from pstparser.empirical import benchmarks
from pstparser.empirical.benchmarks import (
    DatasetPin,
    SourceRecord,
    Transform,
    llmseceval_id_map,
    load_bigcodebench,
    load_llmseceval,
    read_sources,
    write_sources,
)
from pstparser.empirical.cli import app

BCB_REVISION = "1" * 40
LSE_REVISION = "2" * 40
SPLIT = "v9.9.9"
# Line separators that universal-newline reading or str.splitlines would alter.
AWKWARD = (
    "Sum two numbers.\r\nKeep\u2028this\x0band this.  \nYou should write:\n```\ndef f(a, b):\n```"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def bcb_row(number: int, prompt: str, **changes: str) -> dict[str, str]:
    row = {
        "task_id": f"BigCodeBench/{number}",
        "complete_prompt": f"def f{number}():\n    '''doc {number}'''\n",
        "instruct_prompt": prompt,
        "canonical_solution": f"    return SOLUTION_{number}\n",
        "code_prompt": f"def f{number}():\n",
        "test": f"class TestCases{number}: pass\n",
        "entry_point": f"f{number}",
        "doc_struct": "{}",
        "libs": "[]",
    }
    row.update(changes)
    return row


BCB_ROWS = [
    bcb_row(0, AWKWARD),
    bcb_row(1, "Second task, with {braces} and <language>."),
    bcb_row(2, "Third task."),
]


def write_parquet(path: Path, rows: list[dict[str, str]]) -> str:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return sha256_bytes(path.read_bytes())


@pytest.fixture
def parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "bcb.parquet"
    digest = write_parquet(path, BCB_ROWS)
    monkeypatch.setattr(
        benchmarks, "BIGCODEBENCH_PINS", {(BCB_REVISION, SPLIT): DatasetPin(digest, 3)}
    )
    return path


def pin_parquet(monkeypatch: pytest.MonkeyPatch, path: Path, rows: list[dict[str, str]]) -> None:
    digest = write_parquet(path, rows)
    monkeypatch.setattr(
        benchmarks, "BIGCODEBENCH_PINS", {(BCB_REVISION, SPLIT): DatasetPin(digest, len(rows))}
    )


# --------------------------------------------------------------------------- BigCodeBench


def test_bigcodebench_text_is_the_instruct_prompt_byte_for_byte(parquet: Path) -> None:
    records = load_bigcodebench(parquet, BCB_REVISION, SPLIT)

    assert [record.task_id for record in records] == [
        "BigCodeBench/0",
        "BigCodeBench/1",
        "BigCodeBench/2",
    ]
    first = records[0]
    assert first.text == AWKWARD
    assert first.text.encode("utf-8") == AWKWARD.encode("utf-8")
    assert first.source_sha256 == sha256_text(AWKWARD)
    assert first.original_sha256 == first.source_sha256
    assert first.transforms == ()
    assert first.benchmark == "bigcodebench"
    assert first.dataset_revision == BCB_REVISION
    assert first.dataset_file_sha256 == sha256_bytes(parquet.read_bytes())
    assert first.original_id == "BigCodeBench/0"
    assert first.family_id is None
    assert first.text_field == "instruct_prompt"
    # A placeholder-looking string in BigCodeBench is content, not a placeholder.
    assert records[1].text == "Second task, with {braces} and <language>."


def test_bigcodebench_record_has_no_oracle_field(parquet: Path) -> None:
    records = load_bigcodebench(parquet, BCB_REVISION, SPLIT)

    assert {field.name for field in dataclasses.fields(SourceRecord)} == {
        "task_id",
        "text",
        "source_sha256",
        "original_sha256",
        "benchmark",
        "dataset_revision",
        "dataset_file_sha256",
        "original_id",
        "family_id",
        "text_field",
        "transforms",
    }
    serialised = json.dumps([record.as_dict() for record in records])
    for row in BCB_ROWS:
        for oracle in ("canonical_solution", "test", "complete_prompt"):
            assert json.dumps(row[oracle])[1:-1] not in serialised


@pytest.mark.parametrize("oracle", ["test", "canonical_solution"])
def test_prompt_containing_its_test_or_solution_is_refused(
    oracle: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaked = "assert f(1, 2) == 3"
    rows = [bcb_row(0, f"Do it. Hint: {leaked}", **{oracle: leaked}), bcb_row(1, "Other.")]
    path = tmp_path / "leaky.parquet"
    pin_parquet(monkeypatch, path, rows)

    with pytest.raises(ValueError, match=r"BigCodeBench/0.*" + oracle):
        load_bigcodebench(path, BCB_REVISION, SPLIT)


def test_bigcodebench_refuses_an_unpinned_revision_or_split(parquet: Path) -> None:
    with pytest.raises(ValueError, match="no pinned"):
        load_bigcodebench(parquet, "3" * 40, SPLIT)
    with pytest.raises(ValueError, match="no pinned"):
        load_bigcodebench(parquet, BCB_REVISION, "v0.0.0")


def test_bigcodebench_refuses_a_file_with_another_hash(
    parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        benchmarks, "BIGCODEBENCH_PINS", {(BCB_REVISION, SPLIT): DatasetPin("0" * 64, 3)}
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_bigcodebench(parquet, BCB_REVISION, SPLIT)


def test_bigcodebench_refuses_an_unexpected_record_count(
    parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = sha256_bytes(parquet.read_bytes())
    monkeypatch.setattr(
        benchmarks, "BIGCODEBENCH_PINS", {(BCB_REVISION, SPLIT): DatasetPin(digest, 4)}
    )
    with pytest.raises(ValueError, match="3 records, expected 4"):
        load_bigcodebench(parquet, BCB_REVISION, SPLIT)


@pytest.mark.parametrize(
    "rows,message",
    [
        ([bcb_row(0, "a"), bcb_row(0, "b")], "duplicate"),
        ([bcb_row(0, "a"), bcb_row(1, " \n")], "empty"),
    ],
)
def test_bigcodebench_refuses_duplicate_ids_and_empty_prompts(
    rows: list[dict[str, str]],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bad.parquet"
    pin_parquet(monkeypatch, path, rows)
    with pytest.raises(ValueError, match=message):
        load_bigcodebench(path, BCB_REVISION, SPLIT)


def test_the_pinned_datasets_are_the_ones_of_the_protocol() -> None:
    # Hugging Face revision of bigcode/bigcodebench and the file of split v0.1.4;
    # commit of tuhh-softsec/LLMSecEval and its prompt dataset.
    assert {
        ("b74c0d0bf70d2c0bc459be537895cca163007f1a", "v0.1.4"): DatasetPin(
            "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267", 1140
        )
    } == benchmarks.BIGCODEBENCH_PINS
    assert {
        "39e97f046dbb2c4c604d31b86f2a4d3d27827388": DatasetPin(
            "a1b7307678e061faefb8d666eb6cec506d6a40751175716af07703b3f12433cc", 150
        )
    } == benchmarks.LLMSECEVAL_PINS


# ----------------------------------------------------------------------------- LLMSecEval

HEADER = "Generate <language> code for the following: \n "


def lse_row(prompt_id: str, language: str, scenario: str, sample: int, text: str) -> dict[str, Any]:
    cwe = prompt_id.split("_")[0].removeprefix("CWE-")
    extension = "py" if language == "Python" else "c"
    folder = f"experiments_dow/cwe-{cwe}/{scenario}"
    return {
        "Prompt ID": prompt_id,
        "CWE Name": f"Weakness {cwe}",
        "LLM-generated NL Prompt": "generated " + text,
        "Filename": f"{folder}/gen_scenario/x_{scenario}_scenario_copilot_{sample}.{extension}",
        "Language": language,
        "Vulnerable": True,
        "Original Comments by Pearce": "",
        "Naturalness": 4,
        "Expressiveness": 4,
        "Adequacy": 4,
        "Conciseness": 4,
        "Manually-fixed NL Prompt": text,
    }


LSE_ROWS = [
    # 0-2: C, memory weakness; the third carries the misspelt placeholder.
    lse_row("CWE-119_BOF-1a", "C", "eg-a", 0, HEADER + "Copy bytes."),
    lse_row("CWE-119_BOF-1b", "C", "eg-a", 1, HEADER + "Copy more bytes."),
    lse_row("CWE-119_BOF-1c", "C", "eg-a", 2, "Generate <lanuage> code for the following: x"),
    # 3-4: C, memory weakness, a complete family.
    lse_row("CWE-787_OOW-2a", "C", "eg-b", 0, HEADER + "Write a buffer."),
    lse_row("CWE-787_OOW-2b", "C", "eg-b", 1, HEADER + "Write two buffers."),
    # 5-6: Python, one prompt ID used twice, listed out of file-name order.
    lse_row("CWE-20_IIV-2b", "Python", "eg-c", 3, HEADER + "Redirect to <url> if it matches."),
    lse_row("CWE-20_IIV-2b", "Python", "eg-c", 0, HEADER + "Check the target."),
    # 7: C, not a memory weakness: part of S2.
    lse_row("CWE-78_INS-1a", "C", "eg-d", 12, HEADER + "Run getent passwd <username>."),
    # 8: Python, a second family.
    lse_row("CWE-89_SQI-1a", "Python", "eg-e", 0, HEADER + "Unsubscribe the email."),
]


def write_json(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_bytes(json.dumps(rows, ensure_ascii=False, indent=1).encode("utf-8"))
    return sha256_bytes(path.read_bytes())


def pin_json(monkeypatch: pytest.MonkeyPatch, path: Path, rows: list[dict[str, Any]]) -> None:
    digest = write_json(path, rows)
    monkeypatch.setattr(
        benchmarks, "LLMSECEVAL_PINS", {LSE_REVISION: DatasetPin(digest, len(rows))}
    )


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "llmseceval.json"
    pin_json(monkeypatch, path, LSE_ROWS)
    return path


def test_llmseceval_s2_holds_python_and_transferable_c_prompts(dataset: Path) -> None:
    records = load_llmseceval(dataset, LSE_REVISION)

    assert [(r.task_id, r.original_id, r.family_id) for r in records] == [
        ("LSE-005", "CWE-20_IIV-2b", "20-IIV-2"),
        ("LSE-006", "CWE-20_IIV-2b", "20-IIV-2"),
        ("LSE-007", "CWE-78_INS-1a", "78-INS-1"),
        ("LSE-008", "CWE-89_SQI-1a", "89-SQI-1"),
    ]
    assert {r.benchmark for r in records} == {"llmseceval"}
    assert {r.text_field for r in records} == {"Manually-fixed NL Prompt"}
    assert {r.dataset_file_sha256 for r in records} == {sha256_bytes(dataset.read_bytes())}


def test_llmseceval_ids_follow_the_array_and_separate_a_repeated_prompt_id(dataset: Path) -> None:
    by_id = {r.task_id: r for r in load_llmseceval(dataset, LSE_REVISION)}

    # The array order wins over the file-name order: copilot_3 precedes copilot_0.
    assert by_id["LSE-005"].text.endswith("Redirect to <url> if it matches.")
    assert by_id["LSE-006"].text.endswith("Check the target.")
    rows = {row["task_id"]: row for row in llmseceval_id_map(dataset, LSE_REVISION)}
    assert rows["LSE-005"]["filename"].endswith("copilot_3.py")
    assert rows["LSE-006"]["filename"].endswith("copilot_0.py")
    assert rows["LSE-005"]["prompt_id"] == rows["LSE-006"]["prompt_id"] == "CWE-20_IIV-2b"


def test_language_placeholder_is_replaced_once_and_the_others_are_kept(dataset: Path) -> None:
    by_id = {r.task_id: r for r in load_llmseceval(dataset, LSE_REVISION)}

    url = by_id["LSE-005"]
    assert url.text == "Generate Python code for the following: \n Redirect to <url> if it matches."
    assert url.transforms == (Transform("placeholder", "<language>", "Python", 1),)
    assert url.source_sha256 == sha256_text(url.text)
    assert url.original_sha256 == sha256_text(HEADER + "Redirect to <url> if it matches.")
    assert url.original_sha256 != url.source_sha256
    user = by_id["LSE-007"]
    assert user.text == "Generate Python code for the following: \n Run getent passwd <username>."
    assert all("<language>" not in r.text for r in by_id.values())


def test_llmseceval_selections(dataset: Path) -> None:
    python = load_llmseceval(dataset, LSE_REVISION, selection="python-only")
    memory = load_llmseceval(dataset, LSE_REVISION, selection="c-memory")

    assert [r.task_id for r in python] == ["LSE-005", "LSE-006", "LSE-008"]
    # The family with the misspelt placeholder is left out whole.
    assert [(r.task_id, r.family_id) for r in memory] == [
        ("LSE-003", "787-OOW-2"),
        ("LSE-004", "787-OOW-2"),
    ]
    with pytest.raises(ValueError, match="selection"):
        load_llmseceval(dataset, LSE_REVISION, selection="all")


@pytest.mark.parametrize(
    "text,count",
    [
        ("Generate <lanuage> code for the following: x", 0),
        ("Generate <language> code in <language> for the following: x", 2),
    ],
)
def test_selected_record_without_exactly_one_placeholder_is_refused(
    text: str, count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [*LSE_ROWS, lse_row("CWE-89_SQI-1b", "Python", "eg-e", 1, text)]
    path = tmp_path / "bad.json"
    pin_json(monkeypatch, path, rows)

    with pytest.raises(ValueError, match=rf"LSE-009.*{count} time"):
        load_llmseceval(path, LSE_REVISION)


def test_llmseceval_refuses_another_file_or_revision(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="no pinned"):
        load_llmseceval(dataset, "4" * 40)
    monkeypatch.setattr(benchmarks, "LLMSECEVAL_PINS", {LSE_REVISION: DatasetPin("0" * 64, 9)})
    with pytest.raises(ValueError, match="SHA-256"):
        load_llmseceval(dataset, LSE_REVISION)


def test_family_spread_over_two_scenario_folders_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [*LSE_ROWS, lse_row("CWE-89_SQI-1b", "Python", "eg-other", 1, HEADER + "x")]
    path = tmp_path / "bad.json"
    pin_json(monkeypatch, path, rows)

    with pytest.raises(ValueError, match="scenario folder"):
        load_llmseceval(path, LSE_REVISION)


def test_prompt_naming_its_weakness_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [*LSE_ROWS, lse_row("CWE-89_SQI-1b", "Python", "eg-e", 1, HEADER + "Avoid CWE-89.")]
    path = tmp_path / "bad.json"
    pin_json(monkeypatch, path, rows)

    with pytest.raises(ValueError, match=r"LSE-009.*weakness"):
        load_llmseceval(path, LSE_REVISION)


def test_id_map_lists_every_record_without_any_prompt_text(dataset: Path) -> None:
    rows = llmseceval_id_map(dataset, LSE_REVISION)

    assert [row["task_id"] for row in rows] == [f"LSE-{index:03d}" for index in range(9)]
    assert rows[7] == {
        "task_id": "LSE-007",
        "record_index": 7,
        "prompt_id": "CWE-78_INS-1a",
        "filename": "experiments_dow/cwe-78/eg-d/gen_scenario/x_eg-d_scenario_copilot_12.c",
        "family_id": "78-INS-1",
        "cwe": "CWE-78",
        "language": "C",
        "selections": ["S2"],
        "original_sha256": sha256_text(HEADER + "Run getent passwd <username>."),
    }
    assert [row["selections"] for row in rows] == [
        [],
        [],
        [],
        ["c-memory"],
        ["c-memory"],
        ["S2", "python-only"],
        ["S2", "python-only"],
        ["S2"],
        ["S2", "python-only"],
    ]
    serialised = json.dumps(rows)
    assert "Copy bytes" not in serialised and "getent" not in serialised


# ------------------------------------------------------------------------ files and command


def test_sources_round_trip_keeps_unusual_line_separators(parquet: Path, tmp_path: Path) -> None:
    records = load_bigcodebench(parquet, BCB_REVISION, SPLIT)
    path = tmp_path / "sources.jsonl"

    write_sources(records, path)

    assert read_sources(path) == records
    assert b"\r\n" not in path.read_bytes().replace(b"\\r\\n", b"")
    with pytest.raises(FileExistsError):
        write_sources(records, path)


def test_read_sources_refuses_a_text_that_no_longer_matches_its_hash(
    parquet: Path, tmp_path: Path
) -> None:
    records = load_bigcodebench(parquet, BCB_REVISION, SPLIT)
    path = tmp_path / "sources.jsonl"
    write_sources(records, path)
    edited = path.read_bytes().replace(b"Third task.", b"Third task!")
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_bytes(edited)

    with pytest.raises(ValueError, match="BigCodeBench/2"):
        read_sources(tampered)


def test_import_command_writes_sources_and_a_manifest(parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "import" / "sources.jsonl"
    arguments = [
        "import-benchmark",
        "bigcodebench",
        str(parquet),
        "--revision",
        BCB_REVISION,
        "--split",
        SPLIT,
        "--out",
        str(out),
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert read_sources(out) == load_bigcodebench(parquet, BCB_REVISION, SPLIT)
    manifest = json.loads((out.parent / "empirical-import_manifest.json").read_text("utf-8"))
    assert manifest["stage"] == "empirical-import"
    assert manifest["benchmark"] == "bigcodebench"
    assert manifest["dataset_revision"] == BCB_REVISION
    assert manifest["split"] == SPLIT
    assert manifest["dataset_file_sha256"] == sha256_bytes(parquet.read_bytes())
    assert manifest["sources_sha256"] == sha256_bytes(out.read_bytes())
    assert manifest["records"] == 3
    assert manifest["text_field"] == "instruct_prompt"
    assert manifest["transforms"] == []
    # A second import never overwrites the first.
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0
    assert isinstance(again.exception, FileExistsError)


def test_import_command_for_llmseceval_writes_the_id_map(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "import" / "sources.jsonl"
    id_map = tmp_path / "import" / "ids.json"

    result = CliRunner().invoke(
        app,
        [
            "import-benchmark",
            "llmseceval",
            str(dataset),
            "--revision",
            LSE_REVISION,
            "--out",
            str(out),
            "--id-map",
            str(id_map),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [r.task_id for r in read_sources(out)] == ["LSE-005", "LSE-006", "LSE-007", "LSE-008"]
    manifest = json.loads((out.parent / "empirical-import_manifest.json").read_text("utf-8"))
    assert manifest["selection"] == "S2"
    assert manifest["records"] == 4
    assert manifest["families"] == 3
    assert manifest["transforms"] == [
        {"kind": "placeholder", "old": "<language>", "new": "Python", "count": 1, "records": 4}
    ]
    assert manifest["id_map_sha256"] == sha256_bytes(id_map.read_bytes())
    payload = json.loads(id_map.read_text("utf-8"))
    assert payload["dataset_revision"] == LSE_REVISION
    assert payload["dataset_file_sha256"] == sha256_bytes(dataset.read_bytes())
    assert payload["records"] == llmseceval_id_map(dataset, LSE_REVISION)


def test_import_command_refuses_an_unknown_benchmark(parquet: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "import-benchmark",
            "humaneval",
            str(parquet),
            "--revision",
            BCB_REVISION,
            "--out",
            str(tmp_path / "x" / "sources.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "x").exists()
