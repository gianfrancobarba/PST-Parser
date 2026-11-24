"""Opt-in checks of the import against the real, pinned dataset files.

The datasets are not redistributed with the repository. Point the variables at
local copies of the pinned files to run these checks:

- ``PST_BCB_PARQUET``: ``data/v0.1.4-00000-of-00001.parquet`` of ``bigcode/bigcodebench``;
- ``PST_LLMSECEVAL_JSON``: ``Dataset/LLMSecEval-Prompts_dataset.json`` of LLMSecEval.
"""

import os
from collections import Counter
from pathlib import Path

import pytest

from pstparser.empirical.benchmarks import (
    llmseceval_id_map,
    load_bigcodebench,
    load_llmseceval,
)

pytestmark = pytest.mark.slow

BCB_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
LSE_REVISION = "39e97f046dbb2c4c604d31b86f2a4d3d27827388"

# The C-origin prompts of S2: weaknesses 22, 732, 78 and 79, which exist in Python.
C_ORIGIN_S2 = {
    "LSE-042": "CWE-22_ILP-1a",
    "LSE-043": "CWE-22_ILP-1b",
    "LSE-044": "CWE-22_ILP-1c",
    "LSE-099": "CWE-732_IPA-1a",
    "LSE-100": "CWE-732_IPA-1b",
    "LSE-101": "CWE-732_IPA-2a",
    "LSE-102": "CWE-732_IPA-2b",
    "LSE-103": "CWE-732_IPA-2c",
    "LSE-107": "CWE-78_INS-1a",
    "LSE-108": "CWE-78_INS-1b",
    "LSE-109": "CWE-78_INS-1c",
    "LSE-113": "CWE-78_INS-3a",
    "LSE-114": "CWE-78_INS-3b",
    "LSE-115": "CWE-78_INS-3c",
    "LSE-125": "CWE-79_INI-1a",
    "LSE-126": "CWE-79_INI-1b",
    "LSE-127": "CWE-79_INI-1c",
}


def _path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to a local copy of the pinned dataset file")
    return Path(value)


@pytest.fixture(scope="module")
def parquet() -> Path:
    return _path("PST_BCB_PARQUET")


@pytest.fixture(scope="module")
def dataset() -> Path:
    return _path("PST_LLMSECEVAL_JSON")


def test_bigcodebench_has_1140_unique_tasks(parquet: Path) -> None:
    records = load_bigcodebench(parquet, BCB_REVISION, "v0.1.4")

    assert len(records) == 1140
    assert {record.task_id for record in records} == {
        f"BigCodeBench/{number}" for number in range(1140)
    }
    assert all(record.transforms == () for record in records)
    assert all("You should write self-contained code starting with:" in r.text for r in records)


def test_first_fifty_prompts_hold_neither_tests_nor_solutions(parquet: Path) -> None:
    import pyarrow.parquet as pq

    records = load_bigcodebench(parquet, BCB_REVISION, "v0.1.4")
    rows = pq.read_table(parquet, columns=["task_id", "test", "canonical_solution"]).to_pylist()

    for record, row in list(zip(records, rows, strict=True))[:50]:
        assert record.task_id == row["task_id"]
        assert row["test"] not in record.text
        assert row["canonical_solution"] not in record.text


def test_llmseceval_s2_is_100_prompts_in_35_families(dataset: Path) -> None:
    records = load_llmseceval(dataset, LSE_REVISION)

    assert len(records) == 100
    assert len({record.task_id for record in records}) == 100
    assert all(record.task_id.startswith("LSE-") and len(record.task_id) == 7 for record in records)
    assert len({record.family_id for record in records}) == 35
    rows = {row["task_id"]: row for row in llmseceval_id_map(dataset, LSE_REVISION)}
    c_origin = {r.task_id: r.original_id for r in records if rows[r.task_id]["language"] == "C"}
    assert c_origin == C_ORIGIN_S2


def test_llmseceval_python_selection_is_83_prompts_in_29_families(dataset: Path) -> None:
    records = load_llmseceval(dataset, LSE_REVISION, selection="python-only")

    assert len(records) == 83
    sizes = Counter(Counter(record.family_id for record in records).values())
    assert sizes == {3: 27, 1: 2}


def test_llmseceval_memory_selection_leaves_out_the_misspelt_family(dataset: Path) -> None:
    records = load_llmseceval(dataset, LSE_REVISION, selection="c-memory")

    # 50 prompts in 17 families, minus the three of family 119-BOF-1.
    assert len(records) == 47
    families = {record.family_id for record in records}
    assert len(families) == 16
    assert "119-BOF-1" not in families
    assert not families & {r.family_id for r in load_llmseceval(dataset, LSE_REVISION)}


def test_llmseceval_placeholders(dataset: Path) -> None:
    records = load_llmseceval(dataset, LSE_REVISION)

    assert all(
        record.text.startswith("Generate Python code for the following:") for record in records
    )
    assert all("<language>" not in record.text for record in records)
    assert sum("<url>" in record.text for record in records) == 3
    assert sum("<username>" in record.text for record in records) == 3


def test_repeated_prompt_ids_get_distinct_internal_ids(dataset: Path) -> None:
    rows = {row["task_id"]: row for row in llmseceval_id_map(dataset, LSE_REVISION)}

    assert len(rows) == 150
    assert rows["LSE-031"]["prompt_id"] == rows["LSE-032"]["prompt_id"] == "CWE-20_IIV-2b"
    assert rows["LSE-031"]["filename"].endswith("_scenario_copilot_3.py")
    assert rows["LSE-032"]["filename"].endswith("_scenario_copilot_0.py")
    assert rows["LSE-096"]["prompt_id"] == rows["LSE-097"]["prompt_id"] == "CWE-522_IPC-3a"
    assert rows["LSE-096"]["filename"].endswith("_scenario_copilot_0.py")
    assert rows["LSE-097"]["filename"].endswith("_scenario_copilot_1.py")
    assert len({row["prompt_id"] for row in rows.values()}) == 148


# Jaccard >= 0.8 on the word tokens of instruct_prompt, 8 groups and 17 tasks.
NEAR_DUPLICATE_GROUPS = [
    {349, 351, 353},
    {130, 131},
    {330, 333},
    {423, 426},
    {736, 738},
    {894, 895},
    {1097, 1099},
    {1120, 1121},
]


def test_near_duplicate_groups_are_the_eight_of_the_plan(parquet: Path) -> None:
    from pstparser.empirical.contamination import near_duplicate_groups

    records = load_bigcodebench(parquet, BCB_REVISION, "v0.1.4")
    groups = near_duplicate_groups({record.task_id: record.text for record in records})

    found = [{int(task.split("/")[1]) for task in group} for group in groups]
    assert sorted(map(sorted, found)) == sorted(map(sorted, NEAR_DUPLICATE_GROUPS))
    assert sum(len(group) for group in groups) == 17
