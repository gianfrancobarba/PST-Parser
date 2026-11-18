"""The sampling commands: frozen files, their hashes and their refusals."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from pstparser.empirical import benchmarks
from pstparser.empirical.benchmarks import DatasetPin, SourceRecord, write_sources
from pstparser.empirical.cli import app
from pstparser.empirical.sampling import (
    Permutation,
    permutation_document,
    read_permutation_document,
    read_sample_document,
    sample,
    sample_document,
    seeded_order,
    stratified_permutation,
)

DATASET = "d" * 64
SEED = 7319
OUTPUT = "The function should output with:\n    int: x\n"
RAISES = "The function should raise the exception for: ValueError: bad.\n"
RULE = "bigcodebench-exceptions-doctest-v1"
EXCLUDE = {"T/7": "network_dependency"}
HARD_REVISION = "2" * 40
SPLIT = "v9.9.9"
HARD = ["T/0", "T/1", "T/2", "T/3", "T/7"]


def record(task: str, text: str = "x", family: str | None = None) -> SourceRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceRecord(
        task_id=task,
        text=text,
        source_sha256=digest,
        original_sha256=digest,
        benchmark="bigcodebench",
        dataset_revision="1" * 40,
        dataset_file_sha256=DATASET,
        original_id=task,
        family_id=family,
        text_field="instruct_prompt",
        transforms=(),
    )


def population() -> list[SourceRecord]:
    """Forty sources: every fourth one has an exceptions section."""
    return [
        record(f"T/{index}", f"Task {index}.\n" + (RAISES if index % 4 == 0 else "") + OUTPUT)
        for index in range(40)
    ]


def families() -> list[SourceRecord]:
    sizes = {"F1": 3, "F2": 3, "F3": 2, "F4": 3, "F5": 3, "F6": 1, "F7": 3}
    return [
        record(f"LSE-{family}-{index}", family=family)
        for family, size in sizes.items()
        for index in range(size)
    ]


def hard_file(tmp_path: Path) -> Path:
    """Parquet naming five sources as Hard; the pin is set by the ``pins`` fixture."""
    path = tmp_path / "in" / "hard.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([{"task_id": task} for task in HARD]), path)
    return path


@pytest.fixture(autouse=True)
def pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = file_sha256(hard_file(tmp_path))
    monkeypatch.setattr(
        benchmarks,
        "BIGCODEBENCH_HARD_PINS",
        {(HARD_REVISION, SPLIT): DatasetPin(digest, len(HARD))},
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_inputs(tmp_path: Path, **calibration: Any) -> tuple[Path, Path]:
    sources = tmp_path / "in" / "sources.jsonl"
    write_sources(population(), sources)
    exclusions = tmp_path / "in" / "exclusions.json"
    document = {
        "format_version": 1,
        "scope": "all",
        "unexplained": [],
        "dataset_file_sha256": DATASET,
        "tasks": 40,
        "excluded": [{"task_id": task, "reason": reason} for task, reason in EXCLUDE.items()],
        **calibration,
    }
    exclusions.write_text(json.dumps(document), encoding="utf-8")
    return sources, exclusions


def sample_arguments(sources: Path, exclusions: Path, tmp_path: Path) -> list[str]:
    return [
        "sample",
        str(sources),
        "--exclude",
        str(exclusions),
        "--seed",
        str(SEED),
        "--n",
        "20",
        "--permutation-out",
        str(tmp_path / "samples" / "permutation.json"),
        "--out",
        str(tmp_path / "samples" / "final.json"),
        "--hard-parquet",
        str(hard_file(tmp_path)),
        "--hard-revision",
        HARD_REVISION,
        "--split",
        SPLIT,
    ]


def test_sample_command_freezes_the_permutation_and_its_prefix(tmp_path: Path) -> None:
    sources, exclusions = write_inputs(tmp_path)
    arguments = sample_arguments(sources, exclusions, tmp_path)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    permutation_file = tmp_path / "samples" / "permutation.json"
    final_file = tmp_path / "samples" / "final.json"
    frozen, final = load(permutation_file), load(final_file)
    assert frozen["kind"] == "permutation" and frozen["method"] == "sha256-order-v1"
    assert frozen["strata_rule"] == RULE
    assert (frozen["seed"], frozen["population"], frozen["eligible"]) == (SEED, 40, 39)
    assert frozen["excluded"] == EXCLUDE
    assert frozen["stratum_sizes"] == {"exceptions/no-doctest": 10, "no-exceptions/no-doctest": 29}
    assert frozen["dataset_file_sha256"] == DATASET
    assert frozen["sources_sha256"] == file_sha256(sources)
    assert frozen["exclusions_sha256"] == file_sha256(exclusions)
    assert [row["position"] for row in frozen["order"]] == list(range(39))
    assert "T/7" not in {row["task_id"] for row in frozen["order"]}
    assert final["kind"] == "final-sample" and final["n"] == 20
    assert final["permutation_sha256"] == file_sha256(permutation_file)
    assert final["tasks"] == frozen["order"][:20]
    assert final["stratum_counts"] == {"exceptions/no-doctest": 5, "no-exceptions/no-doctest": 15}
    # T/7 is Hard but excluded: four eligible Hard tasks remain.
    assert frozen["eligible_hard"] == 4
    assert {row["task_id"] for row in frozen["order"] if row["hard"]} == {
        "T/0",
        "T/1",
        "T/2",
        "T/3",
    }
    manifest = load(tmp_path / "samples" / "empirical-sample_manifest.json")
    assert manifest["sample_sha256"] == file_sha256(final_file)
    assert manifest["permutation_sha256"] == file_sha256(permutation_file)
    assert manifest["seed"] == SEED
    # The frozen files are never overwritten.
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


@pytest.mark.parametrize(
    "calibration,message",
    [
        ({"scope": "subset"}, "complete, conclusive"),
        ({"unexplained": ["T/1"]}, "complete, conclusive"),
        ({"dataset_file_sha256": "e" * 64}, "another dataset file"),
        ({"tasks": 39}, "do not cover every source"),
        ({"excluded": [{"task_id": "T/7", "reason": "model_failed"}]}, "not an admitted one"),
    ],
)
def test_sample_command_refuses_unusable_exclusions(
    calibration: dict[str, Any], message: str, tmp_path: Path
) -> None:
    sources, exclusions = write_inputs(tmp_path, **calibration)

    result = CliRunner().invoke(app, sample_arguments(sources, exclusions, tmp_path))

    assert result.exit_code != 0
    assert message in str(result.exception)
    assert not (tmp_path / "samples").exists()


def test_sample_command_refuses_an_unknown_strata_rule(tmp_path: Path) -> None:
    sources, exclusions = write_inputs(tmp_path)

    result = CliRunner().invoke(
        app, [*sample_arguments(sources, exclusions, tmp_path), "--strata", "by-length"]
    )

    assert result.exit_code != 0
    assert "unknown strata rule" in str(result.exception)


def frozen() -> Permutation:
    return stratified_permutation(
        population(),
        dataset_sha256=DATASET,
        seed=SEED,
        strata=lambda source: "exceptions" if RAISES in source.text else "none",
        exclude=EXCLUDE,
    )


def frozen_files(tmp_path: Path) -> tuple[Path, Path, set[str]]:
    """Write a permutation document and its final sample of 20, with 5 Hard tasks."""
    perm = frozen()
    hard = set(perm.order[20:24]) | {perm.order[0]}
    permutation_file = tmp_path / "samples" / "permutation.json"
    final_file = tmp_path / "samples" / "final.json"
    permutation_file.parent.mkdir(parents=True)
    document = permutation_document(
        perm,
        benchmark="bigcodebench",
        dataset_revision="1" * 40,
        strata_rule="test",
        hard_ids=hard,
    )
    permutation_file.write_text(json.dumps(document), encoding="utf-8")
    final = sample_document(
        "final-sample",
        sample(perm, 20),
        perm,
        hard,
        benchmark="bigcodebench",
        permutation_sha256=file_sha256(permutation_file),
    )
    final_file.write_text(json.dumps(final), encoding="utf-8")
    return permutation_file, final_file, hard


def pilot_arguments(permutation_file: Path, final_file: Path, tmp_path: Path) -> list[str]:
    return [
        "sample-pilot",
        str(permutation_file),
        str(final_file),
        "--seed",
        str(SEED),
        "--n",
        "6",
        "--out",
        str(tmp_path / "pilot" / "pilot.json"),
        "--full-out",
        str(tmp_path / "pilot" / "full.json"),
    ]


def test_documents_round_trip(tmp_path: Path) -> None:
    permutation_file, final_file, hard = frozen_files(tmp_path)
    perm = frozen()

    rebuilt, rebuilt_hard = read_permutation_document(load(permutation_file))

    assert rebuilt == perm
    assert rebuilt_hard == frozenset(hard)
    assert load(permutation_file)["eligible_hard"] == 5
    final = read_sample_document(load(final_file), "final-sample")
    assert final.task_ids == perm.order[:20] and final.seed == SEED
    assert load(final_file)["hard_count"] == 1
    with pytest.raises(ValueError, match="not a pilot-sample document"):
        read_sample_document(load(final_file), "pilot-sample")
    with pytest.raises(ValueError, match="not a permutation document"):
        read_permutation_document(load(final_file))


def test_pilot_command_draws_outside_the_final_sample_and_hard(tmp_path: Path) -> None:
    permutation_file, final_file, hard = frozen_files(tmp_path)
    perm = frozen()
    candidates = [task for task in perm.order[20:] if task not in hard]

    result = CliRunner().invoke(app, pilot_arguments(permutation_file, final_file, tmp_path))

    assert result.exit_code == 0, result.output
    pilot = load(tmp_path / "pilot" / "pilot.json")
    full = load(tmp_path / "pilot" / "full.json")
    pilot_ids = [row["task_id"] for row in pilot["tasks"]]
    assert len(candidates) == 15
    assert pilot["kind"] == "pilot-sample" and pilot["n"] == 6 and pilot["hard_count"] == 0
    assert pilot["purpose"] == "pilot" and pilot["hard_excluded"] == 5
    assert pilot_ids == seeded_order(candidates, SEED, "pilot", DATASET)[:6]
    assert not set(pilot_ids) & set(perm.order[:20])
    assert pilot["final_sha256"] == file_sha256(final_file)
    assert pilot["permutation_sha256"] == file_sha256(permutation_file)
    assert full["kind"] == "full-minus-pilot" and full["n"] == 39 - 6
    assert full["pilot_sha256"] == file_sha256(tmp_path / "pilot" / "pilot.json")
    full_ids = [row["task_id"] for row in full["tasks"]]
    assert full_ids[:20] == list(perm.order[:20])
    assert not set(full_ids) & set(pilot_ids)
    assert full["hard_count"] == 5
    manifest = load(tmp_path / "pilot" / "empirical-sample-pilot_manifest.json")
    assert manifest["pilot_sha256"] == full["pilot_sha256"]


def test_pilot_command_refuses_a_final_sample_of_another_permutation_file(tmp_path: Path) -> None:
    permutation_file, final_file, _ = frozen_files(tmp_path)
    permutation_file.write_text(permutation_file.read_text("utf-8") + "\n", encoding="utf-8")

    result = CliRunner().invoke(app, pilot_arguments(permutation_file, final_file, tmp_path))

    assert result.exit_code != 0
    assert "not drawn from this permutation file" in str(result.exception)
    assert not (tmp_path / "pilot").exists()


def family_arguments(pool: Path, other: Path, out: Path) -> list[str]:
    return [
        "sample-families",
        str(pool),
        "--families",
        "4",
        "--family-size",
        "3",
        "--seed",
        str(SEED),
        "--disjoint-from",
        str(other),
        "--out",
        str(out),
    ]


def test_families_command_draws_whole_families_disjoint_from_the_sample(tmp_path: Path) -> None:
    pool = tmp_path / "in" / "pool.jsonl"
    write_sources(families(), pool)
    other = tmp_path / "in" / "sample.jsonl"
    write_sources([record("LSE-S-0", family="S1"), record("LSE-S-1", family="S1")], other)
    out = tmp_path / "pilot" / "families.json"

    result = CliRunner().invoke(app, family_arguments(pool, other, out))

    assert result.exit_code == 0, result.output
    document = load(out)
    assert document["kind"] == "family-sample"
    assert (document["n"], document["candidate_families"], document["family_size"]) == (12, 5, 3)
    expected = seeded_order(["F1", "F2", "F4", "F5", "F7"], SEED, "pilot-families", DATASET)[:4]
    assert document["families"] == expected
    assert Counter(row["family_id"] for row in document["tasks"]) == dict.fromkeys(expected, 3)
    assert document["sources_sha256"] == file_sha256(pool)
    assert document["disjoint_from_sha256"] == file_sha256(other)
    # No prompt text in the frozen file, only identifiers and hashes.
    assert set(document["tasks"][0]) == {"task_id", "family_id", "source_sha256"}


def test_families_command_refuses_a_family_shared_with_the_sample(tmp_path: Path) -> None:
    pool = tmp_path / "in" / "pool.jsonl"
    write_sources(families(), pool)
    other = tmp_path / "in" / "sample.jsonl"
    shared = [record(f"LSE-other-{name}", family=name) for name in ("F1", "F2", "F4", "F5", "F7")]
    write_sources(shared, other)

    result = CliRunner().invoke(
        app, family_arguments(pool, other, tmp_path / "pilot" / "families.json")
    )

    assert result.exit_code != 0
    assert "families shared with the other set" in str(result.exception)
    assert not (tmp_path / "pilot").exists()


def test_sample_command_needs_the_hard_subset_for_bigcodebench(tmp_path: Path) -> None:
    sources, exclusions = write_inputs(tmp_path)
    arguments = sample_arguments(sources, exclusions, tmp_path)
    arguments = arguments[: arguments.index("--hard-parquet")]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert "needs its Hard subset" in str(result.exception)
    assert not (tmp_path / "samples").exists()


def hard_arguments(permutation_file: Path, pilot_file: Path, out: Path) -> list[str]:
    return ["sample-hard", str(permutation_file), "--pilot", str(pilot_file), "--out", str(out)]


def test_hard_command_writes_the_census_of_eligible_hard_tasks(tmp_path: Path) -> None:
    permutation_file, final_file, hard = frozen_files(tmp_path)
    perm = frozen()
    assert (
        CliRunner().invoke(app, pilot_arguments(permutation_file, final_file, tmp_path)).exit_code
        == 0
    )
    pilot_file = tmp_path / "pilot" / "pilot.json"
    out = tmp_path / "hard" / "hard.json"

    result = CliRunner().invoke(app, hard_arguments(permutation_file, pilot_file, out))

    assert result.exit_code == 0, result.output
    census = load(out)
    ids = [row["task_id"] for row in census["tasks"]]
    assert census["kind"] == "hard-census" and census["n"] == 5 == census["hard_count"]
    assert ids == [task for task in perm.order if task in hard]
    assert [row["position"] for row in census["tasks"]] == [perm.order.index(t) for t in ids]
    assert not set(ids) & {row["task_id"] for row in load(pilot_file)["tasks"]}
    assert not set(ids) & set(EXCLUDE)
    assert census["pilot_sha256"] == file_sha256(pilot_file)
    assert census["permutation_sha256"] == file_sha256(permutation_file)
    manifest = load(tmp_path / "hard" / "empirical-sample-hard_manifest.json")
    assert manifest["census_sha256"] == file_sha256(out) and manifest["n"] == 5


def test_hard_command_refuses_a_pilot_of_another_permutation_or_inside_hard(tmp_path: Path) -> None:
    permutation_file, final_file, hard = frozen_files(tmp_path)
    assert (
        CliRunner().invoke(app, pilot_arguments(permutation_file, final_file, tmp_path)).exit_code
        == 0
    )
    pilot_file = tmp_path / "pilot" / "pilot.json"
    document = load(pilot_file)
    document["tasks"][0]["task_id"] = sorted(hard)[0]
    inside = tmp_path / "pilot" / "inside.json"
    inside.write_text(json.dumps(document), encoding="utf-8")

    inside_result = CliRunner().invoke(
        app, hard_arguments(permutation_file, inside, tmp_path / "a" / "h.json")
    )
    document["permutation_sha256"] = "0" * 64
    other = tmp_path / "pilot" / "other.json"
    other.write_text(json.dumps(document), encoding="utf-8")
    other_result = CliRunner().invoke(
        app, hard_arguments(permutation_file, other, tmp_path / "b" / "h.json")
    )

    assert inside_result.exit_code != 0
    assert "a pilot task is in the Hard subset" in str(inside_result.exception)
    assert other_result.exit_code != 0
    assert "not drawn from this permutation file" in str(other_result.exception)
    assert not (tmp_path / "a" / "h.json").exists() and not (tmp_path / "b" / "h.json").exists()
