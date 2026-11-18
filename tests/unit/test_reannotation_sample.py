"""Seeded samples of the re-annotation from scratch (plan of 18/11/2026, P-07).

Thirty BigCodeBench tasks of the prefix with judgement units, thirty LLMSecEval
prompts stratified by family; both drawn with a seed before the gold review.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.benchmarks import SourceRecord, write_sources
from pstparser.empirical.cli import app
from pstparser.empirical.sampling import reannotation_sample, seeded_order

DATASET = "d" * 64
SEED = 7319
CODE = "```\nimport os\ndef task_func(x):\n```"
JUDGED = "Compute the sum. It must not modify the input.\n"
UNJUDGED = "Compute the sum.\n"
TAIL = (
    "The function should output with:\n    float: The sum.\n"
    "You should write self-contained code starting with:\n" + CODE
)
LSE_JUDGED = "Generate Python code for the following: \n The code reads a file. It returns it."
LSE_UNJUDGED = "Generate Python code for the following: \n The code reads a file."


def record(
    task: str, text: str, benchmark: str = "bigcodebench", family: str | None = None
) -> SourceRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceRecord(
        task_id=task,
        text=text,
        source_sha256=digest,
        original_sha256=digest,
        benchmark=benchmark,
        dataset_revision="1" * 40,
        dataset_file_sha256=DATASET,
        original_id=task,
        family_id=family,
        text_field="instruct_prompt" if benchmark == "bigcodebench" else "Manually-fixed NL Prompt",
        transforms=(),
    )


def bcb_population() -> list[SourceRecord]:
    """Twenty tasks; the even ones have a judgement unit (a second prose sentence)."""
    return [
        record(f"BigCodeBench/{i}", (JUDGED if i % 2 == 0 else UNJUDGED) + TAIL) for i in range(20)
    ]


def lse_population() -> list[SourceRecord]:
    """Six families of three prompts; family F5 has no judgement unit at all."""
    return [
        record(
            f"LSE-{family}{member}",
            LSE_UNJUDGED if family == 5 else LSE_JUDGED,
            "llmseceval",
            f"F{family}",
        )
        for family in range(6)
        for member in range(3)
    ]


def test_a_plain_draw_is_the_head_of_the_seeded_order() -> None:
    records = bcb_population()
    drawn = reannotation_sample(records, n=5, seed=SEED, dataset_sha256=DATASET, by_family=False)
    ids = [r.task_id for r in records]
    assert [r.task_id for r in drawn] == seeded_order(ids, SEED, "reannotation", DATASET)[:5]
    other = reannotation_sample(
        records, n=5, seed=SEED + 1, dataset_sha256=DATASET, by_family=False
    )
    assert [r.task_id for r in other] != [r.task_id for r in drawn]


def test_a_family_draw_takes_one_prompt_from_each_of_n_seeded_families() -> None:
    records = lse_population()
    drawn = reannotation_sample(records, n=4, seed=SEED, dataset_sha256=DATASET, by_family=True)
    families = [r.family_id for r in drawn]
    assert len(set(families)) == 4
    expected = seeded_order(
        sorted({r.family_id for r in records}), SEED, "reannotation-families", DATASET
    )[:4]
    assert families == expected


def test_too_few_candidates_are_refused() -> None:
    with pytest.raises(ValueError, match="candidate"):
        reannotation_sample(
            lse_population(), n=7, seed=SEED, dataset_sha256=DATASET, by_family=True
        )
    with pytest.raises(ValueError, match="candidate"):
        reannotation_sample(
            bcb_population(), n=21, seed=SEED, dataset_sha256=DATASET, by_family=False
        )


def test_a_record_of_another_dataset_is_refused() -> None:
    records = [
        *bcb_population()[:3],
        dataclasses.replace(bcb_population()[3], dataset_file_sha256="e" * 64),
    ]
    with pytest.raises(ValueError, match="another dataset"):
        reannotation_sample(records, n=2, seed=SEED, dataset_sha256=DATASET, by_family=False)


def _invoke(arguments: list[str]):
    return CliRunner().invoke(app, ["sample-reannotation", *arguments])


def test_the_command_draws_only_tasks_of_the_frozen_sample_with_judgement_units(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "in" / "sources.jsonl"
    write_sources(bcb_population(), sources)
    frozen = tmp_path / "in" / "prefix.json"
    inside = [f"BigCodeBench/{i}" for i in range(12)]
    frozen.write_text(
        json.dumps({"kind": "final-sample", "tasks": [{"task_id": t} for t in inside]}),
        encoding="utf-8",
    )
    out = tmp_path / "re" / "reannotation-bcb.json"
    arguments = [str(sources), "--rule", "bcb-markers-v1", "--n", "4", "--seed", str(SEED)]
    result = _invoke([*arguments, "--sample", str(frozen), "--out", str(out)])
    assert result.exit_code == 0, result.output
    document = json.loads(out.read_text(encoding="utf-8"))
    tasks = [row["task_id"] for row in document["tasks"]]
    assert len(tasks) == 4 and set(tasks) <= set(inside)
    assert all(int(t.split("/")[1]) % 2 == 0 for t in tasks)
    assert document["candidates"] == 6
    assert document["kind"] == "reannotation-sample"
    assert document["judgement_units_rule"] == "judgement-units-v2"
    assert document["sample_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert (out.parent / "empirical-sample-reannotation_manifest.json").is_file()
    again = _invoke([*arguments, "--sample", str(frozen), "--out", str(out)])
    assert isinstance(again.exception, FileExistsError)


def test_the_command_stratifies_llmseceval_by_family(tmp_path: Path) -> None:
    sources = tmp_path / "in" / "sources.jsonl"
    write_sources(lse_population(), sources)
    out = tmp_path / "re" / "reannotation-lse.json"
    arguments = [str(sources), "--rule", "lse-prefix-v1", "--n", "5", "--seed", str(SEED)]
    result = _invoke([*arguments, "--by-family", "--out", str(out)])
    assert result.exit_code == 0, result.output
    document = json.loads(out.read_text(encoding="utf-8"))
    families = [row["family_id"] for row in document["tasks"]]
    assert sorted(families) == ["F0", "F1", "F2", "F3", "F4"]
    assert document["by_family"] is True


def test_the_protocol_samples_are_drawn_from_the_nested_prefix_and_the_s2_selection() -> None:
    root = Path(__file__).resolve().parents[2] / "protocol"
    bcb = json.loads((root / "reannotation/bcb/reannotation-bcb.json").read_text("utf-8"))
    lse = json.loads((root / "reannotation/lse/reannotation-lse.json").read_text("utf-8"))
    prefix = root / "samples/prefix-400/bcb-prefix-400.json"
    assert bcb["sample_sha256"] == hashlib.sha256(prefix.read_bytes()).hexdigest()
    prefix_tasks = {row["task_id"] for row in json.loads(prefix.read_text("utf-8"))["tasks"]}
    assert {row["task_id"] for row in bcb["tasks"]} <= prefix_tasks
    assert bcb["n"] == lse["n"] == 30 and bcb["seed"] == lse["seed"] == SEED
    assert lse["by_family"] and len({row["family_id"] for row in lse["tasks"]}) == 30
