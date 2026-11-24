"""Contamination by 10-gram overlap, near-duplicate groups and S-NEARDUP."""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical import cli
from pstparser.empirical.analysis import TaskEstimate, _bootstrap
from pstparser.empirical.contamination import (
    ContaminationConfig,
    contamination_rows,
    load_contamination_config,
    near_duplicate_bootstrap,
    near_duplicate_groups,
    ngram_overlap,
    similarity_strata,
)

FROZEN = Path(__file__).resolve().parents[2] / "protocol" / "analysis" / "contamination.json"
SOLUTION = "    total = 0\n    for value in values:\n        total += value\n    return total * 2\n"


def words(text: str) -> list[str]:
    return text.split()


def test_overlap_is_one_with_itself_and_zero_without_shared_ngrams() -> None:
    tokens = words(SOLUTION) * 2
    assert ngram_overlap(tokens, tokens, 10) == 1.0
    assert ngram_overlap([f"a{i}" for i in range(30)], [f"b{i}" for i in range(30)], 10) == 0.0
    # Same tokens, never 10 in the same order: no shared 10-gram.
    assert ngram_overlap(list(reversed(tokens)), tokens, 10) == 0.0


def test_overlap_is_the_share_of_reference_ngrams_found_in_the_output() -> None:
    reference = [str(i) for i in range(12)]  # three distinct 10-grams
    output = ["x", *reference[:11], "y"]  # holds the first two
    assert ngram_overlap(output, reference, 10) == pytest.approx(2 / 3)
    # Extra text around a copied solution does not dilute the share.
    assert ngram_overlap(["prose"] * 50 + reference, reference, 10) == 1.0


def test_overlap_is_undefined_for_a_reference_shorter_than_n() -> None:
    assert ngram_overlap(["a"] * 20, ["a"] * 9, 10) is None
    with pytest.raises(ValueError):
        ngram_overlap(["a"], ["a"], 0)


def test_the_threshold_is_read_from_the_frozen_configuration() -> None:
    config = load_contamination_config(FROZEN)
    assert (config.n, config.threshold) == (10, 0.5)
    assert config.tokenizer.endswith("@c03e6d358207e414f1eca0bb1891e29f1db0e242")


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"format": "x", "n": 10, "threshold": 0.5, "tokenizer": "t"}, "expected format"),
        ({"format": "contamination-config-v1", "n": 10, "threshold": 0.5}, "exactly"),
        ({"format": "contamination-config-v1", "n": 0, "threshold": 0.5, "tokenizer": "t"}, "n"),
        (
            {"format": "contamination-config-v1", "n": 10, "threshold": 1.5, "tokenizer": "t"},
            "threshold",
        ),
    ],
)
def test_a_malformed_configuration_is_refused(document, message, tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_contamination_config(path)


def test_rows_flag_high_similarity_at_the_threshold_and_strata_follow_t() -> None:
    reference = " ".join(str(i) for i in range(11))  # two 10-grams
    half = " ".join(str(i) for i in range(10)) + " z"  # one of two: 0.5
    config = ContaminationConfig(10, 0.5, "words")
    programs = [
        {"task_id": "a", "condition": "T", "replicate_id": 0, "program": half},
        {"task_id": "a", "condition": "H-gold", "replicate_id": 0, "program": "nothing"},
        {"task_id": "b", "condition": "T", "replicate_id": 0, "program": "nothing"},
        {"task_id": "b", "condition": "H-gold", "replicate_id": 0, "program": reference},
        {"task_id": "c", "condition": "T", "replicate_id": 0, "program": reference},
    ]
    rows = contamination_rows(
        programs, {"a": reference, "b": reference, "c": "short"}, words, config
    )
    assert [(r["overlap"], r["high_similarity"]) for r in rows] == [
        (0.5, True),
        (0.0, False),
        (0.0, False),
        (1.0, True),
        (None, None),
    ]
    assert similarity_strata(rows) == {"high": ["a"], "low": ["b"], "undefined": ["c"]}
    with pytest.raises(ValueError, match="no reference solution"):
        contamination_rows(programs, {"a": reference}, words, config)


def test_near_duplicate_groups_are_connected_components() -> None:
    texts = {
        "t1": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        "t2": "alpha beta gamma delta epsilon zeta eta theta iota lambda",  # 9/11
        "t3": "alpha beta gamma delta epsilon zeta eta theta mu lambda",  # 9/11 with t2
        "t4": "completely different words here",
        "t5": "Completely different words, here!",  # same token set as t4
    }
    assert near_duplicate_groups(texts, 0.8) == [["t1", "t2", "t3"], ["t4", "t5"]]
    assert near_duplicate_groups(texts, 0.85) == [["t4", "t5"]]


def test_s_neardup_resamples_groups_as_clusters() -> None:
    estimates = [TaskEstimate(f"t{i}", None, 0.0, float(i % 2), float(i % 2)) for i in range(10)]
    groups = [["t0", "t1", "t2"], ["t5", "t7"]]
    interval, clusters = near_duplicate_bootstrap(estimates, groups, samples=500, seed=7)
    assert clusters == 2 + 5  # two groups and five single tasks
    relabelled = [
        TaskEstimate(
            e.task_id,
            {"t0": "g0", "t1": "g0", "t2": "g0", "t5": "g1", "t7": "g1"}.get(e.task_id, e.task_id),
            e.baseline,
            e.treatment,
            e.difference,
        )
        for e in estimates
    ]
    expected, _, _ = _bootstrap(relabelled, samples=500, seed=7, confidence=0.95)
    assert interval == expected
    with pytest.raises(ValueError, match="two near-duplicate groups"):
        near_duplicate_bootstrap(estimates, [["t0"], ["t0"]], samples=10)


def test_near_duplicates_command(tmp_path: Path) -> None:
    from pstparser.empirical.benchmarks import SourceRecord

    rows = []
    for task, text in (
        ("BigCodeBench/1", "sum the list of values and return it"),
        ("BigCodeBench/2", "sum the list of values and return it!"),
        ("BigCodeBench/3", "draw a histogram"),
    ):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record = SourceRecord(
            task, text, digest, digest, "bigcodebench", "1" * 40, "d" * 64, task, None, "p", ()
        )
        rows.append(json.dumps(record.as_dict()))
    sources = tmp_path / "sources.jsonl"
    sources.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "nd" / "near-duplicates.json"
    result = CliRunner().invoke(cli.app, ["near-duplicates", str(sources), "--out", str(out)])
    assert result.exit_code == 0, result.exception
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["groups"] == [["BigCodeBench/1", "BigCodeBench/2"]]
    assert document["threshold"] == 0.8 and document["tasks"] == 3
