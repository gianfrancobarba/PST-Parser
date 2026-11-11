"""Read-only diagnostics distinguish evidence from similarity candidates."""

import json

import pytest

from pstparser.data.audit import audit_records
from pstparser.data.prepare import PreparedRecord
from pstparser.pst.taxonomy import LEAF_PATHS, assemble


def record(index: int, prompt: str, phrase: str) -> PreparedRecord:
    leaves = {path: [] for path in LEAF_PATHS}
    leaves["main_instruction"] = [phrase]
    return PreparedRecord(index, prompt, json.dumps(assemble(leaves)))


def test_audit_reports_missing_text_and_collapsed_source_without_mutating() -> None:
    rows = [record(7, "hello world", "hello"), record(8, "a\n  b", "a b")]
    before = [row.as_dict() for row in rows]
    report = audit_records(rows, partitions={7: "train", 8: "test"})
    assert report["incomplete_count"] == 1
    assert report["incomplete"][0]["index"] == 7
    assert report["incomplete"][0]["uncovered_spans"] == [{"start": 5, "end": 11, "text": " world"}]
    assert report["tolerant_match_count"] == 1
    assert report["tolerant_matches"][0]["level"] == "collapsed"
    assert report["tolerant_matches"][0]["source_text"] == "a\n  b"
    assert [row.as_dict() for row in rows] == before


def test_exact_groups_and_similarity_candidates_have_explicit_cross_split_flags() -> None:
    rows = [
        record(0, "One two", "One two"),
        record(1, " one  TWO ", " one  TWO "),
        record(2, "one two three", "one two three"),
    ]
    report = audit_records(
        rows, partitions={0: "train", 1: "train", 2: "test"}, similarity_threshold=0.6
    )
    assert report["normalised_exact_groups"][0]["indices"] == [0, 1]
    assert not report["normalised_exact_groups"][0]["cross_partition"]
    assert len(report["near_duplicate_candidates"]) == 2
    assert all(pair["cross_partition"] for pair in report["near_duplicate_candidates"])
    assert report["candidate_components"][0]["indices"] == [0, 1, 2]
    assert audit_records(rows, similarity_threshold=0.9)["near_duplicate_candidates"] == []


def test_audit_rejects_ambiguous_ids_and_invalid_thresholds() -> None:
    row = record(0, "hello", "hello")
    with pytest.raises(ValueError, match="unique"):
        audit_records([row, row])
    with pytest.raises(ValueError, match="unknown"):
        audit_records([row], partitions={99: "train"})
    for threshold in [0, 2, float("nan")]:
        with pytest.raises(ValueError, match="threshold"):
            audit_records([row], similarity_threshold=threshold)
