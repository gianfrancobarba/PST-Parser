"""The parser outputs on the frozen gold of the prefix enter the freeze F4 (A3, 19/11/2026).

The prefix predictions (run ``hauto-prefix-01``) were retargeted on the frozen
gold and annotated with the primary policy ``auto-strict-v1``; S_parse, its
failures, the H-auto annotations, the predictions and the score are frozen so
that the parser gate after F4 reads only frozen files.
"""

import hashlib
import json
from pathlib import Path

from pstparser.empirical.freeze import read_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "protocol" / "freeze-spec.yaml"
PREFIX = ROOT / "protocol" / "hauto" / "prefix"
FILES = (
    "s-parse-auto-strict-v1.json",
    "failures-auto-strict-v1.jsonl",
    "hauto-auto-strict-v1.jsonl",
    "empirical-auto-annotate_manifest.json",
    "predictions.jsonl",
    "id-map.json",
    "score-results.json",
    "run-identity.json",
    "record.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_freeze_lists_the_prefix_outputs_and_the_parser_model() -> None:
    items, _ = read_spec(SPEC)
    expected = {f"protocol/hauto/prefix/{name}" for name in FILES}
    expected.add("protocol/hauto/parser-model.json")
    assert expected <= set(items)


def test_s_parse_and_its_failures_cover_the_frozen_prefix() -> None:
    document = json.loads((PREFIX / "s-parse-auto-strict-v1.json").read_text("utf-8"))
    sample = json.loads(
        (ROOT / "protocol/samples/prefix-600/bcb-prefix-600.json").read_text("utf-8")
    )
    failures = [
        json.loads(line)
        for line in (PREFIX / "failures-auto-strict-v1.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    assert document["policy"] == "auto-strict-v1"
    tasks = set(document["tasks"])
    failed = {row["task_id"] for row in failures}
    assert len(tasks) == document["converted"] and len(failed) == document["failures"]
    assert not tasks & failed
    prefix = {row["task_id"] for row in sample["tasks"]}
    assert tasks | failed == prefix and document["n"] == len(prefix) == 600


def test_the_record_binds_the_run_the_parser_and_the_frozen_gold() -> None:
    record = json.loads((PREFIX / "record.json").read_text("utf-8"))
    identity = json.loads((PREFIX / "run-identity.json").read_text("utf-8"))
    assert identity["run_id"] == record["run_id"] == "hauto-prefix-01"
    assert identity["parser_identity_sha256"] == _sha(ROOT / "protocol/hauto/parser.json")
    assert record["gold_sha256"] == _sha(
        ROOT / "data/annotated/downstream/bcb-gold/annotations.jsonl"
    )
    assert record["require_status"] == "frozen"
    for name in ("predictions.jsonl", "id-map.json", "s-parse-auto-strict-v1.json"):
        assert record["files"][name] == _sha(PREFIX / name), name
    assert record["parser_model_sha256"] == _sha(ROOT / "protocol/hauto/parser-model.json")
