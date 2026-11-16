"""parse-prompts (M19): the parser on benchmark prompts, with an index map and gold targets."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from pstparser.config import load_experiment
from pstparser.data.predictions import PredictionRecord, load_predictions
from pstparser.data.prepare import PreparedRecord
from pstparser.inference import parse_prompts as module
from pstparser.pst.target import serialise_target
from pstparser.pst.taxonomy import LEAF_PATHS, assemble

ROOT = Path(__file__).resolve().parents[2]
TEXT = "Write a function. Return a list."
SPANS = [
    {"path": "main_instruction", "start": 0, "end": 17, "phrase": "Write a function."},
    {"path": "context.format", "start": 18, "end": 32, "phrase": "Return a list."},
]


def test_the_gold_target_lists_each_leaf_phrases_in_source_order() -> None:
    values = {path: [] for path in LEAF_PATHS}
    values["main_instruction"] = ["Write a function."]
    values["context.format"] = ["Return a list."]
    assert module.gold_target(list(reversed(SPANS))) == serialise_target(assemble(values))
    empty = serialise_target(assemble({path: [] for path in LEAF_PATHS}))
    assert module.gold_target([]) == empty


def test_records_keep_the_source_order_and_text() -> None:
    records, tasks = module.prepared_records(
        [
            {"task_id": "B/2", "text": TEXT, "annotations": SPANS},
            {"task_id": "B/1", "text": "Other prompt."},
        ]
    )
    assert tasks == ["B/2", "B/1"]
    assert [(r.index, r.prompt) for r in records] == [(0, TEXT), (1, "Other prompt.")]


@pytest.mark.parametrize(
    "rows",
    [
        [{"task_id": "B/1", "text": "a"}, {"task_id": "B/1", "text": "b"}],
        [{"task_id": "B/1", "text": "  "}],
        [{"task_id": "B/1", "text": TEXT, "annotations": [{**SPANS[0], "phrase": "Other."}]}],
        [],
    ],
)
def test_malformed_sources_are_refused(rows: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        module.prepared_records(rows)


def test_parse_prompts_writes_predictions_index_map_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "sources.jsonl"
    rows = [
        {"task_id": "BigCodeBench/7", "text": TEXT, "annotations": SPANS},
        {"task_id": "BigCodeBench/3", "text": "No gold here."},
    ]
    sources.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    seen: list[Sequence[PreparedRecord]] = []

    def fake_generate(**kwargs: Any) -> Iterable[PredictionRecord]:
        seen.append(kwargs["records"])
        for record in kwargs["records"]:
            yield PredictionRecord(record.index, record.prompt, record.target, "{}", 10, 2, "stop")

    monkeypatch.setattr(module.loader, "load_untuned", lambda config: ("model", "tokenizer"))
    monkeypatch.setattr(module.loader, "runtime_provenance", lambda *a: {"fake": True})
    monkeypatch.setattr(module.generation, "generate_predictions", fake_generate)
    monkeypatch.chdir(ROOT)
    config = load_experiment("configs/experiments/baseline.yaml")
    outcome = module.parse_prompts(config, sources, tmp_path / "run")
    assert outcome.records == 2
    predictions = load_predictions(outcome.predictions_path)
    assert [p.prompt for p in predictions] == [TEXT, "No gold here."]
    assert predictions[0].target == module.gold_target(SPANS)
    id_map = json.loads(outcome.id_map_path.read_text("utf-8"))
    assert id_map["tasks"] == ["BigCodeBench/7", "BigCodeBench/3"]
    assert id_map["format"] == "id-map-v1"
    manifest = json.loads(outcome.manifest_path.read_text("utf-8"))
    assert manifest["stage"] == "parse-prompts"
    assert manifest["gold_targets"] == 1 and manifest["generated"] == 2
    assert manifest["inputs"]["sources"] is not None
    assert manifest["adapter"] is None
    # The system prompt is the parser's own, as in the corpus generation.
    assert len(seen) == 1
    with pytest.raises(FileExistsError):
        module.parse_prompts(config, sources, tmp_path / "run")
