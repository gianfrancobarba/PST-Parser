"""The parser gate is bound to the predictions behind S_parse and to S_gold.

``score`` records the SHA-256 of the predictions it read, their indices and,
with ``--id-map``, the tasks. ``parser-gate`` checks that the score report,
the S_parse document and the index map come from the same predictions, and
that the tasks are exactly those of the frozen sample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pstparser.cli.main import app as parser_app
from pstparser.data.predictions import PredictionRecord, write_predictions
from pstparser.empirical.auto import STRICT, auto_annotate, parser_gate
from pstparser.empirical.cli import app as empirical_app

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tests/assets/configs/valid.yaml"
THRESHOLDS = {"max_strict_failure": 0.5, "min_macro_f1": 0.5}
TASKS = ["BigCodeBench/1", "BigCodeBench/2"]
TEXTS = ["Write a function.", "Return a list. Then stop."]


def leaves(**values: list[str]) -> str:
    from pstparser.pst.target import serialise_target
    from pstparser.pst.taxonomy import LEAF_PATHS, assemble

    tree = {path: [] for path in LEAF_PATHS}
    for key, value in values.items():
        tree[key.replace("__", ".")] = value
    return serialise_target(assemble(tree))


def write_inputs(directory: Path, prediction: str | None = None) -> dict[str, Path]:
    """Predictions, index map, sources and sample of a two-task run."""
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [
        leaves(main_instruction=[TEXTS[0]]),
        prediction or leaves(main_instruction=[TEXTS[1]]),
    ]
    records = [
        PredictionRecord(i, text, leaves(main_instruction=[text]), output)
        for i, (text, output) in enumerate(zip(TEXTS, outputs, strict=True))
    ]
    predictions = write_predictions(records, directory / "predictions.jsonl")
    id_map = directory / "id-map.json"
    id_map.write_text(json.dumps({"format": "id-map-v1", "tasks": TASKS}), encoding="utf-8")
    sources = directory / "sources.jsonl"
    sources.write_text(
        "".join(
            json.dumps({"task_id": t, "text": x}) + "\n" for t, x in zip(TASKS, TEXTS, strict=True)
        ),
        encoding="utf-8",
    )
    sample = directory / "sample.json"
    sample.write_text(json.dumps({"tasks": [{"task_id": t} for t in TASKS]}), encoding="utf-8")
    return {"predictions": predictions, "id_map": id_map, "sources": sources, "sample": sample}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(paths: dict[str, Path], run_dir: Path) -> Path:
    result = CliRunner().invoke(
        parser_app,
        [
            "score",
            "--config",
            str(CONFIG),
            "--predictions",
            str(paths["predictions"]),
            "--id-map",
            str(paths["id_map"]),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    return run_dir / "results.json"


def convert(paths: dict[str, Path], directory: Path) -> Path:
    result = CliRunner().invoke(
        empirical_app,
        [
            "auto-annotate",
            str(paths["predictions"]),
            str(paths["id_map"]),
            str(paths["sources"]),
            "--policy",
            STRICT,
            "--out",
            str(directory / "hauto.canonical.jsonl"),
            "--failures",
            str(directory / "parser_failures.jsonl"),
            "--s-parse",
            str(directory / "s_parse.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    return directory / "s_parse.json"


def gate(
    s_parse: Path, results: Path, id_map: Path, sample: Path, tmp_path: Path, name: str = "g"
) -> Any:
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(THRESHOLDS), encoding="utf-8")
    return CliRunner().invoke(
        empirical_app,
        [
            "parser-gate",
            str(s_parse),
            str(results),
            "--thresholds",
            str(thresholds),
            "--id-map",
            str(id_map),
            "--sample",
            str(sample),
            "--out",
            str(tmp_path / name / "gate.json"),
        ],
    )


def test_score_records_the_digest_of_the_predictions_and_their_tasks(tmp_path) -> None:
    paths = write_inputs(tmp_path / "in")
    results = json.loads(score(paths, tmp_path / "score").read_text(encoding="utf-8"))
    provenance = results["provenance"]
    assert provenance["predictions_sha256"] == sha(paths["predictions"])
    assert provenance["indices"] == [0, 1]
    assert provenance["id_map_sha256"] == sha(paths["id_map"])
    assert provenance["tasks"] == TASKS


def test_the_gate_passes_on_the_same_predictions_and_states_the_f1_definition(tmp_path) -> None:
    paths = write_inputs(tmp_path / "in")
    results = score(paths, tmp_path / "score")
    s_parse = convert(paths, tmp_path / "hauto")
    result = gate(s_parse, results, paths["id_map"], paths["sample"], tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "g" / "gate.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["predictions_sha256"] == sha(paths["predictions"])
    assert report["tasks"] == {"n": 2, "sample_sha256": sha(paths["sample"])}
    definition = report["macro_f1_definition"]
    assert "excluded" in definition and "strict failure rate" in definition


def test_the_gate_refuses_a_score_of_other_predictions(tmp_path) -> None:
    paths = write_inputs(tmp_path / "in")
    s_parse = convert(paths, tmp_path / "hauto")
    other = write_inputs(tmp_path / "other", prediction=leaves(main_instruction=["Return a list."]))
    results = score(other, tmp_path / "score")
    result = gate(s_parse, results, paths["id_map"], paths["sample"], tmp_path)
    assert isinstance(result.exception, ValueError)
    assert "other predictions" in str(result.exception)


def test_the_gate_refuses_a_score_without_provenance(tmp_path) -> None:
    paths = write_inputs(tmp_path / "in")
    s_parse = convert(paths, tmp_path / "hauto")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"total": 2, "field_f1": {}}), encoding="utf-8")
    result = gate(s_parse, results, paths["id_map"], paths["sample"], tmp_path)
    assert isinstance(result.exception, ValueError)
    assert "predictions" in str(result.exception)


def test_the_gate_refuses_tasks_other_than_the_sample(tmp_path) -> None:
    paths = write_inputs(tmp_path / "in")
    results = score(paths, tmp_path / "score")
    s_parse = convert(paths, tmp_path / "hauto")
    sample = tmp_path / "s-gold.json"
    sample.write_text(
        json.dumps({"tasks": [{"task_id": "BigCodeBench/1"}, {"task_id": "BigCodeBench/9"}]}),
        encoding="utf-8",
    )
    result = gate(s_parse, results, paths["id_map"], sample, tmp_path)
    assert isinstance(result.exception, ValueError)
    assert "sample" in str(result.exception)


def test_the_gate_refuses_an_index_map_other_than_the_conversion(tmp_path) -> None:
    paths = write_inputs(tmp_path / "in")
    results = score(paths, tmp_path / "score")
    s_parse = convert(paths, tmp_path / "hauto")
    other = tmp_path / "id-map-other.json"
    other.write_text(
        json.dumps({"format": "id-map-v1", "tasks": TASKS}, indent=2), encoding="utf-8"
    )
    result = gate(s_parse, results, other, paths["sample"], tmp_path)
    assert isinstance(result.exception, ValueError)
    assert "index map" in str(result.exception)


def test_the_library_gate_needs_the_bindings() -> None:
    rows = [
        {"index": i, "prompt": text, "target": "{}", "prediction": leaves(main_instruction=[text])}
        for i, text in enumerate(TEXTS)
    ]
    sources = [{"task_id": t, "text": x} for t, x in zip(TASKS, TEXTS, strict=True)]
    s_parse = auto_annotate(
        rows,
        {"tasks": TASKS},
        sources,
        policy=STRICT,
        inputs={"predictions_sha256": "p" * 64, "id_map_sha256": "m" * 64},
    ).s_parse
    score_report: dict[str, Any] = {
        "total": 2,
        "field_f1": {"prompt.main_instruction": {"f1": 1.0, "support": 4}},
        "provenance": {"predictions_sha256": "p" * 64, "indices": [0, 1]},
    }
    report = parser_gate(
        s_parse,
        score_report,
        THRESHOLDS,
        id_map={"tasks": TASKS},
        id_map_sha256="m" * 64,
        sample_tasks=TASKS,
    )
    assert report["status"] == "pass"
    with pytest.raises(ValueError, match="indices"):
        parser_gate(
            s_parse,
            {**score_report, "provenance": {"predictions_sha256": "p" * 64, "indices": [1, 0]}},
            THRESHOLDS,
            id_map={"tasks": TASKS},
            id_map_sha256="m" * 64,
            sample_tasks=TASKS,
        )


def test_a_final_gate_is_measured_on_s_gold(tmp_path, monkeypatch) -> None:
    from pstparser.empirical import cli

    paths = write_inputs(tmp_path / "in")
    results = score(paths, tmp_path / "score")
    s_parse = convert(paths, tmp_path / "hauto")
    root = tmp_path / "repo"
    s_gold = root / "protocol" / "samples" / cli.S_GOLD_FILE
    s_gold.parent.mkdir(parents=True)
    s_gold.write_text(json.dumps({"tasks": [{"task_id": "BigCodeBench/7"}]}), encoding="utf-8")
    freeze = root / "protocol" / "protocol-freeze-hauto.json"
    monkeypatch.setattr(cli, "_purpose", lambda purpose, path: {"purpose": purpose})
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(THRESHOLDS), encoding="utf-8")
    arguments = [
        "parser-gate",
        str(s_parse),
        str(results),
        "--thresholds",
        str(thresholds),
        "--id-map",
        str(paths["id_map"]),
        "--sample",
        str(paths["sample"]),
        "--out",
        str(tmp_path / "g" / "gate.json"),
        "--purpose",
        "final",
        "--freeze",
        str(freeze),
    ]
    result = CliRunner().invoke(empirical_app, arguments)
    assert isinstance(result.exception, ValueError)
    assert "bcb-prefix-600.json" in str(result.exception)
