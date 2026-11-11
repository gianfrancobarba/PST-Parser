"""Seeding and run provenance."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from pstparser.config import load_experiment
from pstparser.repro import (
    build_manifest,
    digest_directory,
    digest_file,
    digest_text,
    git_state,
    seed_everything,
    write_manifest,
)


def test_seeding_makes_generators_reproducible() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    seed_everything(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_different_seeds_diverge() -> None:
    seed_everything(1)
    first = random.random()

    seed_everything(2)
    second = random.random()

    assert first != second


def test_seed_record_lists_what_was_applied() -> None:
    record = seed_everything(99)

    assert record.global_seed == 99
    assert record.applied["random"] == 99
    assert record.applied["numpy"] == 99
    assert record.applied["torch"] == 99
    assert record.applied["transformers"] == 99
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_seed_record_is_serialisable() -> None:
    payload = json.loads(json.dumps(seed_everything(5).as_dict()))

    assert payload["global_seed"] == 5


def test_file_digest_tracks_content(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("alpha", encoding="utf-8")
    before = digest_file(path)

    path.write_text("beta", encoding="utf-8")

    assert before != digest_file(path)
    assert before == digest_text("alpha")


def test_digest_of_a_missing_file_is_null(tmp_path: Path) -> None:
    assert digest_file(tmp_path / "absent") is None


def test_directory_digest_tracks_names_and_content(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "b.txt").write_text("two", encoding="utf-8")
    before = digest_directory(tmp_path)

    (tmp_path / "b.txt").rename(tmp_path / "c.txt")

    assert before is not None
    assert before != digest_directory(tmp_path)


def test_directory_digest_is_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory, order in ((first, "ab"), (second, "ba")):
        directory.mkdir()
        for name in order:
            (directory / f"{name}.txt").write_text(name, encoding="utf-8")

    assert digest_directory(first) == digest_directory(second)


def test_digest_of_a_missing_directory_is_null(tmp_path: Path) -> None:
    assert digest_directory(tmp_path / "absent") is None


def test_git_state_is_readable_in_the_repository() -> None:
    state = git_state()

    assert state.branch is not None
    assert state.dirty is not None


def test_manifest_captures_configuration_and_inputs(config_dir: Path, tmp_path: Path) -> None:
    config = load_experiment(config_dir / "valid.yaml", root=config_dir)
    seeds = seed_everything(config.seed)

    manifest = build_manifest(
        config=config,
        seeds=seeds,
        stage="unit-test",
        inputs={"corpus": config.data.sources[0].path, "absent": tmp_path / "nope"},
        extra={"custom": 1},
    )

    assert manifest["stage"] == "unit-test"
    assert manifest["experiment"] == "fixture"
    assert manifest["config"]["resolved"]["model"]["backend"] == "hf"
    assert manifest["inputs"]["corpus"] is not None
    assert manifest["inputs"]["absent"] is None
    assert manifest["seeds"]["global_seed"] == config.seed
    assert manifest["environment"]["python"]
    assert manifest["custom"] == 1


def test_manifest_digest_reacts_to_configuration_changes(config_dir: Path) -> None:
    seeds = seed_everything(1)
    base = load_experiment(config_dir / "valid.yaml", root=config_dir)
    changed = load_experiment(config_dir / "valid.yaml", overrides=["lora.r=8"], root=config_dir)

    first = build_manifest(base, seeds, stage="unit-test")["config"]["digest"]
    second = build_manifest(changed, seeds, stage="unit-test")["config"]["digest"]

    assert first != second


def test_manifest_round_trips_through_disk(config_dir: Path, tmp_path: Path) -> None:
    config = load_experiment(config_dir / "valid.yaml", root=config_dir)
    manifest = build_manifest(config, seed_everything(1), stage="unit-test")

    path = write_manifest(manifest, tmp_path / "run")

    assert json.loads(path.read_text(encoding="utf-8"))["experiment"] == "fixture"


def test_stage_manifests_coexist_but_cannot_be_replaced(tmp_path: Path) -> None:
    train = write_manifest({"stage": "train", "marker": 1}, tmp_path)
    generation = write_manifest({"stage": "generate"}, tmp_path)
    assert train.name == "train_manifest.json"
    assert generation.name == "generate_manifest.json"
    with pytest.raises(FileExistsError):
        write_manifest({"stage": "train", "marker": 2}, tmp_path)
    assert json.loads(train.read_text())["marker"] == 1


def test_preflight_protects_outputs_before_a_manifest_exists(tmp_path: Path) -> None:
    from pstparser.repro.manifest import ensure_stage_outputs

    original = tmp_path / "results.json"
    original.write_text("historical", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_stage_outputs(tmp_path, "score", ("results.json",))
    assert original.read_text() == "historical"


def test_source_snapshot_detects_local_changes_without_collecting_secrets(tmp_path: Path) -> None:
    from pstparser.repro.manifest import source_snapshot

    package = tmp_path / "src" / "pstparser"
    package.mkdir(parents=True)
    module = package / "example.py"
    module.write_text("a = 1", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    (tmp_path / ".env").write_text("PRIVATE=secret", encoding="utf-8")
    before = source_snapshot(tmp_path)
    module.write_text("a = 2", encoding="utf-8")
    after = source_snapshot(tmp_path)
    assert before["digest"] != after["digest"]
    assert "secret" not in json.dumps(after)
    assert after["files"]["uv.lock"] is not None
    assert after["files"]["pyproject.toml"] is None


def test_score_and_align_preserve_historical_directory(config_dir: Path, tmp_path: Path) -> None:
    from pstparser.cli.main import align, score
    from pstparser.pst.taxonomy import LEAF_PATHS, assemble

    tree = json.dumps(assemble({leaf: [] for leaf in LEAF_PATHS}))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"index": 1, "prompt": "hello", "target": tree, "prediction": tree}) + "\n",
        encoding="utf-8",
    )
    history = tmp_path / "results.json"
    history.write_text("historical", encoding="utf-8")
    runs = tmp_path / "runs"
    overrides = [f"training.output_dir={runs.as_posix()}"]
    score(config_dir / "valid.yaml", predictions, set_=overrides)
    align(config_dir / "valid.yaml", predictions, set_=overrides)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "predictions.jsonl",
        "results.json",
        "runs",
    ]
    manifests = list(runs.glob("*/*_manifest.json"))
    assert {p.name for p in manifests} == {"score_manifest.json", "align_manifest.json"}
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["inputs"]["predictions"] == digest_file(predictions)
        assert payload["metrics_version"] == "pstparser-metrics-v2"
    assert history.read_text() == "historical"
