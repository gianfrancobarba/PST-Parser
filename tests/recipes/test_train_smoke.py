"""End-to-end training run on a tiny model, exercising the whole recipe on CPU."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pstparser.config import ExperimentConfig, load_experiment
from pstparser.data import prepare_corpus
from pstparser.training import TrainingOutcome, run_training

pytestmark = pytest.mark.slow


@pytest.fixture
def config(config_dir: Path, tmp_path: Path) -> ExperimentConfig:
    """Point every output of the fixture experiment at a temporary directory."""
    return load_experiment(
        config_dir / "valid.yaml",
        overrides=[
            f"data.processed_dir={(tmp_path / 'processed').as_posix()}",
            f"data.split.output_dir={(tmp_path / 'splits').as_posix()}",
            f"training.output_dir={(tmp_path / 'outputs').as_posix()}",
        ],
        root=config_dir,
    )


@pytest.fixture
def outcome(config: ExperimentConfig) -> TrainingOutcome:
    """Prepare the tiny corpus and train on it."""
    prepare_corpus(config.data)
    return run_training(config)


def test_run_directory_is_created(outcome: TrainingOutcome) -> None:
    assert outcome.run_dir.is_dir()
    assert outcome.run_dir.name.endswith("_fixture")


def test_adapter_is_written(outcome: TrainingOutcome) -> None:
    assert (outcome.adapter_dir / "adapter_config.json").is_file()
    assert (outcome.adapter_dir / "adapter_model.safetensors").is_file()
    assert (outcome.adapter_dir / "tokenizer_config.json").is_file()


def test_only_adapters_are_trainable(outcome: TrainingOutcome) -> None:
    assert 0 < outcome.trainable < outcome.total


def test_corpus_is_split_between_the_two_datasets(outcome: TrainingOutcome) -> None:
    assert outcome.train_size + outcome.eval_size == 7
    assert outcome.eval_size > 0


def test_training_loss_is_finite(outcome: TrainingOutcome) -> None:
    losses = [float(entry["loss"]) for entry in outcome.log_history if "loss" in entry]

    assert losses
    assert all(math.isfinite(value) for value in losses)


def test_evaluation_runs_during_training(outcome: TrainingOutcome) -> None:
    assert any("eval_loss" in entry for entry in outcome.log_history)


def test_metrics_are_persisted(outcome: TrainingOutcome) -> None:
    lines = outcome.metrics_path.read_text(encoding="utf-8").splitlines()

    assert lines
    assert all(json.loads(line) for line in lines if line.strip())


def test_manifest_records_provenance(outcome: TrainingOutcome) -> None:
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))

    assert manifest["stage"] == "train"
    assert manifest["experiment"] == "fixture"
    assert manifest["seeds"]["global_seed"] == 7
    assert manifest["seeds"]["applied"]["torch"] == 7
    assert manifest["config"]["resolved"]["lora"]["r"] == 2
    assert manifest["config"]["digest"]
    assert manifest["inputs"]["corpus"]
    assert manifest["inputs"]["records"]
    assert manifest["dataset"]["chat_template_digest"]
    assert manifest["adapter_digest"]
    assert manifest["parameters"]["trainable"] == outcome.trainable
    assert manifest["environment"]["packages"]["torch"]
