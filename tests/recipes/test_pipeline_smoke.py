"""The whole pipeline on a tiny model: prepare, train, generate, score.

The model is untrained noise, so the scores mean nothing. What is verified is
that every stage runs, hands its artefacts to the next one, and records what it
did.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pstparser.config import ExperimentConfig, load_experiment
from pstparser.data import prepare_corpus
from pstparser.evaluation import evaluate, write_report
from pstparser.inference import GenerationOutcome, load_predictions, run_generation
from pstparser.training import TrainingOutcome, run_training

pytestmark = pytest.mark.slow

ASSET_CONFIG = Path("tests/assets/configs/valid.yaml")
ASSET_ROOT = Path("tests/assets/configs")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory holding every artefact produced by the pipeline."""
    return tmp_path_factory.mktemp("pipeline")


@pytest.fixture(scope="module")
def config(workspace: Path) -> ExperimentConfig:
    """Point every output of the fixture experiment at the shared workspace."""
    return load_experiment(
        ASSET_CONFIG,
        overrides=[
            f"data.processed_dir={(workspace / 'processed').as_posix()}",
            f"data.split.output_dir={(workspace / 'splits').as_posix()}",
            f"training.output_dir={(workspace / 'outputs').as_posix()}",
        ],
        root=ASSET_ROOT,
    )


@pytest.fixture(scope="module")
def trained(config: ExperimentConfig) -> TrainingOutcome:
    """Prepare the tiny corpus and train on it once for the whole module."""
    prepare_corpus(config.data)
    return run_training(config)


@pytest.fixture(scope="module")
def generated(config: ExperimentConfig, trained: TrainingOutcome) -> GenerationOutcome:
    """Generate predictions for the evaluation split."""
    return run_generation(config, adapter_dir=trained.adapter_dir)


def test_training_writes_its_artefacts(trained: TrainingOutcome) -> None:
    assert trained.run_dir.is_dir()
    assert (trained.adapter_dir / "adapter_config.json").is_file()
    assert (trained.adapter_dir / "adapter_model.safetensors").is_file()
    assert trained.metrics_path.is_file()


def test_only_adapters_are_trainable(trained: TrainingOutcome) -> None:
    assert 0 < trained.trainable < trained.total


def test_corpus_is_split_between_the_two_datasets(trained: TrainingOutcome) -> None:
    assert trained.train_size + trained.eval_size == 7
    assert trained.eval_size > 0


def test_training_loss_is_finite(trained: TrainingOutcome) -> None:
    losses = [float(entry["loss"]) for entry in trained.log_history if "loss" in entry]

    assert losses
    assert all(math.isfinite(value) for value in losses)


def test_evaluation_runs_during_training(trained: TrainingOutcome) -> None:
    assert any("eval_loss" in entry for entry in trained.log_history)


def test_training_manifest_records_provenance(trained: TrainingOutcome) -> None:
    manifest = json.loads(trained.manifest_path.read_text(encoding="utf-8"))

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
    assert manifest["parameters"]["trainable"] == trained.trainable
    assert manifest["environment"]["packages"]["torch"]


def test_generation_covers_the_evaluation_split(
    generated: GenerationOutcome,
    trained: TrainingOutcome,
) -> None:
    assert len(generated.predictions) == trained.eval_size


def test_predictions_carry_their_inputs(generated: GenerationOutcome) -> None:
    for prediction in generated.predictions:
        assert prediction.prompt
        assert prediction.target
        assert isinstance(prediction.prediction, str)


def test_predictions_round_trip_through_disk(generated: GenerationOutcome) -> None:
    assert load_predictions(generated.predictions_path) == generated.predictions


def test_generation_manifest_points_at_the_adapter(
    generated: GenerationOutcome,
    trained: TrainingOutcome,
) -> None:
    manifest = json.loads(generated.manifest_path.read_text(encoding="utf-8"))

    assert manifest["stage"] == "generate"
    assert manifest["generated"] == len(generated.predictions)
    assert manifest["adapter"] == str(trained.adapter_dir)
    assert manifest["adapter_digest"]


def test_generation_honours_the_limit(
    config: ExperimentConfig,
    trained: TrainingOutcome,
    workspace: Path,
) -> None:
    outcome = run_generation(
        config,
        adapter_dir=trained.adapter_dir,
        run_dir=workspace / "limited",
        limit=1,
    )

    assert len(outcome.predictions) == 1


def test_scoring_runs_on_the_generated_predictions(
    config: ExperimentConfig,
    generated: GenerationOutcome,
    workspace: Path,
) -> None:
    records = load_predictions(generated.predictions_path)

    report = evaluate(
        predictions=[record.prediction for record in records],
        references=[record.target for record in records],
        prompts=[record.prompt for record in records],
        config=config.evaluation,
        indices=[record.index for record in records],
    )
    results_path, details_path = write_report(report, workspace / "scored")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    details = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report.total == len(records)
    assert 0.0 <= results["json_validity_rate"] <= 1.0
    assert len(details) == len(records)
    assert {detail["index"] for detail in details} == {record.index for record in records}
