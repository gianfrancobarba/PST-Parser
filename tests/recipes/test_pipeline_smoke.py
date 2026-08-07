"""The whole pipeline on a tiny model: prepare, train, generate, score, align.

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
from pstparser.data import (
    PreparedRecord,
    load_alignments,
    load_predictions,
    load_split,
    prepare_corpus,
)
from pstparser.evaluation import evaluate, run_alignment, write_report
from pstparser.inference import GenerationOutcome, run_generation
from pstparser.models import load_base
from pstparser.training import TrainingOutcome, build_dataset, build_trainer, run_training

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


def test_training_sees_the_training_set_and_watches_the_validation_set(
    trained: TrainingOutcome,
    config: ExperimentConfig,
) -> None:
    split = load_split(config.data.split.output_dir)

    assert trained.train_size == len(split.train)
    assert trained.eval_size == len(split.val)
    assert trained.eval_size > 0
    # The test set is not among the records the training run ever reads.
    assert len(split) == 7


def test_training_loss_is_finite(trained: TrainingOutcome) -> None:
    losses = [float(entry["loss"]) for entry in trained.log_history if "loss" in entry]

    assert losses
    assert all(math.isfinite(value) for value in losses)


def test_evaluation_runs_during_training(trained: TrainingOutcome) -> None:
    assert any("eval_loss" in entry for entry in trained.log_history)


def test_the_loss_is_computed_on_the_answer_alone(config: ExperimentConfig, tmp_path: Path) -> None:
    """The mask must reach the labels, not merely be asked for.

    Nothing else in the suite looks at what the loss is averaged over, and the
    flag that asks for the masking is the kind that a trainer can accept and not
    honour. Building the trainer is enough: it prepares the dataset in its
    constructor, so no optimisation step is needed to see the result.
    """
    records = [
        PreparedRecord(index=0, prompt="Fix the bug.", target='{"prompt": {}}'),
        PreparedRecord(index=1, prompt="Summarise this.", target='{"prompt": {"a": []}}'),
    ]
    dataset = build_dataset(records, "Segment the prompt.")
    model, tokenizer = load_base(config.model)

    trainer = build_trainer(config, model, tokenizer, dataset, dataset, tmp_path)
    example = trainer.train_dataset[0]
    labels = trainer.data_collator([example])["labels"][0]

    # The prompt is masked out, the answer is not, and neither side is empty:
    # a mask covering everything would also satisfy a laxer assertion.
    assert "completion_mask" in example
    prompt_length = example["completion_mask"].index(1)
    assert prompt_length > 0
    assert (labels[:prompt_length] == -100).all()
    assert (labels[prompt_length:] != -100).any()


def test_training_manifest_records_provenance(trained: TrainingOutcome) -> None:
    manifest = json.loads(trained.manifest_path.read_text(encoding="utf-8"))

    assert manifest["stage"] == "train"
    assert manifest["experiment"] == "fixture"
    assert manifest["seeds"]["global_seed"] == 7
    assert manifest["seeds"]["applied"]["torch"] == 7
    assert manifest["config"]["resolved"]["lora"]["r"] == 2
    assert manifest["config"]["digest"]
    assert manifest["inputs"]["corpus[0]"]
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


def test_the_untuned_model_answers_the_same_questions(
    config: ExperimentConfig,
    workspace: Path,
) -> None:
    """Without an adapter the base model answers, and the run says so.

    It is the comparison that says how much the fine-tuning added, and until
    now there was no way to run it: the adapter was a required argument.
    """
    outcome = run_generation(config, run_dir=workspace / "untuned", limit=1)
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))

    assert len(outcome.predictions) == 1
    assert manifest["adapter"] is None
    assert manifest["adapter_digest"] is None


def test_generating_without_an_adapter_needs_somewhere_to_write(
    config: ExperimentConfig,
) -> None:
    with pytest.raises(ValueError, match="requires a run directory"):
        run_generation(config, limit=1)


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


def test_alignment_runs_on_the_generated_predictions(
    config: ExperimentConfig,
    generated: GenerationOutcome,
    workspace: Path,
) -> None:
    records = load_predictions(generated.predictions_path)

    outcome = run_alignment(
        records,
        workspace / "aligned",
        levels=config.evaluation.alignment_levels,
    )

    assert len(outcome.records) == len(records)
    assert load_alignments(outcome.alignments_path) == outcome.records
    assert outcome.target_scores.total == len(records)
    # The reference is built from the corpus, so every phrase of it must be
    # found in its own prompt: a miss would be an annotation defect.
    assert outcome.target_scores.alignment_rate == 1.0
