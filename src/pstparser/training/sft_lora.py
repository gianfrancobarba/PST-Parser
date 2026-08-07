"""Supervised fine-tuning of low-rank adapters on the segmentation corpus.

The whole procedure is written out in this module rather than split across
layers of abstraction, so that the method can be read end to end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pstparser.config import ExperimentConfig
from pstparser.conversation import build_completion, build_prompt
from pstparser.data import PreparedRecord, load_records, load_split, select
from pstparser.data.prepare import RECORDS_FILE
from pstparser.models.loader import (
    attach_adapter,
    load_base,
    resolve_precision,
    save_adapter,
    trainable_parameters,
)
from pstparser.repro.manifest import build_manifest, digest_directory, write_manifest
from pstparser.repro.seeding import seed_everything

ADAPTER_DIR = "adapter"
METRICS_FILE = "training_metrics.jsonl"


@dataclass(frozen=True)
class TrainingOutcome:
    """Artefacts and summary figures of a completed training run.

    Attributes:
        run_dir: Directory holding every artefact of the run.
        adapter_dir: Directory holding the trained adapter and tokenizer.
        manifest_path: File describing how the run was produced.
        metrics_path: File holding the metrics logged during training.
        train_size: Number of training examples.
        eval_size: Number of evaluation examples.
        trainable: Number of trainable parameters.
        total: Total number of parameters.
        log_history: Metrics emitted by the trainer, in order.
    """

    run_dir: Path
    adapter_dir: Path
    manifest_path: Path
    metrics_path: Path
    train_size: int
    eval_size: int
    trainable: int
    total: int
    log_history: list[dict[str, Any]]


def run_training(config: ExperimentConfig, run_dir: str | Path | None = None) -> TrainingOutcome:
    """Fine-tune adapters on the prepared corpus and persist the result.

    Args:
        config: The resolved experiment configuration.
        run_dir: Directory for the run artefacts. A timestamped directory under
            ``config.training.output_dir`` is created when omitted.

    Returns:
        A summary of the run and the paths it wrote.

    Raises:
        FileNotFoundError: If the prepared corpus or the split is missing.
    """
    seeds = seed_everything(config.seed)
    run_dir = Path(run_dir) if run_dir is not None else _new_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    records_path = Path(config.data.processed_dir) / RECORDS_FILE
    records = load_records(records_path)
    split = load_split(config.data.split.output_dir)
    system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")

    train_dataset = build_dataset(select(records, split.train), system_prompt)
    eval_dataset = build_dataset(select(records, split.val), system_prompt)

    model, tokenizer = load_base(config.model)
    model = attach_adapter(model, config.model, config.lora, config.seed)
    trainable, total = trainable_parameters(model)

    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        run_dir=run_dir,
    )
    trainer.train()

    adapter_dir = save_adapter(model, tokenizer, run_dir / ADAPTER_DIR)
    log_history = [dict(entry) for entry in trainer.state.log_history]
    metrics_path = _write_metrics(log_history, run_dir / METRICS_FILE)

    manifest_path = write_manifest(
        build_manifest(
            config=config,
            seeds=seeds,
            stage="train",
            inputs={
                **{
                    f"corpus[{position}]": source.path
                    for position, source in enumerate(config.data.sources)
                },
                "records": records_path,
                "system_prompt": config.system_prompt_path,
            },
            extra={
                "dataset": {
                    "train_size": len(train_dataset),
                    "eval_size": len(eval_dataset),
                    "chat_template_digest": _chat_template_digest(tokenizer),
                },
                "parameters": {"trainable": trainable, "total": total},
                "adapter_digest": digest_directory(adapter_dir),
            },
        ),
        run_dir,
    )

    return TrainingOutcome(
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
        trainable=trainable,
        total=total,
        log_history=log_history,
    )


def build_dataset(records: list[PreparedRecord], system_prompt: str) -> Any:
    """Turn records into a dataset of prompts paired with their completion.

    The conversation is split in two columns rather than handed over as one
    sequence of turns, because that split is what lets the trainer tell the
    question from the answer and compute the loss on the answer alone. The
    turns themselves come from :mod:`pstparser.conversation`, which is also what
    generation asks: the two must not be able to drift apart.

    Args:
        records: The records to convert.
        system_prompt: System message prepended to every conversation.

    Returns:
        A dataset with a ``prompt`` and a ``completion`` column.
    """
    from datasets import Dataset

    conversations = [
        {
            "prompt": build_prompt(system_prompt, record.prompt),
            "completion": build_completion(record.target),
        }
        for record in records
    ]
    return Dataset.from_list(conversations)


def build_trainer(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    run_dir: Path,
) -> Any:
    """Assemble the trainer from the experiment configuration.

    Args:
        config: The resolved experiment configuration.
        model: The adapted model.
        tokenizer: Tokenizer carrying the conversation template.
        train_dataset: Conversations used for optimisation.
        eval_dataset: Conversations used for periodic evaluation.
        run_dir: Directory receiving checkpoints and trainer state.

    Returns:
        A trainer ready to run.
    """
    from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

    training = config.training
    fp16, bf16 = resolve_precision(training.precision)

    arguments: dict[str, Any] = {
        "output_dir": str(run_dir),
        "max_length": config.model.max_seq_length,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "optim": training.optim,
        "max_steps": training.max_steps,
        "warmup_steps": training.warmup_steps,
        # One seed governs the run. It also reaches the adapter initialisation
        # and, through seed_everything, every library the pipeline touches.
        "seed": config.seed,
        "eval_strategy": training.eval_strategy,
        "eval_steps": training.eval_steps,
        "logging_steps": training.logging_steps,
        "completion_only_loss": training.completion_only_loss,
        "prediction_loss_only": training.prediction_loss_only,
        "fp16": fp16,
        "bf16": bf16,
        "report_to": [],
    }

    if training.load_best_model_at_end:
        arguments.update(
            {
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "save_strategy": training.eval_strategy,
                "save_steps": training.eval_steps,
            }
        )

    # No formatting function: the trainer renders the two columns itself, and a
    # formatter would collapse them into one string, leaving it nothing to mask
    # on. It refuses the combination outright rather than training on the whole
    # sequence without saying so.
    return SFTTrainer(
        model=model,
        args=SFTConfig(**arguments),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def _new_run_dir(config: ExperimentConfig) -> Path:
    """Build a timestamped directory name for a run."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return Path(config.training.output_dir) / f"{stamp}_{config.name}"


def _write_metrics(log_history: list[dict[str, Any]], destination: Path) -> Path:
    """Write the trainer log as one JSON object per line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in log_history:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")
    return destination


def _chat_template_digest(tokenizer: Any) -> str | None:
    """Fingerprint the conversation template, which shapes every training example."""
    from pstparser.repro.manifest import digest_text

    template = getattr(tokenizer, "chat_template", None)
    return digest_text(template) if isinstance(template, str) else None
