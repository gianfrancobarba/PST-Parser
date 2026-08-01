"""Producing predictions from a trained adapter.

Generation is deliberately separated from scoring: it is the only stage that
needs an accelerator, and its output is persisted so that metrics can be
recomputed anywhere, without a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pstparser.config import ExperimentConfig, InferenceConfig
from pstparser.data import PreparedRecord, load_records, load_split, select
from pstparser.data.prepare import RECORDS_FILE
from pstparser.models.loader import BackendError, load_adapter
from pstparser.repro.manifest import build_manifest, digest_directory, write_manifest
from pstparser.repro.seeding import seed_everything

PREDICTIONS_FILE = "predictions.jsonl"


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction alongside the inputs it was produced from.

    Attributes:
        index: Position of the record in the corpus.
        prompt: The raw prompt submitted to the model.
        target: The expected target tree, serialised.
        prediction: The raw model output.
    """

    index: int
    prompt: str
    target: str
    prediction: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the record."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "target": self.target,
            "prediction": self.prediction,
        }


@dataclass(frozen=True)
class GenerationOutcome:
    """Artefacts of a completed generation run.

    Attributes:
        run_dir: Directory holding the artefacts.
        predictions_path: File the predictions were written to.
        manifest_path: File describing how the run was produced.
        predictions: The predictions, in evaluation order.
    """

    run_dir: Path
    predictions_path: Path
    manifest_path: Path
    predictions: list[PredictionRecord]


def run_generation(
    config: ExperimentConfig,
    adapter_dir: str | Path,
    run_dir: str | Path | None = None,
    limit: int | None = None,
) -> GenerationOutcome:
    """Generate a prediction for every record of the evaluation split.

    Args:
        config: The resolved experiment configuration.
        adapter_dir: Directory holding the trained adapter.
        run_dir: Directory for the artefacts. Defaults to the parent of
            ``adapter_dir``, so that predictions sit beside the model that
            produced them.
        limit: Stop after this many records, for a quick check.

    Returns:
        The predictions and the paths they were written to.

    Raises:
        FileNotFoundError: If the prepared corpus, the split or the adapter is
            missing.
    """
    seeds = seed_everything(config.seed)
    adapter_dir = Path(adapter_dir)
    run_dir = Path(run_dir) if run_dir is not None else adapter_dir.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    records_path = Path(config.data.processed_dir) / RECORDS_FILE
    records = load_records(records_path)
    split = load_split(config.data.split.output_dir)
    system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")

    evaluation_records = select(records, split.eval)
    if limit is not None:
        evaluation_records = evaluation_records[:limit]

    model, tokenizer = load_adapter(config.model, adapter_dir)
    predictions = list(
        generate_predictions(
            model=model,
            tokenizer=tokenizer,
            records=evaluation_records,
            system_prompt=system_prompt,
            config=config.inference,
        )
    )

    predictions_path = write_predictions(predictions, run_dir / PREDICTIONS_FILE)
    manifest_path = write_manifest(
        build_manifest(
            config=config,
            seeds=seeds,
            stage="generate",
            inputs={
                "corpus": config.data.source_path,
                "records": records_path,
                "system_prompt": config.system_prompt_path,
            },
            extra={
                "adapter": str(adapter_dir),
                "adapter_digest": digest_directory(adapter_dir),
                "generated": len(predictions),
            },
        ),
        run_dir,
    )

    return GenerationOutcome(
        run_dir=run_dir,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        predictions=predictions,
    )


def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: Sequence[PreparedRecord],
    system_prompt: str,
    config: InferenceConfig,
) -> Iterable[PredictionRecord]:
    """Run the model over a sequence of records.

    Args:
        model: The adapted model, already in inference mode.
        tokenizer: Tokenizer carrying the conversation template.
        records: The records to submit.
        system_prompt: System message prepended to every conversation.
        config: Decoding parameters.

    Yields:
        One prediction per record, in input order.
    """
    processors = build_schema_constraint(model, tokenizer) if config.structured_output else None
    for record in records:
        yield PredictionRecord(
            index=record.index,
            prompt=record.prompt,
            target=record.target,
            prediction=generate_one(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                prompt=record.prompt,
                config=config,
                logits_processors=processors,
            ),
        )


def generate_one(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    prompt: str,
    config: InferenceConfig,
    logits_processors: Any = None,
) -> str:
    """Produce the model's answer to a single prompt.

    The conversation carries the system message and the prompt, and ends with
    the opening of an assistant turn, so that the model continues from there.
    Only the newly generated tokens are decoded.

    Args:
        model: The adapted model, already in inference mode.
        tokenizer: Tokenizer carrying the conversation template.
        system_prompt: System message prepended to the conversation.
        prompt: The raw prompt to segment.
        config: Decoding parameters.
        logits_processors: Constraints applied while decoding, if any.

    Returns:
        The decoded answer, without special tokens.
    """
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model_device(model))

    keywords: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "use_cache": True,
        "do_sample": config.do_sample,
    }
    if config.do_sample:
        keywords["temperature"] = config.temperature
        keywords["top_p"] = config.top_p
    if logits_processors is not None:
        keywords["logits_processor"] = logits_processors

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, **keywords)

    generated = outputs[0][inputs.shape[-1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True))


def model_device(model: Any) -> torch.device:
    """Report the device a model's parameters live on.

    Args:
        model: Any torch module.

    Returns:
        The device of the first parameter, or the CPU for a module without
        parameters.
    """
    for parameter in model.parameters():
        return torch.device(parameter.device)
    return torch.device("cpu")


def write_predictions(
    predictions: Iterable[PredictionRecord],
    destination: str | Path,
) -> Path:
    """Write predictions as one JSON object per line.

    Args:
        predictions: The predictions to write, in the order they should appear.
        destination: File to write. Parent directories are created.

    Returns:
        The path that was written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.as_dict(), ensure_ascii=False))
            handle.write("\n")
    return destination


def load_predictions(source: str | Path) -> list[PredictionRecord]:
    """Read predictions previously written by :func:`write_predictions`.

    Args:
        source: File holding one JSON object per line.

    Returns:
        The predictions, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"predictions file not found: {source}")

    predictions = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        predictions.append(
            PredictionRecord(
                index=int(payload["index"]),
                prompt=str(payload["prompt"]),
                target=str(payload["target"]),
                prediction=str(payload["prediction"]),
            )
        )
    return predictions


def build_schema_constraint(model: Any, tokenizer: Any) -> Any:
    """Build the constraint that restricts decoding to the target schema.

    The grammar is compiled once and reused across the whole evaluation set,
    since compilation dominates the cost of constrained decoding.

    Args:
        model: The model whose vocabulary the constraint is built against.
        tokenizer: Its tokenizer.

    Returns:
        A processor list accepted by the generation call.

    Raises:
        BackendError: If schema enforcement is requested but unavailable.
    """
    try:
        from outlines import from_transformers
        from outlines.backends import get_json_schema_logits_processor
    except ImportError as exc:
        raise BackendError("inference.structured_output requires the 'structured' extra") from exc

    from transformers import LogitsProcessorList

    from pstparser.pst import target_json_schema

    processor = get_json_schema_logits_processor(
        backend_name=None,
        model=from_transformers(model, tokenizer),
        json_schema=json.dumps(target_json_schema()),
    )
    return LogitsProcessorList([processor])
