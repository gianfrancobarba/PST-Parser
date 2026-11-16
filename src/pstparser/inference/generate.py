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
from time import perf_counter
from typing import Any, Literal

import torch

from pstparser.config import ExperimentConfig, InferenceConfig
from pstparser.conversation import build_prompt
from pstparser.data import PreparedRecord, load_records, load_split, select
from pstparser.data.predictions import PredictionRecord, load_predictions, write_predictions
from pstparser.data.prepare import RECORDS_FILE
from pstparser.data.splits import split_input_paths
from pstparser.models.loader import BackendError, load_adapter, load_untuned, runtime_provenance
from pstparser.repro.manifest import (
    build_manifest,
    digest_directory,
    ensure_stage_outputs,
    write_manifest,
)
from pstparser.repro.seeding import seed_everything

PREDICTIONS_FILE = "predictions.jsonl"

#: Sides of the partition predictions can be generated for.
SplitSide = Literal["train", "val", "test"]


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
    adapter_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    limit: int | None = None,
    side: SplitSide = "test",
) -> GenerationOutcome:
    """Generate a prediction for every record of one side of the partition.

    The adapter is optional. Without one the base model answers the same test
    set under the same system message, which is what says how much the
    fine-tuning added: a score has no meaning on its own if nothing untrained
    was ever asked the same question.

    Args:
        config: The resolved experiment configuration.
        adapter_dir: Directory holding the trained adapter. When omitted the
            base model answers untuned.
        run_dir: Directory for the artefacts. Defaults to the parent of
            ``adapter_dir``, so that predictions sit beside the model that
            produced them. Required when there is no adapter.
        limit: Stop after this many records, for a quick check.
        side: Which side of the partition to generate for. The test set is the
            default because it is what results are quoted from; the validation
            set is there for the checks made while developing.

    Returns:
        The predictions and the paths they were written to.

    Raises:
        ValueError: If neither an adapter nor a run directory is given, leaving
            nowhere to write.
        FileNotFoundError: If the prepared corpus, the split or the adapter is
            missing.
    """
    seeds = seed_everything(config.seed)
    adapter_dir = Path(adapter_dir) if adapter_dir is not None else None
    if run_dir is None:
        if adapter_dir is None:
            raise ValueError("generating without an adapter requires a run directory")
        run_dir = adapter_dir.parent
    run_dir = Path(run_dir)
    ensure_stage_outputs(run_dir, "generate", [PREDICTIONS_FILE])

    records_path = Path(config.data.processed_dir) / RECORDS_FILE
    records = load_records(records_path)
    split = load_split(config.data.split.output_dir, records_path=records_path)
    system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")

    evaluation_records = select(records, getattr(split, side))
    if limit is not None:
        evaluation_records = evaluation_records[:limit]

    if adapter_dir is None:
        model, tokenizer = load_untuned(config.model)
    else:
        model, tokenizer = load_adapter(config.model, adapter_dir)
    stream = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        records=evaluation_records,
        system_prompt=system_prompt,
        config=config.inference,
        max_seq_length=config.model.max_seq_length,
    )

    predictions_path = write_predictions(stream, run_dir / PREDICTIONS_FILE)
    predictions = load_predictions(predictions_path)
    manifest_path = write_manifest(
        build_manifest(
            config=config,
            seeds=seeds,
            stage="generate",
            inputs={
                **{
                    f"corpus[{position}]": source.path
                    for position, source in enumerate(config.data.sources)
                },
                "records": records_path,
                "system_prompt": config.system_prompt_path,
                **split_input_paths(config.data.split.output_dir),
            },
            extra={
                # Null rather than absent: a run with no adapter is a claim
                # about the base model, and the file has to say so.
                "adapter": str(adapter_dir) if adapter_dir is not None else None,
                "adapter_digest": (
                    digest_directory(adapter_dir) if adapter_dir is not None else None
                ),
                "split": side,
                "generated": len(predictions),
                "model_runtime": runtime_provenance(model, tokenizer, config.model),
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
    max_seq_length: int | None = None,
) -> Iterable[PredictionRecord]:
    """Run the model over a sequence of records.

    Args:
        model: The adapted model, already in inference mode.
        tokenizer: Tokenizer carrying the conversation template.
        records: The records to submit.
        system_prompt: System message prepended to every conversation.
        config: Decoding parameters.
        max_seq_length: Optional experiment context ceiling, including the input.

    Yields:
        One prediction per record, in input order.
    """
    processors = build_schema_constraint(model, tokenizer) if config.structured_output else None
    for record in records:
        if processors is not None:
            reset_constraint(processors)
        result = generate_messages(
            model,
            tokenizer,
            build_prompt(system_prompt, record.prompt),
            config,
            logits_processors=processors,
            max_seq_length=max_seq_length,
        )
        yield PredictionRecord(
            index=record.index,
            prompt=record.prompt,
            target=record.target,
            prediction=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            finish_reason=result.finish_reason,
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
    return generate_messages(
        model,
        tokenizer,
        build_prompt(system_prompt, prompt),
        config,
        logits_processors=logits_processors,
    ).text


@dataclass(frozen=True)
class GeneratedText:
    """Raw completion and observed termination, without repairing its content."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    elapsed_seconds: float
    generation_config: dict[str, Any]


def decoding_parameters(config: InferenceConfig) -> dict[str, Any]:
    """Return every decoding value handed to the model, and nothing implicit.

    The dictionary is the whole content of the ``GenerationConfig`` built for a
    call. Greedy decoding carries no sampling value; sampling carries all three,
    with ``top_k`` 0 (filter disabled) when none is configured, because the
    library default is 50.
    """
    values: dict[str, Any] = {
        "do_sample": config.do_sample,
        "num_beams": 1,
        "repetition_penalty": config.repetition_penalty,
        "max_new_tokens": config.max_new_tokens,
        "use_cache": True,
    }
    if config.do_sample:
        values["temperature"] = config.temperature
        values["top_p"] = config.top_p
        values["top_k"] = config.top_k if config.top_k is not None else 0
    return values


def prefill_token_ids(tokenizer: Any, assistant_prefill: str) -> list[int]:
    """Token ids of a shared assistant prefill, without special tokens.

    Raises:
        ValueError: If the prefill is empty or tokenizes to nothing.
    """
    if not assistant_prefill:
        raise ValueError("assistant_prefill must be a non-empty string")
    ids = [
        int(value) for value in tokenizer(assistant_prefill, add_special_tokens=False)["input_ids"]
    ]
    if not ids:
        raise ValueError("assistant_prefill tokenizes to no token")
    return ids


def generate_messages(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    config: InferenceConfig,
    *,
    logits_processors: Any = None,
    max_seq_length: int | None = None,
    assistant_prefill: str | None = None,
) -> GeneratedText:
    """Generate from explicit messages after checking the entire token budget.

    No input truncation is requested. The context ceiling is the minimum of
    the explicit experiment limit and finite model/tokenizer limits. Unknown
    ceilings fail closed. EOS takes precedence over the length limit when
    the last permitted token is EOS.

    Decoding uses a ``GenerationConfig`` built from :func:`decoding_parameters`
    alone, with ``use_model_defaults=False``: without that switch transformers
    replaces every field left at the library default with the value published
    in the model's own ``generation_config`` (a repetition penalty, a sampling
    temperature). Only the special token ids still come from the model.

    ``assistant_prefill`` (S-PREFILL sensitivity only) is tokenized on its own,
    without special tokens, and appended after the assistant generation
    prompt; it counts in the token budget and is not part of the returned
    text, which holds only what the model generated.
    """
    from transformers import GenerationConfig

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=False,
    )
    if assistant_prefill is not None:
        prefill = torch.tensor(
            [prefill_token_ids(tokenizer, assistant_prefill)], dtype=inputs.dtype
        )
        inputs = torch.cat([inputs, prefill], dim=-1)
    inputs = inputs.to(model_device(model))

    input_tokens = int(inputs.shape[-1])
    validate_token_budget(model, tokenizer, input_tokens, config.max_new_tokens, max_seq_length)

    decoding = decoding_parameters(config)
    keywords: dict[str, Any] = {
        "generation_config": GenerationConfig(**decoding),
        "use_model_defaults": False,
    }
    if logits_processors is not None:
        keywords["logits_processor"] = logits_processors

    started = perf_counter()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, **keywords)

    generated = outputs[0][inputs.shape[-1] :]
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    count = len(generated)
    reason = (
        "eos"
        if count and int(generated[-1]) in eos_ids
        else "length"
        if count >= config.max_new_tokens
        else "stopped"
    )
    return GeneratedText(
        text=str(tokenizer.decode(generated, skip_special_tokens=True)),
        input_tokens=input_tokens,
        output_tokens=count,
        finish_reason=reason,
        elapsed_seconds=perf_counter() - started,
        generation_config=decoding,
    )


def validate_token_budget(
    model: Any,
    tokenizer: Any,
    input_tokens: int,
    max_new_tokens: int,
    max_seq_length: int | None = None,
) -> None:
    """Check an input/output budget before any completion in its paired study."""
    limits = [
        value
        for value in (
            max_seq_length,
            getattr(getattr(model, "config", None), "max_position_embeddings", None),
            getattr(tokenizer, "model_max_length", None),
        )
        if isinstance(value, int) and 0 < value < 1_000_000_000
    ]
    if not limits:
        raise ValueError("unknown context limit: provide max_seq_length")
    if input_tokens + max_new_tokens > min(limits):
        raise ValueError(
            f"token budget exceeded: input {input_tokens} + output "
            f"{max_new_tokens} > context {min(limits)}; no truncation performed"
        )


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


def reset_constraint(processors: Any) -> None:
    """Return the constraint to its initial state, before the next prediction.

    The processor walks the schema as the model writes, so it finishes a
    prediction in the state where the object has been closed. Handed the next
    record in that state, it either permits nothing but the end of the text or
    refuses the first token outright, and every prediction after the first comes
    back empty. Restarting the walk is not the same as recompiling the grammar:
    the compiled index is what costs, and it is kept.

    Args:
        processors: The constraint built by :func:`build_schema_constraint`.

    Raises:
        BackendError: If a processor cannot be restarted, which would leave
            every prediction after the first silently empty.
    """
    for processor in processors:
        restart = getattr(processor, "reset", None)
        if not callable(restart):
            raise BackendError(
                f"{type(processor).__name__} cannot be restarted between predictions; "
                "constrained decoding would answer the first record and no other"
            )
        restart()


def build_schema_constraint(model: Any, tokenizer: Any) -> Any:
    """Build the constraint that restricts decoding to the target schema.

    The grammar is compiled once and reused across the whole evaluation set,
    since compilation dominates the cost of constrained decoding. The walk over
    it is per prediction, and :func:`reset_constraint` is what restarts it.

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
