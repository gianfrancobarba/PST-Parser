"""Running the frozen parser on benchmark prompts (M19, H-auto).

The parser's ``generate`` command reads only the partitions of its corpus. This
stage submits arbitrary source prompts instead, through the same conversation,
decoding and prediction format, so that ``score`` and ``align`` read its output
unchanged. Each record's index is its line in the source file; ``id-map.json``
maps the index back to the task.

When a source row carries reviewed spans, the record's target is the gold tree
(the spans' phrases per leaf), which is what ``score`` compares against; it is
never read by the conversion of predictions into spans. Without spans the
target is the empty tree.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pstparser.config import ExperimentConfig
from pstparser.data.predictions import write_predictions
from pstparser.data.prepare import PreparedRecord
from pstparser.inference import generate as generation
from pstparser.models import loader
from pstparser.pst.target import serialise_target
from pstparser.pst.taxonomy import LEAF_PATHS, assemble
from pstparser.repro.manifest import (
    build_manifest,
    digest_directory,
    digest_file,
    ensure_stage_outputs,
    write_manifest,
)
from pstparser.repro.seeding import seed_everything

STAGE: Final = "parse-prompts"
ID_MAP_FILE: Final = "id-map.json"
ID_MAP_FORMAT: Final = "id-map-v1"


@dataclass(frozen=True)
class ParseOutcome:
    """Where the predictions, the index map and the manifest were written."""

    run_dir: Path
    predictions_path: Path
    id_map_path: Path
    manifest_path: Path
    records: int


def gold_target(annotations: Sequence[Mapping[str, Any]]) -> str:
    """The corpus target of reviewed spans: each leaf lists its phrases in source order."""
    values: dict[str, list[str]] = {path: [] for path in LEAF_PATHS}
    for span in sorted(annotations, key=lambda item: (item["start"], item["end"])):
        if span["path"] not in values:
            raise ValueError(f"unknown taxonomy leaf: {span['path']}")
        values[span["path"]].append(span["phrase"])
    return serialise_target(assemble(values))


def prepared_records(rows: Sequence[Mapping[str, Any]]) -> tuple[list[PreparedRecord], list[str]]:
    """Records in source order and the task of each index.

    Raises:
        ValueError: If a task repeats, a text is empty, or spans do not match the text.
    """
    records: list[PreparedRecord] = []
    tasks: list[str] = []
    for index, row in enumerate(rows):
        task, text = row["task_id"], row["text"]
        if not isinstance(task, str) or not task.strip() or task in tasks:
            raise ValueError(f"row {index}: task_id must be non-empty and unique")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{task}: empty text")
        annotations = row.get("annotations") or []
        for span in annotations:
            if text[span["start"] : span["end"]] != span["phrase"]:
                raise ValueError(f"{task}: a span differs from the text")
        records.append(PreparedRecord(index=index, prompt=text, target=gold_target(annotations)))
        tasks.append(task)
    if not records:
        raise ValueError("no source prompt")
    return records, tasks


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line]


def parse_prompts(
    config: ExperimentConfig,
    sources: Path,
    run_dir: Path,
    adapter_dir: Path | None = None,
) -> ParseOutcome:
    """Generate one parser prediction per source prompt, with its manifest.

    Raises:
        FileExistsError: If the stage outputs already exist in ``run_dir``.
        ValueError: If the sources are malformed.
    """
    seeds = seed_everything(config.seed)
    ensure_stage_outputs(run_dir, STAGE, [generation.PREDICTIONS_FILE, ID_MAP_FILE])
    records, tasks = prepared_records(_rows(sources))
    gold = sum(record.target != gold_target([]) for record in records)
    system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")
    if adapter_dir is None:
        model, tokenizer = loader.load_untuned(config.model)
    else:
        model, tokenizer = loader.load_adapter(config.model, adapter_dir)
    stream: Iterable[Any] = generation.generate_predictions(
        model=model,
        tokenizer=tokenizer,
        records=records,
        system_prompt=system_prompt,
        config=config.inference,
        max_seq_length=config.model.max_seq_length,
    )
    predictions_path = write_predictions(stream, run_dir / generation.PREDICTIONS_FILE)
    id_map_path = run_dir / ID_MAP_FILE
    with id_map_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {"format": ID_MAP_FORMAT, "sources_sha256": digest_file(sources), "tasks": tasks},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    manifest_path = write_manifest(
        build_manifest(
            config=config,
            seeds=seeds,
            stage=STAGE,
            inputs={"sources": sources, "system_prompt": config.system_prompt_path},
            extra={
                "adapter": str(adapter_dir) if adapter_dir is not None else None,
                "adapter_digest": (
                    digest_directory(adapter_dir) if adapter_dir is not None else None
                ),
                "generated": len(records),
                "gold_targets": gold,
                "id_map_sha256": digest_file(id_map_path),
                "predictions_sha256": digest_file(predictions_path),
                "model_runtime": loader.runtime_provenance(model, tokenizer, config.model),
            },
        ),
        run_dir,
    )
    return ParseOutcome(run_dir, predictions_path, id_map_path, manifest_path, len(records))
