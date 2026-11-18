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

A ``pilot`` or ``final`` run pins the parser (R-HAU-03, C-24): before loading,
the configuration, adapter, system prompt and decoding must match the parser
identity (``protocol/hauto/parser.json``) and the model snapshot its expected
digests; after loading, the adapter's conversation template must be the one of
the reference run. A ``final`` run also needs a verified freeze listing both
files. Every run writes its plan first and appends one prediction per line, so
an interrupted run resumes after its last complete prediction.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pstparser.config import ExperimentConfig
from pstparser.data.prepare import PreparedRecord
from pstparser.inference import generate as generation
from pstparser.models import loader
from pstparser.pst.target import serialise_target
from pstparser.pst.taxonomy import LEAF_PATHS, assemble
from pstparser.repro.manifest import (
    build_manifest,
    digest_directory,
    digest_file,
    digest_text,
    ensure_stage_outputs,
    write_manifest,
)
from pstparser.repro.seeding import seed_everything

STAGE: Final = "parse-prompts"
ID_MAP_FILE: Final = "id-map.json"
ID_MAP_FORMAT: Final = "id-map-v1"
PLAN_FILE: Final = "parse-plan.json"
PLAN_FORMAT: Final = "parse-plan-v1"
IDENTITY_FORMAT: Final = "parser-identity-v1"
PURPOSES: Final = ("exploratory", "pilot", "final")


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


def _document(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: not a JSON object")
    return document


def check_identity(
    config: ExperimentConfig, adapter_dir: Path | None, identity: Mapping[str, Any]
) -> None:
    """Refuse a pinned run whose configuration differs from the parser identity.

    Raises:
        ValueError: On a backend other than hf, an unpinned revision, a missing
            adapter, or any difference in model, adapter, system prompt or decoding.
    """
    model = config.model
    if model.backend != "hf":
        raise ValueError("a pinned parser run requires the hf backend")
    if not re.fullmatch(r"[0-9a-f]{40}", model.revision or ""):
        raise ValueError("a pinned parser run requires an immutable 40-character model revision")
    if identity.get("format") != IDENTITY_FORMAT or identity.get("backend") != "hf":
        raise ValueError(f"the parser identity is not a {IDENTITY_FORMAT} document for hf")
    if identity.get("model") != {"name": model.name, "revision": model.revision}:
        raise ValueError("the configured model or revision differs from the parser identity")
    if adapter_dir is None:
        raise ValueError("a pinned parser run requires the adapter")
    if digest_directory(adapter_dir) != identity.get("adapter", {}).get("digest"):
        raise ValueError("the adapter digest differs from the parser identity")
    if digest_file(config.system_prompt_path) != identity.get("system_prompt_sha256"):
        raise ValueError("the system prompt differs from the parser identity")
    if config.inference.model_dump() != identity.get("decoding"):
        raise ValueError("the decoding differs from the parser identity")
    if config.inference.do_sample or config.inference.repetition_penalty != 1.0:
        raise ValueError("the parser decodes greedily with repetition_penalty 1.0 (M03)")


def _require_freeze(freeze: Path, *listed: Path) -> str:
    """Verify the freeze and check that it lists every given file; return its digest."""
    from pstparser.empirical.freeze import load_freeze, verify_freeze

    mismatches = verify_freeze(freeze)
    if mismatches:
        shown = ", ".join(f"{m.kind} {m.path}" for m in mismatches[:5])
        raise ValueError(f"the freeze does not verify: {shown}")
    items = set(load_freeze(freeze)["items"].values())
    for path in listed:
        if digest_file(path) not in items:
            raise ValueError(f"the freeze does not list {Path(path).name}")
    digest = digest_file(freeze)
    assert digest is not None
    return digest


def _resume(run_dir: Path, plan: Mapping[str, Any], records: Sequence[PreparedRecord]) -> int:
    """Check an interrupted run against its plan; return how many predictions it kept.

    A line cut by the interruption is dropped from the file.

    Raises:
        FileExistsError: If the run already completed.
        ValueError: If the plan or the kept predictions differ.
    """
    plan_path = run_dir / PLAN_FILE
    if not plan_path.is_file() or _document(plan_path) != dict(plan):
        raise ValueError("the run directory holds another plan; resume with the same inputs")
    ensure_stage_outputs(run_dir, STAGE, [ID_MAP_FILE])
    path = run_dir / generation.PREDICTIONS_FILE
    data = path.read_bytes() if path.is_file() else b""
    complete = data[: data.rfind(b"\n") + 1]
    if complete != data:
        with path.open("r+b") as handle:
            handle.truncate(len(complete))
    kept = [json.loads(line) for line in complete.decode("utf-8").split("\n") if line]
    if len(kept) > len(records):
        raise ValueError("the kept predictions outnumber the planned records")
    for row, record in zip(kept, records, strict=False):
        if (row["index"], row["prompt"], row["target"]) != (
            record.index,
            record.prompt,
            record.target,
        ):
            raise ValueError("the kept predictions differ from the planned records")
    return len(kept)


def _append(stream: Iterable[Any], path: Path) -> None:
    """Append predictions one line at a time, each on disk before the next is generated."""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for prediction in stream:
            handle.write(json.dumps(prediction.as_dict(), ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def _verify_loaded(
    model: Any, tokenizer: Any, snapshot: loader.SnapshotIdentity, identity: Mapping[str, Any]
) -> None:
    """Check the loaded parser: the reference template and the pinned base weights.

    Raises:
        ValueError: If the template digest differs or the base model came from elsewhere.
    """
    template = getattr(tokenizer, "chat_template", None)
    if digest_text(str(template)) != identity["adapter"]["chat_template_digest"]:
        raise ValueError("the conversation template differs from the reference run")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    origin = getattr(base, "name_or_path", None)
    if not isinstance(origin, str) or Path(origin) != snapshot.path:
        raise ValueError("the base model was not loaded from the pinned snapshot")


def parse_prompts(
    config: ExperimentConfig,
    sources: Path,
    run_dir: Path,
    adapter_dir: Path | None = None,
    *,
    purpose: str = "exploratory",
    identity: Path | None = None,
    parser_model: Path | None = None,
    freeze: Path | None = None,
    resume: bool = False,
) -> ParseOutcome:
    """Generate one parser prediction per source prompt, with its manifest.

    Args:
        config: The experiment configuration of the parser.
        sources: JSONL of benchmark prompts.
        run_dir: Directory of the predictions, the index map, the plan and the manifest.
        adapter_dir: The trained adapter; without it the base model runs untuned.
        purpose: ``exploratory``, ``pilot`` or ``final``; the last two pin the parser.
        identity: The parser identity (``protocol/hauto/parser.json``), for pinned runs.
        parser_model: Expected digests of the model snapshot (``model-digests``), pinned runs.
        freeze: The verified freeze listing both files, for final runs.
        resume: Continue an interrupted run of the same plan in ``run_dir``.

    Raises:
        FileExistsError: If the stage outputs already exist in ``run_dir`` and
            ``resume`` is not set, or if the run already completed.
        ValueError: If the sources are malformed, or a pinned run differs from its identity.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {', '.join(PURPOSES)}")
    pinned = purpose != "exploratory"
    freeze_sha256 = None
    identity_document: dict[str, Any] | None = None
    expected_files: dict[str, Any] | None = None
    if pinned:
        if identity is None or parser_model is None:
            raise ValueError("a pinned parser run requires the parser identity and model digests")
        identity_document = _document(identity)
        check_identity(config, adapter_dir, identity_document)
        expected_files = _document(parser_model)
        if expected_files.get("name") != config.model.name:
            raise ValueError("the model digests belong to another repository")
        if purpose == "final":
            if freeze is None:
                raise ValueError("a final parser run requires a verified freeze")
            freeze_sha256 = _require_freeze(freeze, identity, parser_model)
    elif freeze is not None:
        freeze_sha256 = _require_freeze(freeze)

    records, tasks = prepared_records(_rows(sources))
    plan = {
        "format": PLAN_FORMAT,
        "purpose": purpose,
        "freeze_sha256": freeze_sha256,
        "sources_sha256": digest_file(sources),
        "config_digest": digest_text(
            json.dumps(config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        ),
        "system_prompt_sha256": digest_file(config.system_prompt_path),
        "adapter_digest": digest_directory(adapter_dir) if adapter_dir is not None else None,
        "identity_sha256": digest_file(identity) if identity is not None else None,
        "parser_model_sha256": digest_file(parser_model) if parser_model is not None else None,
        "records": len(records),
    }
    if resume:
        kept = _resume(run_dir, plan, records)
    else:
        ensure_stage_outputs(run_dir, STAGE, [generation.PREDICTIONS_FILE, ID_MAP_FILE, PLAN_FILE])
        with (run_dir / PLAN_FILE).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        kept = 0

    seeds = seed_everything(config.seed)
    gold = sum(record.target != gold_target([]) for record in records)
    system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")
    snapshot = None
    if pinned:
        assert expected_files is not None
        snapshot = loader.resolve_snapshot(
            config.model.name, config.model.revision or "", hash_weights=True
        )
        if snapshot.sha != config.model.revision:
            raise ValueError(f"the snapshot holds {snapshot.sha}, not {config.model.revision}")
        try:
            loader.verify_expected_files(snapshot, expected_files)
        except loader.BackendError as error:
            raise ValueError(str(error)) from error
    if adapter_dir is None:
        model, tokenizer = loader.load_untuned(config.model)
    else:
        model, tokenizer = loader.load_adapter(config.model, adapter_dir)
    parser: dict[str, Any] | None = None
    if pinned:
        assert snapshot is not None and identity_document is not None and identity is not None
        _verify_loaded(model, tokenizer, snapshot, identity_document)
        parser = {
            "run_id": identity_document.get("run_id"),
            "identity_sha256": digest_file(identity),
            "parser_model_sha256": digest_file(parser_model) if parser_model else None,
            "name": config.model.name,
            "revision": snapshot.sha,
            "revision_verified": True,
            "model_files_verified": True,
            "adapter_verified": True,
            "chat_template_verified": True,
        }
    stream: Iterable[Any] = generation.generate_predictions(
        model=model,
        tokenizer=tokenizer,
        records=records[kept:],
        system_prompt=system_prompt,
        config=config.inference,
        max_seq_length=config.model.max_seq_length,
    )
    predictions_path = run_dir / generation.PREDICTIONS_FILE
    _append(stream, predictions_path)
    id_map_path = run_dir / ID_MAP_FILE
    with id_map_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {"format": ID_MAP_FORMAT, "sources_sha256": digest_file(sources), "tasks": tasks},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    inputs: dict[str, str | Path] = {"sources": sources, "system_prompt": config.system_prompt_path}
    if identity is not None:
        inputs["parser_identity"] = identity
    if parser_model is not None:
        inputs["parser_model"] = parser_model
    runtime = (
        loader.runtime_provenance(model, tokenizer, config.model, snapshot)
        if snapshot is not None
        else loader.runtime_provenance(model, tokenizer, config.model)
    )
    manifest_path = write_manifest(
        build_manifest(
            config=config,
            seeds=seeds,
            stage=STAGE,
            inputs=inputs,
            extra={
                "purpose": purpose,
                "freeze_sha256": freeze_sha256,
                "parser": parser,
                "decoding": config.inference.model_dump(),
                "plan_sha256": digest_file(run_dir / PLAN_FILE),
                "resumed_records": kept,
                "adapter": str(adapter_dir) if adapter_dir is not None else None,
                "adapter_digest": (
                    digest_directory(adapter_dir) if adapter_dir is not None else None
                ),
                "generated": len(records),
                "gold_targets": gold,
                "id_map_sha256": digest_file(id_map_path),
                "predictions_sha256": digest_file(predictions_path),
                "model_runtime": runtime,
            },
        ),
        run_dir,
    )
    return ParseOutcome(run_dir, predictions_path, id_map_path, manifest_path, len(records))
