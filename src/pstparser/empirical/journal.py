"""Frozen requests and append-only completions for resumable paired generation."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pstparser.empirical.contracts import PromptVariant
from pstparser.repro.manifest import digest_text, source_snapshot, write_manifest


def _validate_completion(record: Mapping[str, Any]) -> None:
    """Reject malformed new and persisted completions before they can be skipped."""
    if not isinstance(record.get("raw_output"), str):
        raise ValueError("raw_output must be a string")
    for key in ("input_tokens", "output_tokens"):
        if type(record.get(key)) is not int or record[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if record.get("finish_reason") not in {"eos", "length", "stopped"}:
        raise ValueError("unknown finish_reason")


def request_plan(
    variants: Sequence[PromptVariant],
    replicate_ids: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    """Randomise task/replica blocks and order within each complete T/H pair."""
    by_task: dict[str, dict[str, PromptVariant]] = {}
    for variant in variants:
        sides = by_task.setdefault(variant.task_id, {})
        if variant.condition in sides:
            raise ValueError("duplicate task/condition")
        if digest_text(variant.text) != variant.prompt_hash:
            raise ValueError("prompt hash mismatch")
        sides[variant.condition] = variant
    if not by_task or any(set(sides) != {"T", "H-gold"} for sides in by_task.values()):
        raise ValueError("each task requires exactly T and H-gold")
    if not replicate_ids or len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("replicate IDs must be non-empty and unique")
    if any(type(r) is not int or r < 0 for r in replicate_ids):
        raise ValueError("replicate IDs must be non-negative integers")
    for sides in by_task.values():
        for field in ("source_hash", "annotation_hash", "annotation_version", "renderer_version"):
            if getattr(sides["T"], field) != getattr(sides["H-gold"], field):
                raise ValueError(f"paired variants do not share {field}")
        if sides["T"].source_hash != sides["T"].prompt_hash:
            raise ValueError("traditional prompt differs from canonical source")
    rng = random.Random(seed)
    blocks = [(task, r) for task in sorted(by_task) for r in replicate_ids]
    rng.shuffle(blocks)
    requests = []
    for task, replicate in blocks:
        conditions = ["T", "H-gold"]
        rng.shuffle(conditions)
        for condition in conditions:
            request = by_task[task][condition].as_dict()
            request.update(replicate_id=replicate, seed=seed + replicate)
            request["request_id"] = digest_text(
                json.dumps([task, condition, replicate], ensure_ascii=False)
            )
            requests.append(request)
    return requests


def run_journal(
    requests: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
    directory: Path,
    generate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze requests/config/source, flush each result, and resume only missing IDs.

    The callback must return raw_output, input_tokens, output_tokens and
    finish_reason. Exceptions stop the run without converting infrastructure
    failures to model failures. A torn JSONL line fails closed and requires
    explicit recovery; it is never silently dropped or regenerated.
    """
    directory.mkdir(parents=True, exist_ok=True)
    plan = {
        "stage": "empirical-generate",
        "format_version": 1,
        "requests": list(requests),
        "configuration": dict(configuration),
        "source": source_snapshot(),
    }
    fingerprint = digest_text(json.dumps(plan, sort_keys=True, ensure_ascii=False))
    plan["fingerprint"] = fingerprint
    manifest_path = directory / "empirical-generate_manifest.json"
    journal_path = directory / "generations.jsonl"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError("resume refused: requests, configuration or source changed")
    else:
        if journal_path.exists():
            raise ValueError("journal without manifest")
        write_manifest(plan, directory)
    expected = {r["request_id"]: dict(r) for r in requests}
    if len(expected) != len(requests) or not expected:
        raise ValueError("request plan must have unique, non-empty IDs")
    completed: dict[str, dict[str, Any]] = {}
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            identifier = record["request_id"]
            if identifier in completed or identifier not in expected:
                raise ValueError("duplicate or unknown journal request")
            if any(record.get(key) != value for key, value in expected[identifier].items()):
                raise ValueError("journal input differs from frozen request")
            if record.get("fingerprint") != fingerprint:
                raise ValueError("journal manifest mismatch")
            _validate_completion(record)
            completed[identifier] = record
    with journal_path.open("a", encoding="utf-8", newline="\n") as handle:
        for identifier, request in expected.items():
            if identifier in completed:
                continue
            output = dict(generate(request))
            required = {"raw_output", "input_tokens", "output_tokens", "finish_reason"}
            if not required <= output.keys() or set(output) & set(request):
                raise ValueError("invalid generation result contract")
            _validate_completion(output)
            record = {**request, **output, "fingerprint": fingerprint}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[identifier] = record
    return [completed[r["request_id"]] for r in requests]
