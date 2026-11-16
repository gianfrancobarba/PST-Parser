"""Frozen requests and append-only completions for resumable paired generation."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pstparser.empirical.contracts import CONDITIONS, PromptVariant
from pstparser.empirical.rendering import ENVELOPE_RENDERER_VERSION, envelope_text
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


def _check_envelope(variant: PromptVariant) -> None:
    """E must be the renderer's own envelope of the shared source text."""
    if variant.renderer_version != ENVELOPE_RENDERER_VERSION:
        raise ValueError(f"E requires the envelope renderer {ENVELOPE_RENDERER_VERSION}")
    if variant.annotation_hash or variant.annotation_version:
        raise ValueError("the envelope carries no annotation")
    parsed = json.loads(variant.text)
    inner = parsed.get("prompt") if isinstance(parsed, dict) else None
    if (
        not isinstance(inner, str)
        or envelope_text(inner) != variant.text
        or digest_text(inner) != variant.source_hash
    ):
        raise ValueError("the envelope is not the rendering of its source text")


def request_plan(
    variants: Sequence[PromptVariant],
    replicate_ids: Sequence[int],
    seed: int,
    conditions: Sequence[str] = ("T", "H-gold"),
) -> list[dict[str, Any]]:
    """Randomise task/replica blocks and the order of the conditions within each.

    Every task must hold exactly ``conditions``. All arms of a task share
    ``task_id`` and ``source_hash``; annotation and renderer metadata belong to
    each arm, except that T and H-gold, rendered together, must share them.
    T must equal the source and E must be its envelope. The list shuffled in
    each block follows the order of ``CONDITIONS``, not the declared order, so
    the plan depends only on the set of conditions; with T and H-gold it is
    identical to the plan made before conditions became a parameter.
    """
    if len(conditions) < 2 or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must list at least two distinct condition names")
    for name in conditions:
        if name not in CONDITIONS:
            raise ValueError(f"unregistered condition: {name!r}")
    by_task: dict[str, dict[str, PromptVariant]] = {}
    for variant in variants:
        sides = by_task.setdefault(variant.task_id, {})
        if variant.condition in sides:
            raise ValueError("duplicate task/condition")
        if digest_text(variant.text) != variant.prompt_hash:
            raise ValueError("prompt hash mismatch")
        sides[variant.condition] = variant
    if not by_task or any(set(sides) != set(conditions) for sides in by_task.values()):
        raise ValueError(f"each task requires exactly the conditions {list(conditions)}")
    if not replicate_ids or len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("replicate IDs must be non-empty and unique")
    if any(type(r) is not int or r < 0 for r in replicate_ids):
        raise ValueError("replicate IDs must be non-negative integers")
    for sides in by_task.values():
        arms = [sides[name] for name in conditions]
        if len({arm.source_hash for arm in arms}) != 1:
            raise ValueError("paired variants do not share source_hash")
        if "T" in sides and "H-gold" in sides:
            for field in ("annotation_hash", "annotation_version", "renderer_version"):
                if getattr(sides["T"], field) != getattr(sides["H-gold"], field):
                    raise ValueError(f"paired variants do not share {field}")
        if "T" in sides and sides["T"].source_hash != sides["T"].prompt_hash:
            raise ValueError("traditional prompt differs from canonical source")
        if "E" in sides:
            _check_envelope(sides["E"])
    rng = random.Random(seed)
    blocks = [(task, r) for task in sorted(by_task) for r in replicate_ids]
    rng.shuffle(blocks)
    requests = []
    for task, replicate in blocks:
        order = sorted(conditions, key=CONDITIONS.index)
        rng.shuffle(order)
        for condition in order:
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
        # Split on the line feed only: outputs may hold U+2028, U+2029 or U+0085,
        # which JSON leaves unescaped and str.splitlines would cut.
        lines = journal_path.read_bytes().decode("utf-8").split("\n")
        if lines[-1] == "":
            lines.pop()
        for line in lines:
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
