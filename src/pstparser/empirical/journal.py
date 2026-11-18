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


#: Configuration keys that legitimately differ between the journals of one
#: session: the conditions (H-auto only on S_parse) and the variant file.
MERGE_FREE_KEYS = frozenset({"conditions", "variant_file_sha256"})


def _journal_rows(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(
        (directory / "empirical-generate_manifest.json").read_text(encoding="utf-8")
    )
    text = (directory / "generations.jsonl").read_bytes().decode("utf-8")
    rows = [json.loads(line) for line in text.split("\n") if line]
    expected = {request["request_id"] for request in manifest["requests"]}
    found = [row["request_id"] for row in rows]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError(f"{directory}: incomplete journal, or rows outside its plan")
    if any(row.get("fingerprint") != manifest["fingerprint"] for row in rows):
        raise ValueError(f"{directory}: a row belongs to another plan")
    for row in rows:
        _validate_completion(row)
    return manifest, rows


def merge_journals(directories: Sequence[Path], sample_tasks: Sequence[str]) -> dict[str, Any]:
    """Verified union of the complete journals of one generation session (C-26).

    The core conditions are those of every journal; their rows must cover each
    task of ``sample_tasks`` exactly once, a task belonging to one journal only.
    The other conditions (H-auto) go to the extra rows. The configurations may
    differ only in :data:`MERGE_FREE_KEYS`, and every row must come from the
    declared GPU with one frozen environment.

    Raises:
        ValueError: If a journal is incomplete, a task is in more than one
            journal, the core does not cover exactly the sample, the
            configurations differ, a row lacks its session identity or comes
            from another GPU or environment, or a journal uses a prefill.
    """
    if len(directories) < 2:
        raise ValueError("a union needs at least two journals")
    loaded = [_journal_rows(Path(directory)) for directory in directories]
    configurations = [manifest["configuration"] for manifest, _ in loaded]
    if any("assistant_prefill" in configuration for configuration in configurations):
        raise ValueError("journals with an assistant prefill are not merged")
    fixed = [
        {key: value for key, value in configuration.items() if key not in MERGE_FREE_KEYS}
        for configuration in configurations
    ]
    if any(other != fixed[0] for other in fixed[1:]):
        raise ValueError("the journals were generated with different configurations")
    gpu = configurations[0].get("gpu")
    environments = set()
    sessions = set()
    for _, rows in loaded:
        for row in rows:
            if not row.get("gpu_name") or not row.get("environment_digest"):
                raise ValueError("a row has no session identity (gpu_name, environment_digest)")
            if row["gpu_name"] != gpu:
                raise ValueError(f"a row comes from GPU {row['gpu_name']!r}, not {gpu!r}")
            environments.add(row["environment_digest"])
            sessions.add(row.get("session_id"))
    if len(environments) != 1:
        raise ValueError("the journals were generated in different environments")
    owner: dict[str, int] = {}
    for index, (_, rows) in enumerate(loaded):
        for task in {row["task_id"] for row in rows}:
            if owner.setdefault(task, index) != index:
                raise ValueError(f"{task}: the task is in more than one journal")
    condition_sets = [set(configuration["conditions"]) for configuration in configurations]
    core_conditions = set.intersection(*condition_sets)
    extra_conditions = set.union(*condition_sets) - core_conditions
    if set(owner) != set(sample_tasks) or len(set(sample_tasks)) != len(sample_tasks):
        raise ValueError("the journals do not cover exactly the tasks of the sample")
    every = [row for _, rows in loaded for row in rows]
    core = [row for row in every if row["condition"] in core_conditions]
    extra = [row for row in every if row["condition"] in extra_conditions]
    cells = {(row["task_id"], row["condition"], row["replicate_id"]) for row in core}
    replicates = {row["replicate_id"] for row in core}
    expected_cells = len(sample_tasks) * len(core_conditions) * len(replicates)
    if len(cells) != len(core) or len(core) != expected_cells:
        raise ValueError("the core rows do not hold each task, condition and replicate once")
    order = sorted(set().union(*condition_sets), key=CONDITIONS.index)
    return {
        "core_conditions": [name for name in order if name in core_conditions],
        "extra_conditions": [name for name in order if name in extra_conditions],
        "core": core,
        "extra": extra,
        "sessions": sorted(session for session in sessions if session is not None),
        "environment_digest": next(iter(environments)),
        "gpu": gpu,
    }
