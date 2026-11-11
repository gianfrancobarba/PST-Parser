"""Read-only corpus diagnostics; candidates are never semantic corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pstparser.config.loader import load_experiment
from pstparser.data.prepare import PreparedRecord, read_source
from pstparser.data.quality import coverage
from pstparser.data.splits import load_split
from pstparser.pst.alignment import align_tree
from pstparser.pst.target import serialise_target
from pstparser.pst.taxonomy import LEAF_PATHS, assemble


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def audit_records(
    records: Sequence[PreparedRecord],
    *,
    partitions: Mapping[int, str] | None = None,
    similarity_threshold: float = 0.9,
) -> dict[str, Any]:
    """Find uncovered text, tolerant matches and possible duplicate families.

    Similarity is Jaccard over sets of lowercase Unicode word tokens. Its
    configurable threshold selects review candidates, not proven duplication or
    leakage. All unordered pairs are examined, excluding normalised exact pairs
    already reported separately. No annotations or partitions are changed.
    """
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1]")
    ids = [record.index for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Record identifiers must be unique")
    assigned = dict(partitions or {})
    if set(assigned) - set(ids):
        raise ValueError("Partition contains an unknown record identifier")
    incomplete: list[dict[str, Any]] = []
    tolerant: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    exact_groups: dict[str, list[int]] = defaultdict(list)
    support: dict[str, Counter[str]] = defaultdict(Counter)
    tokens: list[set[str]] = []
    normalised: list[str] = []
    level_counts: Counter[str] = Counter()
    for record in records:
        parsed = json.loads(record.target)
        if not isinstance(parsed, dict):
            raise ValueError(f"Target {record.index} is not an object")
        alignment = align_tree(record.prompt, parsed)
        normal = _normalise(record.prompt)
        normalised.append(normal)
        tokens.append(set(re.findall(r"\w+", normal)))
        exact_groups[normal].append(record.index)
        claimed = bytearray(len(record.prompt))
        present = set()
        for leaf in alignment.leaves:
            if leaf.phrase.strip():
                present.add(re.sub(r"\[\d+\]$", "", leaf.path.removeprefix("prompt.")))
            if leaf.start is not None and leaf.end is not None:
                claimed[leaf.start : leaf.end] = b"\x01" * (leaf.end - leaf.start)
                level_counts[str(leaf.level)] += 1
                if leaf.level != "exact":
                    tolerant.append(
                        {
                            "index": record.index,
                            "partition": assigned.get(record.index),
                            **leaf.as_dict(),
                            "source_text": record.prompt[leaf.start : leaf.end],
                        }
                    )
            elif not leaf.blank:
                unresolved.append({"index": record.index, **leaf.as_dict()})
        for path in present:
            support["all"][path] += 1
            support[assigned.get(record.index, "unassigned")][path] += 1
        gaps = []
        cursor = 0
        while cursor < len(record.prompt):
            if claimed[cursor]:
                cursor += 1
                continue
            start = cursor
            while cursor < len(record.prompt) and not claimed[cursor]:
                cursor += 1
            missing = record.prompt[start:cursor]
            if missing.strip():
                gaps.append({"start": start, "end": cursor, "text": missing})
        if gaps:
            incomplete.append(
                {
                    "index": record.index,
                    "partition": assigned.get(record.index),
                    "prompt_sha256": _digest(record.prompt),
                    "coverage_non_whitespace": coverage(alignment),
                    "uncovered_spans": gaps,
                    "decision": "human_review_pending",
                }
            )

    def describe_group(members: list[int]) -> dict[str, Any]:
        sides = sorted({assigned[index] for index in members if index in assigned})
        return {"indices": members, "partitions": sides, "cross_partition": len(sides) > 1}

    exact = [
        {"normalised_sha256": _digest(text), **describe_group(members)}
        for text, members in exact_groups.items()
        if len(members) > 1
    ]
    candidates = []
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if normalised[left] == normalised[right]:
                continue
            union = tokens[left] | tokens[right]
            score = len(tokens[left] & tokens[right]) / len(union) if union else 0.0
            if score >= similarity_threshold:
                indices = [records[left].index, records[right].index]
                candidates.append({**describe_group(indices), "similarity": score})
                adjacency[indices[0]].add(indices[1])
                adjacency[indices[1]].add(indices[0])
    families = []
    seen: set[int] = set()
    for index in sorted(adjacency):
        if index in seen:
            continue
        pending = [index]
        members = []
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            members.append(current)
            pending.extend(adjacency[current] - seen)
        families.append(describe_group(sorted(members)))
    return {
        "audit_version": "1",
        "status": "diagnostics_complete_human_review_open",
        "records": len(records),
        "similarity_threshold": similarity_threshold,
        "normalisation": "lowercase; collapse whitespace; no punctuation removal",
        "candidate_metric": "Jaccard sets of lowercase Unicode regex \\w+ tokens",
        "candidate_limitations": "No semantics, multiplicity or order; components may chain. "
        "Not confirmed families, leakage, exclusions or partition assignments.",
        "incomplete_count": len(incomplete),
        "incomplete": incomplete,
        "tolerant_match_count": len(tolerant),
        "tolerant_matches": tolerant,
        "alignment_level_counts": dict(level_counts),
        "unresolved_leaves": unresolved,
        "normalised_exact_groups": exact,
        "near_duplicate_candidates": candidates,
        "candidate_components": families,
        "support_record_counts": {
            name: {path: counts[path] for path in LEAF_PATHS} for name, counts in support.items()
        },
    }


def audit_config(config_path: Path, *, similarity_threshold: float = 0.9) -> dict[str, Any]:
    """Read configured sources and existing splits without running preparation."""
    config = load_experiment(config_path)
    records: list[PreparedRecord] = []
    origins: dict[int, dict[str, Any]] = {}
    source_hashes = {}
    for source in config.data.sources:
        raw, _ = read_source(source, config.data.prompt_column, config.data.column_mapping)
        for offset, record in enumerate(raw):
            index = len(records)
            records.append(
                PreparedRecord(
                    index, record.prompt, serialise_target(assemble(record.leaves)), record.paradigm
                )
            )
            origins[index] = {
                "path": str(source.path),
                "source_index_zero_based": offset,
                "paradigm": record.paradigm,
            }
        for path in [source.path, source.fixes]:
            if path is not None:
                source_hashes[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    split = load_split(config.data.split.output_dir)
    partitions = {
        index: side for side in ("train", "val", "test") for index in getattr(split, side)
    }
    result = audit_records(
        records, partitions=partitions, similarity_threshold=similarity_threshold
    )
    result["provenance"] = {
        "config_path": str(config_path),
        "resolved_config_sha256": _digest(config.model_dump_json()),
        "source_sha256": source_hashes,
        "split_binding": "legacy membership read for diagnostics only; no corpus binding asserted",
        "split_sha256": {
            side: hashlib.sha256(
                (Path(config.data.split.output_dir) / f"{side}.json").read_bytes()
            ).hexdigest()
            for side in ("train", "val", "test")
        },
    }
    for section in ("incomplete", "tolerant_matches", "unresolved_leaves"):
        for item in result[section]:
            item["origin"] = origins[item["index"]]
    result["record_inventory"] = [
        {
            "index": record.index,
            "prompt_sha256": _digest(record.prompt),
            "target_sha256": _digest(record.target),
            "partition": partitions.get(record.index),
            **origins[record.index],
        }
        for record in records
    ]
    return result


def main() -> None:
    """Write a new diagnostic artifact, refusing to overwrite an existing file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    args = parser.parse_args()
    report = audit_config(args.config, similarity_threshold=args.similarity_threshold)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
