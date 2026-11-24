"""Prespecified exploratory analyses of contamination and near duplicates.

- Contamination: how much of a task's reference solution a generated program
  reproduces, as the share of the distinct token 10-grams of the reference that
  also occur in the program; a program is "highly similar" at or above a
  threshold read from the frozen configuration (0.5). The reference
  solution is read only here, at analysis time, from the oracle file: it never
  reaches generation or rendering.
- Near duplicates: BigCodeBench prompts whose sets of word tokens have
  a Jaccard index of at least 0.8, joined into connected groups; the S-NEARDUP
  sensitivity resamples these groups as clusters, every other task alone.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from pstparser.empirical.analysis import TaskEstimate, _bootstrap

CONTAMINATION_FORMAT: Final = "contamination-config-v1"
NEAR_DUPLICATE_RULE: Final = "near-duplicates-jaccard-v1"
_WORD: Final = re.compile(r"\w+")


@dataclass(frozen=True)
class ContaminationConfig:
    """Frozen parameters of the contamination analysis."""

    n: int
    threshold: float
    tokenizer: str

    def __post_init__(self) -> None:
        """Refuse a non-positive n or a threshold outside (0, 1]."""
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer")
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must lie in (0, 1]")
        if not self.tokenizer.strip():
            raise ValueError("the tokenizer must be named")


def load_contamination_config(path: str | Path) -> ContaminationConfig:
    """Read the frozen configuration; every key is required and no other is allowed.

    Raises:
        ValueError: If the format or the key set is not the expected one.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != CONTAMINATION_FORMAT:
        raise ValueError(f"expected format {CONTAMINATION_FORMAT}")
    keys = {"format", "n", "threshold", "tokenizer"}
    if set(document) != keys:
        raise ValueError(f"the configuration requires exactly: {sorted(keys)}")
    return ContaminationConfig(document["n"], document["threshold"], document["tokenizer"])


def ngrams(tokens: Sequence[Hashable], n: int) -> set[tuple[Hashable, ...]]:
    """Distinct contiguous n-grams of a token sequence."""
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def ngram_overlap(
    output: Sequence[Hashable], reference: Sequence[Hashable], n: int = 10
) -> float | None:
    """Share of the distinct n-grams of ``reference`` that also occur in ``output``.

    1 when the output contains the reference, 0 when they share no n-gram; None
    when the reference is shorter than n tokens, so the share is undefined.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    wanted = ngrams(reference, n)
    if not wanted:
        return None
    return len(wanted & ngrams(output, n)) / len(wanted)


def contamination_rows(
    programs: Sequence[Mapping[str, Any]],
    solutions: Mapping[str, str],
    tokenize: Callable[[str], Sequence[Hashable]],
    config: ContaminationConfig,
) -> list[dict[str, Any]]:
    """Overlap and the high-similarity flag of every program.

    ``programs`` rows carry ``task_id``, ``condition``, ``replicate_id`` and
    ``program`` (the code given to the oracle). A program whose reference is
    too short to define an overlap is flagged ``None``, never "low".

    Raises:
        ValueError: If a task has no reference solution.
    """
    rows = []
    for row in programs:
        if row["task_id"] not in solutions:
            raise ValueError(f"{row['task_id']}: no reference solution")
        overlap = ngram_overlap(
            tokenize(row["program"]), tokenize(solutions[row["task_id"]]), config.n
        )
        rows.append(
            {
                "task_id": row["task_id"],
                "condition": row["condition"],
                "replicate_id": row["replicate_id"],
                "overlap": overlap,
                "high_similarity": None if overlap is None else overlap >= config.threshold,
            }
        )
    return rows


def similarity_strata(
    rows: Sequence[Mapping[str, Any]], condition: str = "T"
) -> dict[str, list[str]]:
    """Tasks by the similarity of their ``condition`` arm: ``high``, ``low``, ``undefined``.

    With several replicates a task is high when any replicate of the arm is.
    """
    flags: dict[str, list[bool | None]] = {}
    for row in rows:
        if row["condition"] == condition:
            flags.setdefault(row["task_id"], []).append(row["high_similarity"])
    strata: dict[str, list[str]] = {"high": [], "low": [], "undefined": []}
    for task, values in sorted(flags.items()):
        if any(value is True for value in values):
            strata["high"].append(task)
        elif all(value is None for value in values):
            strata["undefined"].append(task)
        else:
            strata["low"].append(task)
    return strata


def word_tokens(text: str) -> set[str]:
    """Lower-cased word tokens, the token set of the near-duplicate check."""
    return set(_WORD.findall(text.lower()))


def near_duplicate_groups(texts: Mapping[str, str], threshold: float = 0.8) -> list[list[str]]:
    """Connected groups of tasks whose token sets have Jaccard >= ``threshold``.

    Groups are sorted by decreasing size, then by their first task; tasks in a
    group keep the order of ``texts``. Tasks without a near duplicate are left out.
    """
    names = list(texts)
    tokens = [word_tokens(texts[name]) for name in names]
    parent = list(range(len(names)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            union = tokens[i] | tokens[j]
            if union and len(tokens[i] & tokens[j]) / len(union) >= threshold:
                parent[root(i)] = root(j)
    groups: dict[int, list[str]] = {}
    for index, name in enumerate(names):
        groups.setdefault(root(index), []).append(name)
    found = [group for group in groups.values() if len(group) > 1]
    return sorted(found, key=lambda group: (-len(group), names.index(group[0])))


def near_duplicate_bootstrap(
    estimates: Sequence[TaskEstimate],
    groups: Sequence[Sequence[str]],
    *,
    samples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[tuple[float, float], int]:
    """S-NEARDUP: percentile bootstrap with near-duplicate groups as clusters.

    Every task outside a group is its own cluster. Returns the interval and the
    number of clusters.

    Raises:
        ValueError: If a task is in two groups or already has a family.
    """
    cluster: dict[str, str] = {}
    for index, group in enumerate(groups):
        for task in group:
            if task in cluster:
                raise ValueError(f"{task} is in two near-duplicate groups")
            cluster[task] = f"near-duplicates-{index}"
    if any(estimate.family_id is not None for estimate in estimates):
        raise ValueError("S-NEARDUP applies to tasks without families")
    relabelled = [
        replace(estimate, family_id=cluster.get(estimate.task_id, estimate.task_id))
        for estimate in estimates
    ]
    interval, clusters, _ = _bootstrap(
        relabelled, samples=samples, seed=seed, confidence=confidence
    )
    return interval, clusters
