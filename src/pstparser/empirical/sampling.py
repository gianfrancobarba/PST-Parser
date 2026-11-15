"""Draw the task samples before any generation, from one frozen permutation.

Every order is defined by a published formula rather than by the state of a
random generator: tasks are sorted by ``SHA-256(seed | purpose | dataset hash |
task)``. The result depends on nothing but the seed and the pinned dataset, so
it can be recomputed in any language and on any platform.

The final sample is a prefix of a stratified permutation of the eligible tasks.
It is drawn first; the pilot pool is then drawn from what the prefix left out,
so the two can never overlap.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pstparser.empirical.benchmarks import SourceRecord

ORDER_METHOD: Final = "sha256-order-v1"
BIGCODEBENCH_STRATA: Final = "bigcodebench-exceptions-doctest-v1"

#: Reasons an exclusion may carry: properties of the benchmark, never of a model.
EXCLUSION_REASONS: Final = frozenset(
    {
        "reference_fails_offline",
        "trivial_program_passes",
        "empty_program_passes",
        "network_dependency",
    }
)

_EXCEPTIONS_SECTION: Final = "The function should raise the exception for:"
_DOCTEST_MARK: Final = ">>>"


def order_key(seed: int, purpose: str, dataset_sha256: str, item: str) -> str:
    """Return the sort key of one item for one seed, purpose and dataset."""
    if type(seed) is not int or seed < 0:
        raise ValueError("the seed must be a non-negative integer")
    material = f"{seed}|{purpose}|{dataset_sha256}|{item}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def seeded_order(items: Collection[str], seed: int, purpose: str, dataset_sha256: str) -> list[str]:
    """Order items by their key; the identifier breaks an (astronomically unlikely) tie."""
    if len(set(items)) != len(items):
        raise ValueError("duplicate item")
    return sorted(items, key=lambda item: (order_key(seed, purpose, dataset_sha256, item), item))


def bigcodebench_stratum(record: SourceRecord) -> str:
    """Stratum of a BigCodeBench prompt: exceptions section by doctest example."""
    exceptions = "exceptions" if _EXCEPTIONS_SECTION in record.text else "no-exceptions"
    doctest = "doctest" if _DOCTEST_MARK in record.text else "no-doctest"
    return f"{exceptions}/{doctest}"


STRATA: Final[dict[str, Callable[[SourceRecord], str]]] = {
    BIGCODEBENCH_STRATA: bigcodebench_stratum,
}


@dataclass(frozen=True)
class Permutation:
    """Frozen order of every eligible task, with what was left out and why."""

    order: tuple[str, ...]
    strata: Mapping[str, str]
    seed: int
    dataset_sha256: str
    text_field: str
    population: int
    excluded: Mapping[str, str]

    def stratum_sizes(self) -> dict[str, int]:
        """Number of eligible tasks in each stratum."""
        return dict(sorted(Counter(self.strata[task] for task in self.order).items()))


@dataclass(frozen=True)
class SampleRecord:
    """A sample: its tasks with their position in the permutation."""

    task_ids: tuple[str, ...]
    positions: tuple[int, ...]
    seed: int
    dataset_sha256: str
    text_field: str
    excluded: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "n": len(self.task_ids),
            "seed": self.seed,
            "dataset_file_sha256": self.dataset_sha256,
            "text_field": self.text_field,
            "excluded": dict(self.excluded),
            "tasks": [
                {"position": position, "task_id": task}
                for position, task in zip(self.positions, self.task_ids, strict=True)
            ],
        }


def stratified_permutation(
    records: Sequence[SourceRecord],
    *,
    dataset_sha256: str,
    seed: int,
    strata: Callable[[SourceRecord], str],
    exclude: Mapping[str, str],
) -> Permutation:
    """Order the eligible tasks so that every prefix is a proportional stratified sample.

    Within a stratum the order is the seeded one. Strata are then interleaved:
    position ``k`` goes to the stratum that is furthest behind its share
    ``k * N_s / N`` (integer arithmetic; ties go to the stratum name). Every
    prefix therefore holds each stratum in proportion to the population, within
    rounding, and needs no weight in the analysis.

    Raises:
        ValueError: If a record comes from another dataset file, an exclusion
            names an unknown task or carries a reason outside the closed list.
    """
    ids = [record.task_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task")
    fields = {record.text_field for record in records}
    if len(fields) != 1:
        raise ValueError("records must share one text field")
    for record in records:
        if record.dataset_file_sha256 != dataset_sha256:
            raise ValueError(f"{record.task_id}: the record comes from another dataset file")
    if not set(exclude) <= set(ids):
        raise ValueError("an exclusion names a task that is not in the dataset")
    for task, reason in exclude.items():
        if reason not in EXCLUSION_REASONS:
            raise ValueError(f"{task}: exclusion reason {reason!r} is not an admitted one")
    eligible = [record for record in records if record.task_id not in exclude]
    if not eligible:
        raise ValueError("no eligible task")
    stratum_of = {record.task_id: strata(record) for record in eligible}
    queues: dict[str, list[str]] = {}
    for name in sorted(set(stratum_of.values())):
        members = [task for task, stratum in stratum_of.items() if stratum == name]
        queues[name] = seeded_order(members, seed, "permutation", dataset_sha256)
    sizes = {name: len(queue) for name, queue in queues.items()}
    total = len(eligible)
    taken = dict.fromkeys(queues, 0)
    order: list[str] = []
    for position in range(1, total + 1):
        # Deficit against the proportional share, scaled by the total to stay
        # integral. Strata are visited by name and ``max`` keeps the first maximum.
        open_strata = [name for name in queues if taken[name] < sizes[name]]
        chosen = max(open_strata, key=lambda name: position * sizes[name] - taken[name] * total)
        order.append(queues[chosen][taken[chosen]])
        taken[chosen] += 1
    return Permutation(
        order=tuple(order),
        strata=stratum_of,
        seed=seed,
        dataset_sha256=dataset_sha256,
        text_field=fields.pop(),
        population=len(records),
        excluded=dict(exclude),
    )


def sample(permutation: Permutation, n: int = 800, skip: Collection[str] = ()) -> SampleRecord:
    """Take the first ``n`` tasks of the permutation, passing over ``skip``.

    ``skip`` is for an extension made after the pilot pool exists: the pool was
    drawn outside the original prefix, so skipping it never changes that prefix.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if not set(skip) <= set(permutation.order):
        raise ValueError("skip names a task outside the permutation")
    chosen = [
        (position, task) for position, task in enumerate(permutation.order) if task not in skip
    ][:n]
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} tasks available, {n} requested")
    return SampleRecord(
        task_ids=tuple(task for _, task in chosen),
        positions=tuple(position for position, _ in chosen),
        seed=permutation.seed,
        dataset_sha256=permutation.dataset_sha256,
        text_field=permutation.text_field,
        excluded=dict(permutation.excluded),
    )


def pilot_pool(
    permutation: Permutation,
    final: SampleRecord,
    n: int = 30,
    *,
    seed: int,
    hard_ids: Collection[str] | None,
) -> SampleRecord:
    """Draw the pilot tasks from the eligible tasks the final sample left out.

    The final sample must already exist and be a prefix of this permutation.
    ``hard_ids`` are left out of the pool; pass ``None`` only for a benchmark
    without a Hard subset. The draw uses its own order (purpose ``pilot``), so
    it is not the continuation of the permutation.
    """
    if final.dataset_sha256 != permutation.dataset_sha256 or final.seed != permutation.seed:
        raise ValueError("the final sample does not belong to this permutation")
    if final.task_ids != permutation.order[: len(final.task_ids)]:
        raise ValueError("the final sample is not a prefix of this permutation")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    outside = set(permutation.order) - set(final.task_ids) - set(hard_ids or ())
    drawn = seeded_order(outside, seed, "pilot", permutation.dataset_sha256)[:n]
    if len(drawn) < n:
        raise ValueError(f"only {len(drawn)} pilot candidates available, {n} requested")
    position = {task: index for index, task in enumerate(permutation.order)}
    return SampleRecord(
        task_ids=tuple(drawn),
        positions=tuple(position[task] for task in drawn),
        seed=seed,
        dataset_sha256=permutation.dataset_sha256,
        text_field=permutation.text_field,
        excluded=dict(permutation.excluded),
    )


def full_minus(permutation: Permutation, pilot: SampleRecord) -> SampleRecord:
    """Return every eligible task but the pilot ones, in permutation order."""
    return sample(permutation, len(permutation.order) - len(pilot.task_ids), skip=pilot.task_ids)


def family_sample(
    records: Sequence[SourceRecord],
    *,
    families: int,
    seed: int,
    dataset_sha256: str,
    family_size: int | None = None,
    purpose: str = "pilot-families",
) -> list[SourceRecord]:
    """Draw whole families: every record of a drawn family is returned.

    With ``family_size`` only the families of exactly that size are candidates.
    Families come in the seeded order, their records by identifier.
    """
    members: dict[str, list[SourceRecord]] = {}
    for record in records:
        if record.family_id is None:
            raise ValueError(f"{record.task_id}: no family")
        if record.dataset_file_sha256 != dataset_sha256:
            raise ValueError(f"{record.task_id}: the record comes from another dataset file")
        members.setdefault(record.family_id, []).append(record)
    candidates = [
        family
        for family, group in members.items()
        if family_size is None or len(group) == family_size
    ]
    if type(families) is not int or families < 1:
        raise ValueError("families must be a positive integer")
    drawn = seeded_order(candidates, seed, purpose, dataset_sha256)[:families]
    if len(drawn) < families:
        raise ValueError(f"only {len(drawn)} candidate families available, {families} requested")
    return [
        record
        for family in drawn
        for record in sorted(members[family], key=lambda source: source.task_id)
    ]


def shared_families(first: Sequence[SourceRecord], second: Sequence[SourceRecord]) -> set[str]:
    """Families that appear in both sets; a non-empty result means a split family."""
    left = {record.family_id for record in first if record.family_id is not None}
    right = {record.family_id for record in second if record.family_id is not None}
    return left & right


def permutation_document(
    permutation: Permutation,
    *,
    benchmark: str,
    dataset_revision: str,
    strata_rule: str,
    hard_ids: Collection[str],
) -> dict[str, Any]:
    """Describe a permutation as the JSON document that gets frozen."""
    return {
        "format_version": 1,
        "kind": "permutation",
        "benchmark": benchmark,
        "dataset_revision": dataset_revision,
        "dataset_file_sha256": permutation.dataset_sha256,
        "text_field": permutation.text_field,
        "method": ORDER_METHOD,
        "strata_rule": strata_rule,
        "seed": permutation.seed,
        "population": permutation.population,
        "eligible": len(permutation.order),
        "eligible_hard": sum(1 for task in permutation.order if task in hard_ids),
        "stratum_sizes": permutation.stratum_sizes(),
        "excluded": dict(permutation.excluded),
        "order": [
            {
                "position": position,
                "task_id": task,
                "stratum": permutation.strata[task],
                "hard": task in hard_ids,
            }
            for position, task in enumerate(permutation.order)
        ],
    }


def read_permutation_document(document: Mapping[str, Any]) -> tuple[Permutation, frozenset[str]]:
    """Rebuild a permutation and its Hard tasks from a frozen document."""
    if document.get("format_version") != 1 or document.get("kind") != "permutation":
        raise ValueError("not a permutation document")
    if document["method"] != ORDER_METHOD:
        raise ValueError(f"unknown ordering method {document['method']!r}")
    rows = document["order"]
    if [row["position"] for row in rows] != list(range(len(rows))):
        raise ValueError("permutation positions are not consecutive")
    order = tuple(row["task_id"] for row in rows)
    if len(set(order)) != len(order) or len(order) != document["eligible"]:
        raise ValueError("the permutation does not list every eligible task once")
    permutation = Permutation(
        order=order,
        strata={row["task_id"]: row["stratum"] for row in rows},
        seed=document["seed"],
        dataset_sha256=document["dataset_file_sha256"],
        text_field=document["text_field"],
        population=document["population"],
        excluded=dict(document["excluded"]),
    )
    return permutation, frozenset(row["task_id"] for row in rows if row["hard"])


def sample_document(
    kind: str,
    drawn: SampleRecord,
    permutation: Permutation,
    hard_ids: Collection[str],
    **provenance: Any,
) -> dict[str, Any]:
    """Describe a sample drawn from a permutation, with strata and Hard counts."""
    tasks = [
        {
            "position": position,
            "task_id": task,
            "stratum": permutation.strata[task],
            "hard": task in hard_ids,
        }
        for position, task in zip(drawn.positions, drawn.task_ids, strict=True)
    ]
    return {
        "format_version": 1,
        "kind": kind,
        "n": len(tasks),
        "seed": drawn.seed,
        "dataset_file_sha256": drawn.dataset_sha256,
        "text_field": drawn.text_field,
        **provenance,
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in tasks).items())),
        "hard_count": sum(1 for row in tasks if row["hard"]),
        "tasks": tasks,
    }


def read_sample_document(document: Mapping[str, Any], kind: str) -> SampleRecord:
    """Rebuild a sample from its document; the kind must be the expected one."""
    if document.get("format_version") != 1 or document.get("kind") != kind:
        raise ValueError(f"not a {kind} document")
    rows = document["tasks"]
    if len(rows) != document["n"]:
        raise ValueError("the sample does not hold the declared number of tasks")
    return SampleRecord(
        task_ids=tuple(row["task_id"] for row in rows),
        positions=tuple(row["position"] for row in rows),
        seed=document["seed"],
        dataset_sha256=document["dataset_file_sha256"],
        text_field=document["text_field"],
        excluded={},
    )
