"""Task sampling: one frozen permutation, final sample first, pilot from the rest."""

import hashlib
from collections import Counter
from fractions import Fraction

import pytest

from pstparser.empirical.benchmarks import SourceRecord
from pstparser.empirical.sampling import (
    BIGCODEBENCH_STRATA,
    STRATA,
    Permutation,
    bigcodebench_stratum,
    family_sample,
    full_minus,
    order_key,
    pilot_pool,
    sample,
    seeded_order,
    shared_families,
    stratified_permutation,
)

DATASET = "d" * 64
SEED = 7319
# Sizes of the four strata, small enough to enumerate and unequal on purpose.
SIZES = {"a": 61, "b": 23, "c": 11, "d": 5}
TOTAL = sum(SIZES.values())


def record(
    task: str, text: str = "x", family: str | None = None, dataset: str = DATASET
) -> SourceRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceRecord(
        task_id=task,
        text=text,
        source_sha256=digest,
        original_sha256=digest,
        benchmark="bigcodebench",
        dataset_revision="1" * 40,
        dataset_file_sha256=dataset,
        original_id=task,
        family_id=family,
        text_field="instruct_prompt",
        transforms=(),
    )


def population() -> list[SourceRecord]:
    # The text is the stratum label, so the strata function is the identity.
    records = [
        record(f"T/{name}{index}", name) for name, size in SIZES.items() for index in range(size)
    ]
    # Two more tasks, to be excluded.
    return [*records, record("T/x0", "a"), record("T/x1", "b")]


EXCLUDE = {"T/x0": "reference_fails_offline", "T/x1": "trivial_program_passes"}


def permutation(seed: int = SEED) -> Permutation:
    return stratified_permutation(
        population(),
        dataset_sha256=DATASET,
        seed=seed,
        strata=lambda source: source.text,
        exclude=EXCLUDE,
    )


# ------------------------------------------------------------------------------- order


def test_order_key_is_the_published_formula() -> None:
    expected = hashlib.sha256(f"7319|permutation|{DATASET}|BigCodeBench/5".encode()).hexdigest()

    assert order_key(7319, "permutation", DATASET, "BigCodeBench/5") == expected
    assert order_key(7319, "pilot", DATASET, "BigCodeBench/5") != expected
    assert order_key(7320, "permutation", DATASET, "BigCodeBench/5") != expected
    assert order_key(7319, "permutation", "e" * 64, "BigCodeBench/5") != expected
    with pytest.raises(ValueError, match="seed"):
        order_key(-1, "permutation", DATASET, "x")
    with pytest.raises(ValueError, match="seed"):
        order_key(True, "permutation", DATASET, "x")


def test_seeded_order_does_not_depend_on_the_input_order() -> None:
    items = [f"T/{index}" for index in range(50)]

    forward = seeded_order(items, SEED, "permutation", DATASET)
    backward = seeded_order(list(reversed(items)), SEED, "permutation", DATASET)

    assert forward == backward
    assert sorted(forward) == sorted(items)
    assert forward != items and forward != sorted(items)
    with pytest.raises(ValueError, match="duplicate"):
        seeded_order(["a", "a"], SEED, "permutation", DATASET)


# ------------------------------------------------------------------------- permutation


def test_same_seed_gives_the_same_permutation_and_another_seed_another() -> None:
    first, second, other = permutation(), permutation(), permutation(seed=7320)

    assert first.order == second.order
    assert first.order != other.order
    assert sorted(first.order) == sorted(other.order)


def test_permutation_holds_every_eligible_task_once_and_no_excluded_one() -> None:
    frozen = permutation()

    assert len(frozen.order) == len(set(frozen.order)) == TOTAL
    assert not set(frozen.order) & set(EXCLUDE)
    assert frozen.population == TOTAL + 2
    assert frozen.excluded == EXCLUDE
    assert frozen.stratum_sizes() == SIZES
    assert frozen.seed == SEED and frozen.dataset_sha256 == DATASET
    assert frozen.text_field == "instruct_prompt"


def test_every_prefix_is_proportional_to_the_strata_within_rounding() -> None:
    frozen = permutation()

    counts: Counter[str] = Counter()
    for size, task in enumerate(frozen.order, start=1):
        counts[frozen.strata[task]] += 1
        for name, stratum_size in SIZES.items():
            share = Fraction(size * stratum_size, TOTAL)
            assert abs(counts[name] - share) < 1, (size, name)
    assert counts == SIZES


def test_order_inside_a_stratum_is_the_seeded_order() -> None:
    frozen = permutation()

    for name in SIZES:
        members = [task for task in frozen.order if frozen.strata[task] == name]
        assert members == seeded_order(members, SEED, "permutation", DATASET)


def test_permutation_refuses_a_record_from_another_dataset_file() -> None:
    records = [*population(), record("T/z", "a", dataset="e" * 64)]

    with pytest.raises(ValueError, match=r"T/z.*another dataset file"):
        stratified_permutation(
            records, dataset_sha256=DATASET, seed=SEED, strata=lambda s: s.text, exclude={}
        )


@pytest.mark.parametrize(
    "exclude,message",
    [
        ({"T/absent": "reference_fails_offline"}, "not in the dataset"),
        ({"T/a0": "the model failed on it"}, "not an admitted one"),
        ({"T/a0": ""}, "not an admitted one"),
    ],
)
def test_exclusions_need_a_known_task_and_an_enumerated_reason(
    exclude: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        stratified_permutation(
            population(),
            dataset_sha256=DATASET,
            seed=SEED,
            strata=lambda s: s.text,
            exclude=exclude,
        )


def test_bigcodebench_strata_are_exceptions_by_doctest() -> None:
    output = "The function should output with:\n    int: x\n"
    raises = "The function should raise the exception for: ValueError: bad.\n"

    assert bigcodebench_stratum(record("a", "Do it.\n" + output)) == "no-exceptions/no-doctest"
    assert (
        bigcodebench_stratum(record("b", "Do it.\n" + raises + output)) == "exceptions/no-doctest"
    )
    assert (
        bigcodebench_stratum(record("c", "Do it.\n>>> f(1)\n1\n" + output))
        == "no-exceptions/doctest"
    )
    assert (
        bigcodebench_stratum(record("d", "Do it.\n>>> f(1)\n1\n" + raises + output))
        == "exceptions/doctest"
    )
    assert STRATA[BIGCODEBENCH_STRATA] is bigcodebench_stratum


# ------------------------------------------------------------------------------ samples


def test_smaller_and_larger_samples_are_prefixes_of_the_same_permutation() -> None:
    frozen = permutation()

    sixty, eighty = sample(frozen, 60), sample(frozen, 80)

    assert sixty.task_ids == frozen.order[:60]
    assert eighty.task_ids == frozen.order[:80]
    assert eighty.task_ids[:60] == sixty.task_ids
    assert eighty.positions == tuple(range(80))
    assert (eighty.seed, eighty.dataset_sha256, eighty.text_field) == (
        SEED,
        DATASET,
        "instruct_prompt",
    )
    assert eighty.excluded == EXCLUDE
    assert eighty.as_dict()["tasks"][3] == {"position": 3, "task_id": frozen.order[3]}
    assert eighty.as_dict()["n"] == 80


def test_sample_strata_match_the_population_within_rounding() -> None:
    frozen = permutation()

    counts = Counter(frozen.strata[task] for task in sample(frozen, 80).task_ids)

    for name, size in SIZES.items():
        assert abs(counts[name] - Fraction(80 * size, TOTAL)) < 1


def test_sample_refuses_more_tasks_than_there_are() -> None:
    frozen = permutation()

    assert sample(frozen, TOTAL).task_ids == frozen.order
    with pytest.raises(ValueError, match="only 100 tasks available, 101 requested"):
        sample(frozen, TOTAL + 1)
    with pytest.raises(ValueError, match="positive"):
        sample(frozen, 0)


def test_pilot_is_disjoint_from_the_final_sample_and_from_hard() -> None:
    frozen = permutation()
    final = sample(frozen, 80)
    hard = set(frozen.order[80:90]) | set(frozen.order[:5])

    pilot = pilot_pool(frozen, final, 6, seed=SEED, hard_ids=hard)

    assert len(pilot.task_ids) == len(set(pilot.task_ids)) == 6
    assert not set(pilot.task_ids) & set(final.task_ids)
    assert not set(pilot.task_ids) & hard
    assert set(pilot.task_ids) <= set(frozen.order[90:])
    assert pilot.positions == tuple(frozen.order.index(task) for task in pilot.task_ids)
    # The pilot is its own draw, not the next tasks of the permutation.
    assert pilot.task_ids == tuple(seeded_order(frozen.order[90:], SEED, "pilot", DATASET)[:6])
    assert set(pilot.task_ids) != set(frozen.order[90:96])
    assert pilot_pool(frozen, final, 6, seed=SEED, hard_ids=hard) == pilot
    assert pilot_pool(frozen, final, 6, seed=SEED + 1, hard_ids=hard) != pilot


def test_pilot_needs_the_final_sample_to_be_a_prefix_of_the_permutation() -> None:
    frozen = permutation()
    final = sample(frozen, 80)
    not_a_prefix = sample(frozen, 80, skip=frozen.order[:1])
    foreign = sample(permutation(seed=7320), 80)

    with pytest.raises(ValueError, match="not a prefix"):
        pilot_pool(frozen, not_a_prefix, 6, seed=SEED, hard_ids=())
    with pytest.raises(ValueError, match="does not belong"):
        pilot_pool(frozen, foreign, 6, seed=SEED, hard_ids=())
    with pytest.raises(ValueError, match="only 20 pilot candidates available, 21 requested"):
        pilot_pool(frozen, final, 21, seed=SEED, hard_ids=())
    with pytest.raises(ValueError, match="positive"):
        pilot_pool(frozen, final, 0, seed=SEED, hard_ids=())


def test_extension_by_prefix_keeps_the_first_sample_and_skips_the_pilot() -> None:
    frozen = permutation()
    final = sample(frozen, 80)
    pilot = pilot_pool(frozen, final, 6, seed=SEED, hard_ids=())

    extended = sample(frozen, 90, skip=pilot.task_ids)

    assert extended.task_ids[:80] == final.task_ids
    assert not set(extended.task_ids) & set(pilot.task_ids)
    assert extended.positions == tuple(sorted(extended.positions))
    with pytest.raises(ValueError, match="outside the permutation"):
        sample(frozen, 90, skip=["T/absent"])


def test_full_set_is_every_eligible_task_but_the_pilot() -> None:
    frozen = permutation()
    final = sample(frozen, 80)
    pilot = pilot_pool(frozen, final, 6, seed=SEED, hard_ids=())

    full = full_minus(frozen, pilot)

    assert len(full.task_ids) == TOTAL - 6
    assert set(full.task_ids) == set(frozen.order) - set(pilot.task_ids)
    assert full.task_ids[:80] == final.task_ids


# ----------------------------------------------------------------------------- families


def families() -> list[SourceRecord]:
    sizes = {"F1": 3, "F2": 3, "F3": 2, "F4": 3, "F5": 3, "F6": 1, "F7": 3}
    return [
        record(f"LSE-{family}-{index}", family=family)
        for family, size in sizes.items()
        for index in range(size)
    ]


def test_family_sample_returns_whole_families_of_the_requested_size() -> None:
    records = families()

    drawn = family_sample(records, families=4, seed=SEED, dataset_sha256=DATASET, family_size=3)

    by_family = Counter(source.family_id for source in drawn)
    assert len(drawn) == 12
    assert set(by_family.values()) == {3}
    assert set(by_family) <= {"F1", "F2", "F4", "F5", "F7"}
    assert (
        list(by_family)
        == seeded_order(["F1", "F2", "F4", "F5", "F7"], SEED, "pilot-families", DATASET)[:4]
    )
    # Neither the order of the input nor the order inside a family matters.
    assert drawn == family_sample(
        list(reversed(records)), families=4, seed=SEED, dataset_sha256=DATASET, family_size=3
    )
    assert [source.task_id for source in drawn[:3]] == sorted(s.task_id for s in drawn[:3])


def test_family_sample_refuses_too_few_candidates_and_records_without_family() -> None:
    with pytest.raises(ValueError, match="only 5 candidate families available, 6 requested"):
        family_sample(families(), families=6, seed=SEED, dataset_sha256=DATASET, family_size=3)
    with pytest.raises(ValueError, match="no family"):
        family_sample([record("T/1")], families=1, seed=SEED, dataset_sha256=DATASET)
    with pytest.raises(ValueError, match="another dataset file"):
        family_sample(families(), families=1, seed=SEED, dataset_sha256="e" * 64)


def test_no_family_is_split_between_pilot_and_final() -> None:
    records = families()
    pilot = family_sample(records, families=2, seed=SEED, dataset_sha256=DATASET, family_size=3)
    pilot_families = {source.family_id for source in pilot}
    final = [source for source in records if source.family_id not in pilot_families]

    assert shared_families(pilot, final) == set()
    assert shared_families(pilot, records) == pilot_families
