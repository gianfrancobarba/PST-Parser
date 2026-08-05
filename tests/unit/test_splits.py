"""Partitioning of the corpus and persistence of the resulting indices."""

from __future__ import annotations

from pathlib import Path

import pytest

from pstparser.data import Split, group_key, load_split, make_split, save_split


def keys(count: int) -> list[str]:
    """Build one key per record, all distinct, so that nothing is grouped."""
    return [group_key(f"prompt number {index}") for index in range(count)]


def test_partition_covers_every_record_exactly_once() -> None:
    split = make_split(keys(100), val_fraction=0.1, test_fraction=0.1, random_state=42)

    assert sorted(split.train + split.val + split.test) == list(range(100))
    assert len(split) == 100


def test_each_side_lands_near_its_requested_share() -> None:
    split = make_split(keys(1000), val_fraction=0.09, test_fraction=0.09, random_state=42)

    assert 80 <= len(split.val) <= 100
    assert 80 <= len(split.test) <= 100
    assert len(split.train) > len(split.val) + len(split.test)


def test_records_sharing_a_prompt_stay_on_the_same_side() -> None:
    # Five distinct prompts, each written four times, interleaved.
    prompts = [f"prompt {index % 5}" for index in range(20)]

    split = make_split(
        [group_key(prompt) for prompt in prompts],
        val_fraction=0.2,
        test_fraction=0.2,
        random_state=7,
    )

    sides = (set(split.train), set(split.val), set(split.test))
    for group in range(5):
        members = {index for index, prompt in enumerate(prompts) if prompt == f"prompt {group}"}
        assert any(members <= side for side in sides)


def test_spacing_and_case_do_not_make_a_new_prompt() -> None:
    assert group_key("Summarise  the\nCODE.") == group_key("summarise the code.")


def test_partition_is_deterministic_for_a_given_seed() -> None:
    first = make_split(keys(60), val_fraction=0.1, test_fraction=0.1, random_state=11)
    second = make_split(keys(60), val_fraction=0.1, test_fraction=0.1, random_state=11)

    assert first == second


def test_partition_changes_with_the_seed() -> None:
    first = make_split(keys(60), val_fraction=0.1, test_fraction=0.1, random_state=1)
    second = make_split(keys(60), val_fraction=0.1, test_fraction=0.1, random_state=2)

    assert first != second


def test_partition_is_shuffled() -> None:
    split = make_split(keys(60), val_fraction=0.1, test_fraction=0.1, random_state=3)

    assert split.test != list(range(len(split.test)))


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        make_split([], val_fraction=0.1, test_fraction=0.1, random_state=42)


def test_fractions_that_leave_no_training_set_are_rejected() -> None:
    with pytest.raises(ValueError, match="must leave a training set"):
        make_split(keys(50), val_fraction=0.6, test_fraction=0.5, random_state=42)


def test_a_side_that_comes_out_empty_is_rejected() -> None:
    # Every record carries the same prompt, so they form one block that cannot
    # be divided: whichever side takes it, the other two are left with nothing.
    with pytest.raises(ValueError, match="came out empty"):
        make_split([group_key("same")] * 3, val_fraction=0.3, test_fraction=0.3, random_state=42)


def test_partition_round_trips_through_disk(tmp_path: Path) -> None:
    split = make_split(keys(40), val_fraction=0.15, test_fraction=0.15, random_state=5)

    train_path, val_path, test_path = save_split(split, tmp_path / "nested")

    assert train_path.is_file()
    assert val_path.is_file()
    assert test_path.is_file()
    assert load_split(tmp_path / "nested") == split


def test_a_missing_partition_file_is_reported(tmp_path: Path) -> None:
    save_split(Split(train=[0], val=[1], test=[2]), tmp_path)
    (tmp_path / "test.json").unlink()

    with pytest.raises(FileNotFoundError, match="split file not found"):
        load_split(tmp_path)


def test_loading_an_absent_partition_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="split file not found"):
        load_split(tmp_path)
