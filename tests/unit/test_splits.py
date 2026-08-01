"""Partitioning of the corpus and persistence of the resulting indices."""

from __future__ import annotations

from pathlib import Path

import pytest

from pstparser.data import load_split, make_split, save_split


def test_partition_covers_every_record_exactly_once() -> None:
    split = make_split(total=100, eval_fraction=0.1, random_state=42)

    assert sorted(split.train + split.eval) == list(range(100))
    assert len(split) == 100


def test_evaluation_size_is_rounded_up() -> None:
    split = make_split(total=975, eval_fraction=0.09, random_state=42)

    assert len(split.eval) == 88
    assert len(split.train) == 887


def test_partition_is_deterministic_for_a_given_seed() -> None:
    first = make_split(total=50, eval_fraction=0.2, random_state=7)
    second = make_split(total=50, eval_fraction=0.2, random_state=7)

    assert first == second


def test_partition_changes_with_the_seed() -> None:
    first = make_split(total=50, eval_fraction=0.2, random_state=7)
    second = make_split(total=50, eval_fraction=0.2, random_state=8)

    assert first != second


def test_partition_is_shuffled() -> None:
    split = make_split(total=100, eval_fraction=0.1, random_state=42)

    assert split.train != sorted(split.train)


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        make_split(total=0, eval_fraction=0.1, random_state=42)


def test_fraction_leaving_an_empty_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot partition 3 records"):
        make_split(total=3, eval_fraction=0.99, random_state=42)


def test_partition_round_trips_through_disk(tmp_path: Path) -> None:
    split = make_split(total=40, eval_fraction=0.25, random_state=1)

    train_path, eval_path = save_split(split, tmp_path / "nested")

    assert train_path.is_file()
    assert eval_path.is_file()
    assert load_split(tmp_path / "nested") == split


def test_loading_an_absent_partition_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="split file not found"):
        load_split(tmp_path)
