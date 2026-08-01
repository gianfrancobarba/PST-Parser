"""Partitioning of the corpus into a training and an evaluation set.

Splits are computed once and written to disk as lists of record indices, so
that every later stage reads the same partition instead of recomputing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

TRAIN_FILE = "train.json"
EVAL_FILE = "eval.json"


@dataclass(frozen=True)
class Split:
    """Indices of the records assigned to each partition.

    Attributes:
        train: Indices of the training records.
        eval: Indices of the evaluation records.
    """

    train: list[int]
    eval: list[int]

    def __len__(self) -> int:
        """Total number of partitioned records."""
        return len(self.train) + len(self.eval)


def make_split(total: int, eval_fraction: float, random_state: int) -> Split:
    """Partition record indices into a training and an evaluation set.

    Args:
        total: Number of records in the corpus.
        eval_fraction: Fraction of records held out for evaluation. The size of
            the evaluation set is rounded up.
        random_state: Seed of the shuffle applied before partitioning.

    Returns:
        The partition.

    Raises:
        ValueError: If the corpus is empty, or if the requested fraction leaves
            one of the two sides empty.
    """
    if total <= 0:
        raise ValueError("cannot partition an empty corpus")

    try:
        train, evaluation = train_test_split(
            list(range(total)),
            test_size=eval_fraction,
            random_state=random_state,
        )
    except ValueError as exc:
        raise ValueError(
            f"cannot partition {total} records with eval_fraction {eval_fraction}: {exc}"
        ) from exc

    return Split(train=list(train), eval=list(evaluation))


def save_split(split: Split, directory: str | Path) -> tuple[Path, Path]:
    """Write a partition to disk.

    Args:
        split: The partition to persist.
        directory: Destination directory, created if absent.

    Returns:
        The paths of the training and evaluation files.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    train_path = directory / TRAIN_FILE
    eval_path = directory / EVAL_FILE
    train_path.write_text(json.dumps(split.train), encoding="utf-8")
    eval_path.write_text(json.dumps(split.eval), encoding="utf-8")
    return train_path, eval_path


def load_split(directory: str | Path) -> Split:
    """Read a partition previously written by :func:`save_split`.

    Args:
        directory: Directory holding the partition files.

    Returns:
        The partition.

    Raises:
        FileNotFoundError: If either partition file is missing.
    """
    directory = Path(directory)
    train_path = directory / TRAIN_FILE
    eval_path = directory / EVAL_FILE
    for path in (train_path, eval_path):
        if not path.is_file():
            raise FileNotFoundError(f"split file not found: {path}")

    return Split(
        train=json.loads(train_path.read_text(encoding="utf-8")),
        eval=json.loads(eval_path.read_text(encoding="utf-8")),
    )
