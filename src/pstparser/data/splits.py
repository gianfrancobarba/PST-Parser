"""Partitioning of the corpus into training, validation and test sets.

Three sets rather than two. Watching one set while training and then reporting
on the same set does not measure generalisation: it measures how well the run
was stopped. The validation set is what the training loop may look at; the test
set is what the results are quoted from, and it is read once.

Records that share a prompt are kept on the same side. The corpus holds prompts
that occur more than once, and letting a prompt appear in training and again in
the test set would let a model score on text it had already been shown.

Splits are computed once and written to disk as lists of record indices, so that
every later stage reads the same partition instead of recomputing it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TRAIN_FILE = "train.json"
VAL_FILE = "val.json"
TEST_FILE = "test.json"


@dataclass(frozen=True)
class Split:
    """Indices of the records assigned to each partition.

    Attributes:
        train: Indices the model is fitted on.
        val: Indices watched during training.
        test: Indices the reported results are computed on.
    """

    train: list[int]
    val: list[int]
    test: list[int]

    def __len__(self) -> int:
        """Total number of partitioned records."""
        return len(self.train) + len(self.val) + len(self.test)


def group_key(prompt: str) -> str:
    """Build the key that decides which records must stay together.

    Two prompts that differ only in spacing or in case are the same prompt for
    this purpose: the point is to keep a model from being tested on text it was
    trained on, and neither difference would prevent that.

    Args:
        prompt: The raw prompt.

    Returns:
        A digest of its normalised form.
    """
    normalised = " ".join(prompt.split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def make_split(
    keys: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    random_state: int,
) -> Split:
    """Partition records into training, validation and test sets.

    Records sharing a key are assigned as a block, so the sizes land near the
    requested fractions rather than exactly on them.

    Args:
        keys: Grouping key of each record, in corpus order.
        val_fraction: Share of records to watch during training.
        test_fraction: Share of records to reserve for the reported results.
        random_state: Seed of the shuffle applied to the groups.

    Returns:
        The partition, with the indices of each side in increasing order.

    Raises:
        ValueError: If the corpus is empty, if the fractions do not leave a
            training set, or if a side comes out empty.
    """
    total = len(keys)
    if total <= 0:
        raise ValueError("cannot partition an empty corpus")
    if not 0.0 < val_fraction + test_fraction < 1.0:
        raise ValueError(
            f"val_fraction and test_fraction must leave a training set, got "
            f"{val_fraction} and {test_fraction}"
        )

    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)

    order = sorted(groups)
    random.Random(random_state).shuffle(order)

    wanted = {"test": test_fraction * total, "val": val_fraction * total}
    assigned: dict[str, list[int]] = {"test": [], "val": [], "train": []}
    for key in order:
        side = next((name for name, size in wanted.items() if len(assigned[name]) < size), "train")
        assigned[side].extend(groups[key])

    split = Split(
        train=sorted(assigned["train"]),
        val=sorted(assigned["val"]),
        test=sorted(assigned["test"]),
    )
    empty = [
        name
        for name, side in (("train", split.train), ("val", split.val), ("test", split.test))
        if not side
    ]
    if empty:
        raise ValueError(
            f"cannot partition {total} records into the requested fractions: "
            f"{', '.join(empty)} came out empty"
        )
    return split


def save_split(split: Split, directory: str | Path) -> tuple[Path, Path, Path]:
    """Write a partition to disk.

    Args:
        split: The partition to persist.
        directory: Destination directory, created if absent.

    Returns:
        The paths of the training, validation and test files.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, indices in (
        (TRAIN_FILE, split.train),
        (VAL_FILE, split.val),
        (TEST_FILE, split.test),
    ):
        path = directory / name
        path.write_text(json.dumps(indices), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1], paths[2]


def load_split(directory: str | Path) -> Split:
    """Read a partition previously written by :func:`save_split`.

    Args:
        directory: Directory holding the partition files.

    Returns:
        The partition.

    Raises:
        FileNotFoundError: If any partition file is missing.
    """
    directory = Path(directory)
    sides = []
    for name in (TRAIN_FILE, VAL_FILE, TEST_FILE):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"split file not found: {path}")
        sides.append(json.loads(path.read_text(encoding="utf-8")))

    return Split(train=sides[0], val=sides[1], test=sides[2])
