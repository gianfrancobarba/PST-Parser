"""Seeding of every source of randomness touched by the pipeline."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedRecord:
    """The seeds that were applied, for inclusion in a run manifest.

    Attributes:
        global_seed: Value requested by the caller.
        applied: Seed applied to each library, keyed by library name.
        unavailable: Libraries that were not importable and therefore not
            seeded.
    """

    global_seed: int
    applied: dict[str, int] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the record."""
        return {
            "global_seed": self.global_seed,
            "applied": dict(self.applied),
            "unavailable": list(self.unavailable),
        }


def seed_everything(seed: int) -> SeedRecord:
    """Seed the standard library, NumPy and torch when present.

    ``PYTHONHASHSEED`` is exported for the benefit of subprocesses; the hash
    randomisation of the current interpreter is fixed at start-up and cannot be
    changed here.

    Args:
        seed: Value applied to every generator.

    Returns:
        A record of what was seeded.
    """
    applied: dict[str, int] = {}
    unavailable: list[str] = []

    os.environ["PYTHONHASHSEED"] = str(seed)
    applied["pythonhashseed"] = seed

    random.seed(seed)
    applied["random"] = seed

    np.random.seed(seed)
    applied["numpy"] = seed

    try:
        import torch
    except ImportError:
        unavailable.append("torch")
    else:
        torch.manual_seed(seed)
        applied["torch"] = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            applied["torch.cuda"] = seed

    try:
        import transformers
    except ImportError:
        unavailable.append("transformers")
    else:
        transformers.set_seed(seed)
        applied["transformers"] = seed

    return SeedRecord(global_seed=seed, applied=applied, unavailable=unavailable)
