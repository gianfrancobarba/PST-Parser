"""Union of the outcome ledgers of the check runs of one generation session.

With two journals in a session (plan of 18/11/2026, P-04, C-26), the oracle runs
once on the merged core generations (T, H-gold, E on S_gold) and once on the
extra ones (H-auto on S_parse). ``hauto-ledgers`` reads a single ledger holding
T and H-auto under one ``analysis_id``: this module writes it, without changing
any outcome, and refuses mixed endpoints and cells judged in two ledgers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from pstparser.empirical.analysis import OutcomeRecord


def merge_ledgers(
    ledgers: Sequence[Sequence[OutcomeRecord]], analysis_id: str
) -> list[OutcomeRecord]:
    """Concatenate ledgers of one endpoint under ``analysis_id``, keeping every outcome.

    Raises:
        ValueError: If the union is empty, holds more than one endpoint, or a
            cell (task, condition, replicate) appears more than once.
    """
    if not analysis_id.strip():
        raise ValueError("give a non-empty analysis_id")
    records = [record for ledger in ledgers for record in ledger]
    if not records:
        raise ValueError("the union of the ledgers is empty")
    if len({record.endpoint for record in records}) != 1:
        raise ValueError("the ledgers must hold one endpoint")
    cells = Counter((r.task_id, r.condition, r.replicate_id) for r in records)
    repeated = sorted(cell for cell, count in cells.items() if count > 1)
    if repeated:
        raise ValueError(f"cells judged more than once: {repeated[:5]}")
    return [replace(record, analysis_id=analysis_id) for record in records]
