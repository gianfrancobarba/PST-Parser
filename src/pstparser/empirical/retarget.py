"""Retarget parser predictions on the reviewed or frozen gold.

``parse-prompts`` writes, beside each raw prediction, the target built from the
spans of its source file; the D-01 and prefix sources carried the rule drafts.
The parser gate measures F1 against the gold, so before ``score`` the target of
every record is replaced by the gold tree of the given annotations. Nothing
else changes: index, prompt, raw prediction and token counts stay as written,
so ``auto-annotate`` (which never reads the target) gives the same conversion.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Final

from pstparser.data.predictions import PredictionRecord
from pstparser.empirical import annotations as gold_rows
from pstparser.empirical.benchmarks import SourceRecord

RETARGET_VERSION: Final = "retarget-gold-v1"


def retarget_predictions(
    records: Sequence[PredictionRecord],
    tasks: Sequence[str],
    sources: Sequence[SourceRecord],
    gold: Sequence[gold_rows.AnnotationRow],
    *,
    require_status: str,
) -> list[PredictionRecord]:
    """Replace the target of every prediction with the gold tree of its task.

    Args:
        records: Predictions of one ``parse-prompts`` run.
        tasks: The ``tasks`` of its ``id-map.json`` (task of each index).
        sources: Benchmark sources of those tasks.
        gold: Annotations in ``require_status``, covering exactly the parsed tasks.
        require_status: ``reviewed`` or ``frozen`` (final gate).

    Raises:
        ValueError: If an index has no task, a task repeats, the gold does not
            cover the parsed tasks exactly, a row is in another state, or a
            prompt differs from the text of its gold source.
    """
    from pstparser.inference.parse_prompts import gold_target

    if len(set(tasks)) != len(tasks):
        raise ValueError("the id-map repeats a task")
    if sorted(r.index for r in records) != list(range(len(tasks))):
        raise ValueError("the predictions do not hold one record per index of the id-map")
    parsed = set(tasks)
    by_id = {record.task_id: record for record in sources if record.task_id in parsed}
    prompts = gold_rows.canonicalize(
        [by_id[task] for task in tasks if task in by_id],
        list(gold),
        require_status=require_status,
        sample_ids=list(tasks),
    )
    targets = {prompt.task_id: prompt for prompt in prompts}
    if set(targets) != parsed:
        raise ValueError("the gold does not cover the parsed tasks exactly")
    out = []
    for record in records:
        prompt = targets[tasks[record.index]]
        if prompt.text != record.prompt:
            raise ValueError(f"{prompt.task_id}: the text differs from the gold source")
        spans = [dataclasses.asdict(span) for span in prompt.annotations]
        out.append(dataclasses.replace(record, target=gold_target(spans)))
    return out
