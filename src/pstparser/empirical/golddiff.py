"""Diff between the rule drafts and the reviewed gold (plan of 18/11/2026, P-07).

A prespecified analysis: how much the review changed the drafts proposed by the
rules. Characters are compared with the renderer's separator policy (the
labels of :func:`pstparser.empirical.agreement.character_labels`), never
counting whitespace. Each draft span is one unit of the rule that proposed it
(``Unit.rule``, ``Unit.kind``): a unit is changed when a character in it has a
new leaf, or when the gold puts a span boundary inside it or removes the one at
its start.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Final

from pstparser.empirical.agreement import character_labels
from pstparser.empirical.annotations import AnnotationRow, proposed_units
from pstparser.empirical.benchmarks import SourceRecord

DIFF_VERSION: Final = "draft-gold-diff-v1"


def _starts(row: AnnotationRow) -> set[int]:
    return {span.start for span in row.spans}


def gold_diff(
    sources: Sequence[SourceRecord],
    drafts: Sequence[AnnotationRow],
    gold: Sequence[AnnotationRow],
    rule: str,
) -> dict[str, Any]:
    """Characters and units changed by the review, by rule, on the tasks of ``gold``.

    ``reviewer_decided_share`` is the share of non-whitespace characters that
    lie in ``judgement`` units, whose leaf the rules leave to the reviewer.

    Raises:
        ValueError: If a gold task has no draft or no source, a row belongs to
            another source, a draft is not the proposal of ``rule``, or a gold
            row is still a draft.
    """
    texts = {source.task_id: source for source in sources}
    by_task = {row.task_id: row for row in drafts}
    totals = Counter[str]()
    by_rule: dict[str, dict[str, Any]] = {}
    transitions = Counter[str]()
    per_task: dict[str, dict[str, Any]] = {}
    for row in sorted(gold, key=lambda item: item.task_id):
        task = row.task_id
        if task not in by_task:
            raise ValueError(f"{task}: a gold task without a draft")
        if task not in texts:
            raise ValueError(f"{task}: a gold task without a source")
        source, draft = texts[task], by_task[task]
        if {row.source_sha256, draft.source_sha256} != {source.source_sha256}:
            raise ValueError(f"{task}: an annotation does not belong to this source")
        if row.status == "draft":
            raise ValueError(f"{task}: the gold row is still a draft")
        units = proposed_units(source, rule)
        proposal = [(unit.path, unit.start, unit.end) for unit in units]
        if draft.annotator != f"rule:{rule}" or proposal != [
            (span.path, span.start, span.end) for span in draft.spans
        ]:
            raise ValueError(f"{task}: the draft is not the proposal of rule {rule}")
        text = source.text
        before = character_labels(text, draft.spans)
        after = character_labels(text, row.spans)
        draft_starts, gold_starts = _starts(draft), _starts(row)
        task_changed = 0
        for unit in units:
            counted = [i for i in range(unit.start, unit.end) if not text[i].isspace()]
            changed = [i for i in counted if before[i] != after[i]]
            boundary_moved = any(
                (i in draft_starts) != (i in gold_starts) for i in range(unit.start, unit.end)
            )
            entry = by_rule.setdefault(
                unit.rule,
                {
                    "kind": unit.kind,
                    "units": 0,
                    "units_changed": 0,
                    "characters": 0,
                    "characters_changed": 0,
                },
            )
            entry["units"] += 1
            entry["units_changed"] += int(bool(changed) or boundary_moved)
            entry["characters"] += len(counted)
            entry["characters_changed"] += len(changed)
            if unit.kind == "judgement":
                totals["reviewer_decided_characters"] += len(counted)
        counted_all = [i for i, character in enumerate(text) if not character.isspace()]
        changed_all = [i for i in counted_all if before[i] != after[i]]
        for i in changed_all:
            transitions[f"{before[i]} -> {after[i]}"] += 1
        task_changed = len(changed_all)
        modified = bool(changed_all) or draft_starts != gold_starts
        totals["characters"] += len(counted_all)
        totals["characters_changed"] += task_changed
        totals["tasks_changed"] += int(modified)
        per_task[task] = {"changed": modified, "characters_changed": task_changed}
    tasks = len(per_task)
    characters = totals["characters"]
    return {
        "diff_version": DIFF_VERSION,
        "rule": rule,
        "tasks": tasks,
        "tasks_changed": totals["tasks_changed"],
        "tasks_changed_share": totals["tasks_changed"] / tasks if tasks else None,
        "characters": characters,
        "characters_changed": totals["characters_changed"],
        "characters_changed_share": (
            totals["characters_changed"] / characters if characters else None
        ),
        "reviewer_decided_characters": totals["reviewer_decided_characters"],
        "reviewer_decided_share": (
            totals["reviewer_decided_characters"] / characters if characters else None
        ),
        "by_rule": dict(sorted(by_rule.items())),
        "transitions": dict(sorted(transitions.items())),
        "per_task": per_task,
    }
