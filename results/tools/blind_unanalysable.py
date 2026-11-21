"""The reviewer's judgment on the unanalysable cases of the blind review (deviation of 21/11).

Written after the freeze (394c893) and before any judgment of the blind review
exists; it lives in results/ because the freeze admits unlisted files only there.
Exploratory and descriptive only: counts and exact intervals, no test.

The cases with targeted state U are those CodeQL could not analyse; every one of
them is in the blind review. The frozen ``blind-score`` does not use the
reviewer's ``scenario_vulnerable`` on them. Here, per condition: cases, yes, no,
undecided (empty), the share of yes among the decided cases and its 95%
Clopper-Pearson interval.

usage (repository root, .venv):
  python -I results/tools/blind_unanalysable.py SHEET_COMPLETED KEY --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pstparser.empirical.auto import clopper_pearson

VERSION = "blind-unanalysable-v1"


def _cell(values: list[bool | None]) -> dict[str, Any]:
    yes = sum(1 for v in values if v is True)
    no = sum(1 for v in values if v is False)
    decided = yes + no
    return {
        "cases": len(values),
        "yes": yes,
        "no": no,
        "undecided": len(values) - decided,
        "share_yes": yes / decided if decided else None,
        "share_yes_ci95": list(clopper_pearson(yes, decided)) if decided else None,
    }


def count_unanalysable(
    sheet: Sequence[Mapping[str, Any]], key: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Count ``scenario_vulnerable`` on the cases whose targeted state is U.

    Raises:
        ValueError: If the sheet and the key do not hold the same cases.
    """
    rows = {row["case_id"]: row for row in sheet}
    if len(rows) != len(sheet) or set(rows) != {case["case_id"] for case in key}:
        raise ValueError("the sheet does not match the key")
    per_condition: dict[str, list[bool | None]] = {}
    for case in key:
        if case["states"].get("targeted") == "U":
            value = rows[case["case_id"]]["scenario_vulnerable"]
            per_condition.setdefault(case["condition"], []).append(value)
    return {
        "format": VERSION,
        "per_condition": {c: _cell(v) for c, v in sorted(per_condition.items())},
        "total": _cell([v for values in per_condition.values() for v in values]),
    }


def _lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    """Write the report to a new file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", type=Path)
    parser.add_argument("key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = count_unanalysable(_lines(args.sheet), _lines(args.key))
    report["inputs"] = {
        "sheet_sha256": hashlib.sha256(args.sheet.read_bytes()).hexdigest(),
        "key_sha256": hashlib.sha256(args.key.read_bytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["total"]), args.out)


if __name__ == "__main__":
    main()
