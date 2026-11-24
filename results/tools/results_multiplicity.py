"""Multiplicity of the exploratory contrasts: count, expected false readings, Benjamini-Hochberg.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md`` (A8): a sensitivity only, the E1-E4
readings stay those of the frozen and exploratory reports.

The p-values are the primary test of each exploratory paired report: the mid-p
McNemar for one greedy pair per task, the studentized sign-flip otherwise.
Included: the 13 exploratory rows of the frozen paired table (every row but
H1), the reports of G (3), R (4) and S (30). Not included: subgroups, domains
and strata (intervals only, no test), and the sensitivities of the audit.

usage (repository root, .venv):
  python -I results/tools/results_multiplicity.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

VERSION: Final = "results-multiplicity-v1"
Q: Final = 0.05
FROZEN: Final = "outputs/ripresa/e/final-01/analysis"
CONFIRMATORY: Final = ("final-bcb-core", ("T", "H-gold"))
GROUPS: Final = {
    "G": "outputs/ripresa/g/analysis/*/report.json",
    "R": "outputs/ripresa/r/gen/analysis/*/report.json",
    "S": "outputs/ripresa/s/gen/*/eval/analysis/attach/*/*/report.json",
}


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def summary(rows: Sequence[Mapping[str, Any]], q: float = Q) -> dict[str, Any]:
    """Counts of raw and adjusted p-values under the thresholds."""
    p = [row["p"] for row in rows]
    adjusted = benjamini_hochberg(p)
    return {
        "contrasts": len(p),
        "raw_below_0.05": sum(value <= 0.05 for value in p),
        "expected_false_at_0.05": 0.05 * len(p),
        "bh_below_q": sum(value <= q for value in adjusted),
        "q": q,
    }


def _primary_p(report: Mapping[str, Any]) -> float | None:
    for key in ("midp_p", "signflip_p"):
        if report.get(key) is not None:
            return float(report[key])
    return None


def collect(root: Path) -> list[dict[str, Any]]:
    """Every exploratory contrast with its primary p-value."""
    rows = []
    for path in sorted((root / FROZEN).glob("*/report.json")):
        report = json.loads(path.read_text("utf-8"))
        if (report["analysis_id"], tuple(report["conditions"])) == (
            CONFIRMATORY[0],
            CONFIRMATORY[1],
        ):
            continue
        rows.append(
            {"group": "frozen plan", "source": path.relative_to(root).as_posix(), "report": report}
        )
    for group, pattern in GROUPS.items():
        for path in sorted(root.glob(pattern)):
            rows.append(
                {
                    "group": group,
                    "source": path.relative_to(root).as_posix(),
                    "report": json.loads(path.read_text("utf-8")),
                }
            )
    result = []
    for row in rows:
        report = row["report"]
        p = _primary_p(report)
        if p is None:
            raise ValueError(f"{row['source']}: no primary p-value")
        result.append(
            {
                "group": row["group"],
                "source": row["source"],
                "conditions": list(report["conditions"]),
                "delta": report["delta"],
                "p": p,
            }
        )
    return result


def build(root: Path) -> dict[str, Any]:
    """Contrasts, adjusted p-values and the summary per group and overall."""
    rows = collect(root)
    adjusted = benjamini_hochberg([row["p"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["p_bh"] = value
    groups = sorted({row["group"] for row in rows})
    return {
        "version": VERSION,
        "overall": summary(rows),
        "per_group": {g: summary([r for r in rows if r["group"] == g]) for g in groups},
        "contrasts": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``multiplicity.json`` into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "multiplicity.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "overall": result["overall"]}))


if __name__ == "__main__":
    main()
