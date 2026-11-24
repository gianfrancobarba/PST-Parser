"""Unified table of the H-auto contrasts: parser r3 and r4, strict and attach conversion.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md``; it re-reads the reports of the
frozen analysis and of the exploratory analyses G and R (each computed by the
frozen ``analyze_paired``) and reads them E1-E4 at 3, 5 and 10 pp as the
frozen tables do. It also gathers the decompositions
Δ_auto = Δ_gold + Δ_par by parser quality. No new estimate.

usage (repository root, .venv):
  python -I results/tools/results_hauto.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pstparser.empirical.inference import four_way_reading

VERSION: Final = "results-hauto-v1"
MARGINS: Final = {"3pp": 0.03, "5pp": 0.05, "10pp": 0.10}
E: Final = "outputs/ripresa/e/final-01/analysis"
G: Final = "outputs/ripresa/g/analysis"
R: Final = "outputs/ripresa/r/gen/analysis"

# (identifier, parser, conversion, set, analysis level, report path)
REPORTS: Final = (
    (
        "r3-strict-end-to-end",
        "r3",
        "strict",
        "600, failures as fail",
        "plan",
        f"{E}/bcb-h3-end-to-end",
    ),
    (
        "r3-strict-s-parse",
        "r3",
        "strict",
        "506 converted (S_parse)",
        "plan (H3 descriptive)",
        f"{E}/bcb-h3-s-parse",
    ),
    ("r3-strict-vs-gold", "r3", "strict", "506, against H-gold", "plan (RQ3c)", f"{E}/bcb-rq3c"),
    (
        "r3-attach-end-to-end",
        "r3",
        "attach",
        "600, failures as fail",
        "G (post hoc)",
        f"{G}/end-to-end",
    ),
    ("r3-attach-converted", "r3", "attach", "593 converted", "G (post hoc)", f"{G}/converted"),
    (
        "r3-attach-recovered",
        "r3",
        "attach",
        "87 recovered by attach",
        "G (post hoc)",
        f"{G}/recovered",
    ),
    (
        "r4-attach-end-to-end",
        "r4",
        "attach",
        "600, failures as fail",
        "R (post hoc)",
        f"{R}/a-end-to-end-attach",
    ),
    (
        "r4-strict-end-to-end",
        "r4",
        "strict",
        "600, failures as fail",
        "R (post hoc)",
        f"{R}/a-end-to-end-strict",
    ),
    ("r4-attach-converted", "r4", "attach", "597 converted", "R (post hoc)", f"{R}/b-converted"),
    (
        "r4-attach-vs-gold",
        "r4",
        "attach",
        "597, against H-gold",
        "R (post hoc)",
        f"{R}/c-converted-vs-gold",
    ),
)
DECOMPOSITIONS: Final = {
    "r3-strict": "outputs/ripresa/e/final-01/exploratory/hauto-decomposition/decomposition.json",
    "r4-attach": "outputs/ripresa/r/gen/exploratory/hauto-decomposition-attach/decomposition.json",
    "r4-strict": "outputs/ripresa/r/gen/exploratory/hauto-decomposition-strict/decomposition.json",
}


def row(report: Mapping[str, Any]) -> dict[str, Any]:
    """n, b, c, Δ, intervals and the E1-E4 readings of a paired greedy report."""
    table = report["mcnemar_table"]
    ci90 = report["tango_ci_90"]
    readable = report["midp_p"] is not None and ci90 is not None
    return {
        "conditions": list(report["conditions"]),
        "n": report["included_tasks"],
        "b": table["baseline_only"],
        "c": table["treatment_only"],
        "delta": report["delta"],
        "tango_ci": report["tango_ci"],
        "tango_ci_90": ci90,
        "midp_p": report["midp_p"],
        "reading": (
            {
                name: four_way_reading(report["midp_p"], ci90, alpha=0.05, margin=margin)
                for name, margin in MARGINS.items()
            }
            if readable
            else None
        ),
    }


def decomposition_row(grouping: str, level: Mapping[str, Any]) -> dict[str, Any]:
    """One level of a decomposition: tasks and the three contrasts."""
    contrasts = level["contrasts"]
    return {
        "grouping": grouping,
        "level": level["level"],
        "tasks": level["tasks"],
        "reported": level["reported"],
        "f1_range": level.get("f1_range"),
        **{f"{key}": contrasts[key]["delta"] for key in ("delta_auto", "delta_gold", "delta_par")},
        **{
            f"{key}_ci95": contrasts[key]["ci95"]
            for key in ("delta_auto", "delta_gold", "delta_par")
        },
        "counts": {
            key: contrasts[key]["counts"] for key in ("delta_auto", "delta_gold", "delta_par")
        },
    }


def build(root: Path) -> dict[str, Any]:
    """The unified table and the decompositions."""
    rows = []
    for identifier, parser, conversion, evaluation_set, level, folder in REPORTS:
        report = json.loads((root / folder / "report.json").read_text("utf-8"))
        rows.append(
            {
                "id": identifier,
                "parser": parser,
                "conversion": conversion,
                "set": evaluation_set,
                "level": level,
                "source": f"{folder}/report.json",
                **row(report),
            }
        )
    decompositions = {}
    for name, relative in DECOMPOSITIONS.items():
        payload = json.loads((root / relative).read_text("utf-8"))
        decompositions[name] = [
            decomposition_row(grouping, level)
            for grouping in ("all", "validity", "strict_reconstruction", "f1_tercile_s_parse")
            if grouping in payload
            for level in payload[grouping]["levels"]
        ]
    return {"version": VERSION, "contrasts": rows, "decompositions": decompositions}


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``hauto.json`` into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "hauto.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
