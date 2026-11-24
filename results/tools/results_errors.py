"""Error analysis of the correctness contrasts: failure taxonomy, transitions, flips, review.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md``; descriptive only.

- Failure categories per condition and the baseline -> treatment transitions
  among the discordant pairs, read from the frozen ``taxonomy.json`` of plan §6.
- Flip overlap (part of the concordance, A6): whether the tasks that change
  outcome between T and H-gold are the same tasks that change outcome under the
  control forms (E, F), against the overlap expected if flips were independent
  (|A| |B| / n). A ratio well above 1 marks tasks that flip under any change of
  the prompt surface.
- The tables of the blind qualitative reading of the 40 discordant pairs.

usage (repository root, .venv):
  python -I results/tools/results_errors.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final

VERSION: Final = "results-errors-v1"
DESIGNS: Final = ("bcb-h1", "bcb-h3-s-parse", "bcb-rq3c")
QUALITATIVE: Final = "outputs/ripresa/q/tables.json"


def _frame_module() -> ModuleType:
    path = Path(__file__).with_name("results_frame.py")
    spec = importlib.util.spec_from_file_location("results_frame", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frame = _frame_module()


def discordant(cells: Sequence[Mapping[str, Any]], pair: Sequence[str]) -> set[str]:
    """Tasks whose outcome differs between the two conditions of ``pair``."""
    arms: dict[str, dict[str, str]] = {}
    for cell in cells:
        if cell["condition"] in pair:
            arms.setdefault(cell["task_id"], {})[cell["condition"]] = cell["outcome"]
    return {t for t, a in arms.items() if len(a) == 2 and a[pair[0]] != a[pair[1]]}


def flip_overlap(first: set[str], second: set[str], n: int) -> dict[str, Any]:
    """Tasks flipping in both comparisons, against the count expected if independent."""
    expected = len(first) * len(second) / n
    both = len(first & second)
    return {
        "first": len(first),
        "second": len(second),
        "both": both,
        "expected_if_independent": expected,
        "ratio": both / expected if expected else None,
    }


def build(root: Path) -> dict[str, Any]:
    """Taxonomy, transitions, flip overlap and the qualitative tables."""
    taxonomy = {
        design: json.loads(
            (root / frame.RUN / "exploratory" / design / "taxonomy.json").read_text("utf-8")
        )
        for design in DESIGNS
    }
    records = frame.read_outcomes(root / frame.EVAL / "ledgers/final-bcb.jsonl")
    cells = [
        {"task_id": r.task_id, "condition": r.condition, "outcome": r.outcome} for r in records
    ]
    n = len({c["task_id"] for c in cells})
    flips = {
        name: discordant(cells, pair)
        for name, pair in (
            ("T-H-gold", ("T", "H-gold")),
            ("T-E", ("T", "E")),
            ("T-F", ("T", "F")),
            ("E-H-gold", ("E", "H-gold")),
            ("F-H-gold", ("F", "H-gold")),
        )
    }
    overlap = {
        "T-H-gold with T-E": flip_overlap(flips["T-H-gold"], flips["T-E"], n),
        "T-H-gold with T-F": flip_overlap(flips["T-H-gold"], flips["T-F"], n),
        "T-E with T-F": flip_overlap(flips["T-E"], flips["T-F"], n),
        "T-H-gold in T-E or T-F": {
            "tasks": len(flips["T-H-gold"]),
            "also_flip_under_a_control": len(flips["T-H-gold"] & (flips["T-E"] | flips["T-F"])),
        },
    }
    return {
        "version": VERSION,
        "taxonomy": {
            design: {
                "per_condition": value["per_condition"],
                "transitions": value.get("transitions"),
            }
            for design, value in taxonomy.items()
        },
        "flips": {name: len(tasks) for name, tasks in flips.items()},
        "flip_overlap": overlap,
        "qualitative": json.loads((root / QUALITATIVE).read_text("utf-8"))["tables"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``errors.json`` into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "errors.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
