"""Longitudinal table of the parser versions: starting work, reference run, r3, r4.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md`` (figures from existing estimates);
it computes no new score. Every number is read from a versioned or collected
score file, or transcribed from a document named in ``DOCUMENTED``.

Rows:
- ``scorziello``: the starting work, 109 test prompts (tables 6.1-6.3 of its
  thesis); validity there is ``json.loads`` only. Transcribed.
- ``august-original`` and ``august-definitions``: the historical reference run
  (``results/``), 88 prompts, the same predictions scored by the starting
  computation and by the published definitions.
- ``r3-devgpt``: r3 on its 100 test prompts. Transcribed: the run directory of
  the training is not part of this repository.
- ``r3-bcb``, ``r4-bcb``: the 600-task frozen gold of BigCodeBench (r3 is the
  frozen score of the parser gate).
- ``r3-lse``, ``r4-lse``: the 100 LLMSecEval prompts of the study.

The rows are not paired: prompts, corpora and metric definitions differ
between the first four. Macro F1 is the gate definition (mean of the leaf F1
over the leaves with support) and is given only where the study reports it.

usage (repository root, .venv):
  python -I results/tools/results_parser.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

VERSION: Final = "results-parser-v1"
TOLERANCE: Final = 1e-12
LEAVES: Final = (
    "main_instruction",
    "context.role",
    "context.format",
    "context.constraints",
    "context.data",
    "examples",
    "reasoning.influence",
    "reasoning.reasoning_examples",
    "reasoning.paths",
)

DOCUMENTED: Final[tuple[dict[str, Any], ...]] = (
    {
        "id": "scorziello",
        "label": "Starting work (Scorziello)",
        "set": "109 test prompts (equal to the validation set)",
        "n": 109,
        "validity": 1.0,
        "validity_rule": "json.loads only",
        "coverage": 0.9991,
        "hallucination": 0.0018,
        "f1": {
            "examples": 1.0,
            "context.data": 0.8805,
            "main_instruction": 0.8662,
            "context.role": 0.7857,
            "context.format": 0.5928,
            "context.constraints": 0.3453,
        },
        "source": "G. Scorziello, master thesis, tables 6.1, 6.2 and 6.3",
    },
    {
        "id": "r3-devgpt",
        "label": "r3, test split",
        "set": "100 test prompts of the parser corpus",
        "n": 100,
        "validity": 1.0,
        "validity_rule": "parse and schema",
        "coverage": 0.9993,
        "hallucination": 0.0,
        "f1": {
            "context.role": 1.0,
            "examples": 0.9726,
            "context.data": 0.9538,
            "reasoning.paths": 0.9421,
            "main_instruction": 0.9340,
            "context.format": 0.9242,
            "reasoning.influence": 0.9084,
            "context.constraints": 0.8401,
            "reasoning.reasoning_examples": 0.6917,
        },
        "source": "score of the r3 training run on its test split (run directory not versioned)",
    },
)

# id -> (label, evaluation set, score file relative to the repository root)
LOADED: Final = {
    "august-original": (
        "Reference run, starting computation",
        "88 prompts",
        "results/results.json",
    ),
    "august-definitions": (
        "Reference run, published definitions",
        "88 prompts",
        "results/conformant/results.json",
    ),
    "r3-bcb": (
        "r3, BigCodeBench gold",
        "600 tasks, frozen gold",
        "protocol/hauto/prefix/score-results.json",
    ),
    "r4-bcb": (
        "r4, BigCodeBench gold",
        "600 tasks, frozen gold",
        "outputs/ripresa/r/eval/r4/score/results.json",
    ),
    "r3-lse": (
        "r3, LLMSecEval gold",
        "100 prompts",
        "outputs/ripresa/s/eval/r3/score/results.json",
    ),
    "r4-lse": (
        "r4, LLMSecEval gold",
        "100 prompts",
        "outputs/ripresa/s/eval/r4/score/results.json",
    ),
}
# Reports that state a macro F1, checked against the leaves of the score.
MACRO_REPORTS: Final = {
    "r3-bcb": ("outputs/ripresa/r/eval/r4/report.json", "r3_macro_f1"),
    "r4-bcb": ("outputs/ripresa/r/eval/r4/report.json", "r4.macro_f1"),
    "r3-lse": ("outputs/ripresa/s/eval/r3/report.json", "macro_f1"),
    "r4-lse": ("outputs/ripresa/s/eval/r4/report.json", "macro_f1"),
}
R3_REPRODUCTION: Final = "outputs/ripresa/r/eval/r3-reproduction/score/results.json"


def _leaf_name(name: str) -> str:
    name = name.removeprefix("prompt.")
    return "context.constraints" if name == "context.constrains" else name


def load_score(path: Path) -> dict[str, Any]:
    """A score file as a row: counts, validity, coverage, hallucination, leaves."""
    payload = json.loads(path.read_text("utf-8"))
    leaves: dict[str, dict[str, Any]] = {}
    for name, value in payload["field_f1"].items():
        if isinstance(value, Mapping):
            leaf = {
                key: value.get(key) for key in ("f1", "precision", "recall", "support", "predicted")
            }
        else:
            leaf = {
                "f1": value,
                "precision": None,
                "recall": None,
                "support": None,
                "predicted": None,
            }
        leaves[_leaf_name(name)] = leaf
    return {
        "total": payload["total"],
        "valid": payload["valid"],
        "validity": payload["json_validity_rate"],
        "coverage": payload["mean_coverage_score"],
        "hallucination": payload["mean_hallucination_rate"],
        "leaves": leaves,
    }


def row_f1(
    leaves: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """F1 per leaf, empty where the gold has no token; predicted tokens of those leaves.

    A leaf without gold support has no F1 to report (a score of 0 there only says
    the parser placed tokens where the gold has none): its F1 is left empty and
    its predicted tokens are reported apart.
    """
    f1: dict[str, float | None] = {}
    unsupported: dict[str, int] = {}
    for name, leaf in leaves.items():
        if leaf.get("support") == 0:
            f1[name] = None
            if leaf.get("predicted"):
                unsupported[name] = leaf["predicted"]
        else:
            f1[name] = leaf["f1"]
    return f1, unsupported


def macro_f1(leaves: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Mean of the leaf F1 over the leaves with support (the gate definition)."""
    values = [
        leaf["f1"] for leaf in leaves.values() if leaf.get("support") and leaf["f1"] is not None
    ]
    return sum(values) / len(values) if values else None


def same_leaves(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Differences between the leaves of two scores."""
    diffs = []
    for name in sorted(set(left["leaves"]) | set(right["leaves"])):
        a, b = left["leaves"].get(name), right["leaves"].get(name)
        if a is None or b is None:
            diffs.append(f"{name}: only in one score")
            continue
        for key in ("f1", "precision", "recall", "support"):
            x, y = a[key], b[key]
            if (x is None) != (y is None) or (x is not None and abs(x - y) > TOLERANCE):
                diffs.append(f"{name}: {key} {x} != {y}")
    return diffs


def leaf_table(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    """One row per version, one column per leaf of the taxonomy (F1)."""
    columns = ["id", *LEAVES]
    return columns, [[row["id"], *(row["f1"].get(leaf) for leaf in LEAVES)] for row in rows]


def _dotted(payload: Mapping[str, Any], key: str) -> Any:
    value: Any = payload
    for part in key.split("."):
        value = value[part]
    return value


def build(root: Path) -> dict[str, Any]:
    """All rows, with the checks of the loaded scores against the study reports."""
    rows: list[dict[str, Any]] = [dict(row) for row in DOCUMENTED]
    checks: dict[str, Any] = {}
    for identifier, (label, evaluation_set, relative) in LOADED.items():
        score = load_score(root / relative)
        row = {
            "id": identifier,
            "label": label,
            "set": evaluation_set,
            "n": score["total"],
            "valid": score["valid"],
            "validity": score["validity"],
            "validity_rule": "starting computation"
            if identifier == "august-original"
            else "parse and schema",
            "coverage": score["coverage"],
            "hallucination": score["hallucination"],
            "f1": row_f1(score["leaves"])[0],
            "predicted_without_support": row_f1(score["leaves"])[1],
            "support": {name: leaf["support"] for name, leaf in score["leaves"].items()},
            "source": relative,
        }
        if identifier in MACRO_REPORTS:
            report_path, key = MACRO_REPORTS[identifier]
            stated = _dotted(json.loads((root / report_path).read_text("utf-8")), key)
            computed = macro_f1(score["leaves"])
            row["macro_f1"] = stated
            checks[f"{identifier} macro F1"] = {
                "stated": stated,
                "from_leaves": computed,
                "identical": computed is not None and abs(computed - stated) <= TOLERANCE,
            }
        rows.append(row)
    reproduction = same_leaves(
        load_score(root / LOADED["r3-bcb"][2]), load_score(root / R3_REPRODUCTION)
    )
    checks["r3 reproduction equals the frozen gate score"] = {
        "identical": not reproduction,
        "differences": reproduction,
    }
    order = [
        "scorziello",
        "august-original",
        "august-definitions",
        "r3-devgpt",
        "r3-bcb",
        "r4-bcb",
        "r3-lse",
        "r4-lse",
    ]
    rows.sort(key=lambda row: order.index(row["id"]))
    return {"version": VERSION, "rows": rows, "checks": checks}


def _figure(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = "osD^v<>Px"
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, row in enumerate(rows):
        points = [
            (row["f1"][leaf], LEAVES.index(leaf))
            for leaf in LEAVES
            if row["f1"].get(leaf) is not None
        ]
        if not points:
            continue
        xs, ys = zip(*points, strict=True)
        offset = (index - (len(rows) - 1) / 2) * 0.08
        ax.scatter(
            xs,
            [y + offset for y in ys],
            marker=markers[index % len(markers)],
            label=row["label"],
            s=28,
        )
    ax.set_yticks(range(len(LEAVES)), LEAVES)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("F1 per leaf")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the table (JSON and CSV of the leaf F1) and the figure into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "parser-versions.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    columns, table = leaf_table(result["rows"])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([["" if cell is None else cell for cell in row] for row in table])
    with (args.out / "parser-leaf-f1.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(buffer.getvalue())
    _figure(result["rows"], args.out / "parser-leaf-f1.png")
    ok = all(check["identical"] for check in result["checks"].values())
    print(json.dumps({"out": str(args.out), "checks_identical": ok}))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
