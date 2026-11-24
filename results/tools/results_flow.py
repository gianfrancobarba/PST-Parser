"""Sample flow and data quality of the study: from the planned requests to the outcomes.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md``; descriptive only, no estimate.

Tables:
- correctness flow per condition: planned tasks, parser failures (H-auto),
  generated answers, truncated answers, answers with no code, outcome reasons,
  passes, flaky outcomes;
- security flow per measure and condition: programs, extraction status,
  programs analysed by CodeQL, usable programs, states V/N/I/U;
- tokens per journal and condition (input and output), finish reasons;
- the T variants: the prompt equals the source text, and the journal sent it;
- determinism: the frozen comparisons of the regenerated requests.

usage (repository root, .venv):
  python -I results/tools/results_flow.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

VERSION: Final = "results-flow-v1"
CONDITIONS: Final = ("T", "E", "F", "H-gold", "H-auto")
STATES: Final = ("V", "N", "I", "U")
EVAL: Final = "outputs/ripresa/d/eval"
RUN: Final = "outputs/ripresa/e/final-01"
SECURITY: Final = {
    "greedy-general": f"{EVAL}/codeql-final-lse-greedy-general/details.jsonl",
    "sampled-general": f"{EVAL}/codeql-final-lse-sampled-general/details.jsonl",
    "sampled-targeted": f"{EVAL}/codeql-final-lse-sampled-targeted/details.jsonl",
    "sampled-targeted-fp": f"{EVAL}/codeql-final-lse-sampled-targeted-fp/details.jsonl",
}
JOURNALS: Final = ("bcb-hauto", "bcb-rest", "lse-greedy", "lse-sampled")


def _key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (row["task_id"], row["condition"], row["replicate_id"])


def correctness_flow(
    ledger: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
    end_to_end: Sequence[Mapping[str, Any]],
    planned: int,
) -> list[dict[str, Any]]:
    """One row per condition, from the planned tasks to the outcomes.

    Raises:
        ValueError: If a detail row disagrees with the ledger or has no ledger row.
    """
    outcomes = {_key(row): (row["outcome"], row["reason"]) for row in ledger}
    by_key = {_key(row): row for row in details}
    if set(by_key) != set(outcomes) or any(
        (by_key[key]["outcome"], by_key[key]["reason"]) != value for key, value in outcomes.items()
    ):
        raise ValueError("the harness details disagree with the ledger")
    parser_failures = Counter(
        row["reason"].removeprefix("parser_failure:")
        for row in end_to_end
        if row["condition"] == "H-auto" and row["reason"].startswith("parser_failure:")
    )
    rows = []
    for condition in CONDITIONS:
        cells = [row for row in details if row["condition"] == condition]
        if not cells:
            continue
        reasons = Counter(row["reason"] for row in cells)
        rows.append(
            {
                "condition": condition,
                "planned": planned,
                "parser_failures": dict(parser_failures) if condition == "H-auto" else {},
                "generated": len(cells),
                "truncated": sum(row["finish_reason"] != "eos" for row in cells),
                "no_code": reasons.get("no_code", 0),
                "pass": sum(row["outcome"] == "pass" for row in cells),
                "flaky": sum(bool(row["flaky"]) for row in cells),
                "reasons": dict(reasons),
            }
        )
    return rows


def security_flow(details: Sequence[Mapping[str, Any]], measure: str) -> list[dict[str, Any]]:
    """One row per condition: programs, extraction, analysis, usability and states."""
    rows = []
    for condition in CONDITIONS:
        cells = [row for row in details if row["condition"] == condition]
        if not cells:
            continue
        states = Counter(row["state"] for row in cells)
        rows.append(
            {
                "measure": measure,
                "condition": condition,
                "programs": len(cells),
                "extraction": dict(Counter(row["extraction_status"] for row in cells)),
                "analysed": sum(bool(row["analysed"]) for row in cells),
                "usable": sum(bool(row["usable"]) for row in cells),
                "states": {state: states.get(state, 0) for state in STATES},
            }
        )
    return rows


def token_summary(journal: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Input and output tokens and finish reasons per condition."""
    rows = []
    for condition in CONDITIONS:
        cells = [row for row in journal if row["condition"] == condition]
        if not cells:
            continue
        row: dict[str, Any] = {"condition": condition, "n": len(cells)}
        for side in ("input", "output"):
            values = [cell[f"{side}_tokens"] for cell in cells]
            row[f"{side}_mean"] = statistics.fmean(values)
            row[f"{side}_median"] = statistics.median(values)
            row[f"{side}_max"] = max(values)
        row["finish"] = dict(Counter(cell["finish_reason"] for cell in cells))
        rows.append(row)
    return rows


def t_variants_check(
    variants: Sequence[Mapping[str, Any]], journal: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """T prompts equal to their source text, and journal texts equal to the frozen variants."""
    t_rows = {row["task_id"]: row for row in variants if row["condition"] == "T"}
    sent = [row for row in journal if row["condition"] == "T"]
    return {
        "t_variants": len(t_rows),
        "prompt_equals_source": sum(
            row["prompt_hash"] == row["source_hash"] for row in t_rows.values()
        ),
        "journal_rows": len(sent),
        "journal_text_equals_variant": sum(
            row["task_id"] in t_rows and row["text"] == t_rows[row["task_id"]]["text"]
            for row in sent
        ),
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_bytes().decode("utf-8").split("\n") if x.strip()]


def build(root: Path) -> dict[str, Any]:
    """Every table of the flow, read from the collected and evaluated files."""
    ledger = _jsonl(root / EVAL / "ledgers/final-bcb.jsonl")
    details = _jsonl(root / RUN / "evaluation/bcb-details.jsonl")
    end_to_end = _jsonl(root / EVAL / "ledgers/hauto/outcomes-end-to-end.jsonl")
    planned = len({row["task_id"] for row in ledger})
    security = []
    for measure, relative in SECURITY.items():
        security += security_flow(_jsonl(root / relative), measure)
    journals = {
        name: _jsonl(root / RUN / "generation" / name / "generations.jsonl") for name in JOURNALS
    }
    tokens = {name: token_summary(rows) for name, rows in journals.items()}
    variants = []
    for name in ("bcb-hauto", "bcb-rest", "lse"):
        variants += _jsonl(root / RUN / "inputs" / f"{name}.variants.jsonl")
    bcb_variants = [row for row in variants if str(row["task_id"]).startswith("BigCodeBench/")]
    lse_variants = [row for row in variants if not str(row["task_id"]).startswith("BigCodeBench/")]
    determinism = {
        path.parent.name: json.loads(path.read_text("utf-8"))
        for path in sorted((root / EVAL / "determinism").glob("*/determinism.json"))
    }
    return {
        "version": VERSION,
        "correctness": correctness_flow(ledger, details, end_to_end, planned),
        "security": security,
        "tokens": tokens,
        "t_variants": {
            "bcb": t_variants_check(bcb_variants, journals["bcb-hauto"] + journals["bcb-rest"]),
            "lse": t_variants_check(lse_variants, journals["lse-greedy"] + journals["lse-sampled"]),
        },
        "determinism": {
            name: {key: value[key] for key in ("compared", "identical", "differing")}
            for name, value in determinism.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``flow.json`` into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "flow.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
