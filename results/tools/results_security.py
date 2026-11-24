"""Reading tables of the security contrasts (exploratory): frozen measures, Δ_I, CWE, S, audit.

Written after the frozen results were seen, under the row "Results analysis
completion" of ``protocol/deviations.md`` (A2, A9); descriptive except where a
frozen estimate is re-read.

- The five frozen security reports: Δ of the share of state N, CR1 at 95% and
  90%, studentized sign-flip p, E1-E4 at 3, 5 and 10 pp, Δ_fam, leave one
  family out, family bootstrap and the bounds for U.
- (A2) Δ_I, the share of unusable programs, H-gold - T: the frozen
  ``analyze_paired`` on the same design and ledger with the state relabelled
  (I counts as the event, every other state as its absence), so the estimator,
  the families, the seed and the intervals are the frozen ones. Sampled
  general measure; the greedy measure is descriptive.
- Alerts per 100 lines of code: the frozen ``loc_tables``.
- States per CWE and condition on the sampled general measure (counts only).
- Analysis S (r3 and r4): every report re-read the same way.
- The oracle audit sensitivities S-ORA-1..4 and the blind review, as stored.

usage (repository root, .venv):
  python -I results/tools/results_security.py --root . --out NEW_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from pstparser.empirical.analysis import OutcomeRecord, analyze_paired, read_outcomes
from pstparser.empirical.inference import cluster_interval_at, four_way_reading
from pstparser.empirical.tables import loc_tables

VERSION: Final = "results-security-v1"
MARGINS: Final = {"3pp": 0.03, "5pp": 0.05, "10pp": 0.10}
STATES: Final = ("V", "N", "I", "U")
FROZEN: Final = (
    "lse-general",
    "lse-greedy-general",
    "lse-general-python",
    "lse-targeted",
    "lse-targeted-fp",
)
DETAILS: Final = {
    "sampled-general": "outputs/ripresa/d/eval/codeql-final-lse-sampled-general/details.jsonl",
    "greedy-general": "outputs/ripresa/d/eval/codeql-final-lse-greedy-general/details.jsonl",
}
S_ROOT: Final = "outputs/ripresa/s/gen"
AUDIT: Final = "outputs/ripresa/audit/sora"
BLIND: Final = "outputs/ripresa/e/final-01/review/blind-score.json"


def _frame_module() -> ModuleType:
    path = Path(__file__).with_name("results_frame.py")
    spec = importlib.util.spec_from_file_location("results_frame", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frame = _frame_module()


def unusable_as_endpoint(record: OutcomeRecord) -> OutcomeRecord:
    """State I becomes the counted state (N); every other state becomes V."""
    return replace(record, outcome="N" if record.outcome == "I" else "V")


def security_row(report: Mapping[str, Any]) -> dict[str, Any]:
    """Δ, CR1 at 95% and 90%, the sign-flip p and the readings of a security report."""
    ci90 = (
        list(cluster_interval_at(report["cr1_ci"], report["cr1_se"], report["cr1_df"], 0.90))
        if report.get("cr1_ci") is not None
        else None
    )
    p = report.get("signflip_p")
    readings = (
        {name: four_way_reading(p, ci90, alpha=0.05, margin=m) for name, m in MARGINS.items()}
        if p is not None and ci90 is not None
        else {}
    )
    family = report.get("family_sensitivity") or {}
    return {
        "conditions": list(report["conditions"]),
        "prompts": report["included_tasks"],
        "delta": report["delta"],
        "cr1_ci": report.get("cr1_ci"),
        "cr1_ci_90": ci90,
        "signflip_p": p,
        "reading_3pp": readings.get("3pp"),
        "reading_5pp": readings.get("5pp"),
        "reading_10pp": readings.get("10pp"),
        "outcome_counts": report.get("outcome_counts"),
        "bootstrap_ci": report.get("bootstrap_ci"),
        "delta_fam": family.get("delta_fam"),
        "delta_fam_ci": family.get("delta_fam_ci"),
        "lofo": [family.get("lofo_min"), family.get("lofo_max")] if family else None,
        "u_bounds": {
            key: value for key, value in (report.get("sensitivity") or {}).items() if "U" in key
        },
    }


def cwe(family_id: str) -> str:
    """The CWE of a LLMSecEval family (its numeric prefix)."""
    return f"CWE-{family_id.split('-', 1)[0]}"


def cwe_table(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Programs and states per CWE and condition."""
    groups: dict[tuple[str, str], Counter[str]] = {}
    for row in details:
        groups.setdefault((cwe(row["family_id"]), row["condition"]), Counter())[row["state"]] += 1
    rows = []
    for (name, condition), states in sorted(groups.items()):
        total = sum(states.values())
        rows.append(
            {
                "cwe": name,
                "condition": condition,
                "programs": total,
                **{state: states.get(state, 0) for state in STATES},
                "share_N": states.get("N", 0) / total,
            }
        )
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_bytes().decode("utf-8").split("\n") if x.strip()]


def unusable_share(root: Path, design_name: str) -> dict[str, Any]:
    """(A2) Δ_I by the frozen estimator on the relabelled ledger."""
    design = frame.load_design(
        root / frame.TREE / "protocol/analysis/designs" / design_name / "design.json"
    )
    ledger = root / frame.SOURCES[design_name]
    records = [unusable_as_endpoint(r) for r in read_outcomes(ledger)]
    report = analyze_paired(design, records, ledger_conditions=frame.ledger_conditions(ledger))
    return security_row(json.loads(json.dumps(report.as_dict())))


def _audit(root: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for path in sorted((root / AUDIT).glob("ora*.json")):
        payload = json.loads(path.read_text("utf-8"))
        for contrast, value in payload.items():
            if not isinstance(value, Mapping):
                continue
            for variant, result in value.items():
                if isinstance(result, Mapping) and "reading" in result:
                    found[f"{path.stem}/{contrast}/{variant}"] = {
                        key: result.get(key) for key in ("delta", "interval_90", "p", "reading")
                    }
    return found


def build(root: Path) -> dict[str, Any]:
    """Every security table."""
    frozen = {
        name: security_row(
            json.loads((root / frame.RUN / "analysis" / name / "report.json").read_text("utf-8"))
        )
        for name in FROZEN
    }
    unusable = {
        "sampled-general": unusable_share(root, "lse-general"),
        "greedy-general (descriptive)": unusable_share(root, "lse-greedy-general"),
    }
    loc = {}
    for name, relative in DETAILS.items():
        table = loc_tables(_jsonl(root / relative))[0]
        loc[name] = [dict(zip(table.columns, row, strict=True)) for row in table.rows]
    s_rows = []
    for parser in ("r3", "r4"):
        base = root / S_ROOT / parser / "eval/analysis/attach"
        for path in sorted(base.glob("*/*/report.json")):
            s_rows.append(
                {
                    "parser": parser,
                    "measure": path.parent.parent.name,
                    "contrast": path.parent.name,
                    **security_row(json.loads(path.read_text("utf-8"))),
                }
            )
    return {
        "version": VERSION,
        "frozen": frozen,
        "unusable_share": unusable,
        "alerts_per_100_loc": loc,
        "cwe_sampled_general": cwe_table(_jsonl(root / DETAILS["sampled-general"])),
        "s": s_rows,
        "audit": _audit(root),
        "blind_review": json.loads((root / BLIND).read_text("utf-8")),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``security.json`` into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    result = build(args.root)
    args.out.mkdir(parents=True)
    with (args.out / "security.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
