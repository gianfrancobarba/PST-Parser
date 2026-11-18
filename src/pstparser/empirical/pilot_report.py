"""Non-directional pilot report: technical go/no-go criteria, never an effect estimate.

The pilot measures costs, faults and technical parameters (criteria G1-G14 of
the pilot plan). It must not show which condition does better: per condition it
reports only technical states (truncation, extraction, responses that treat the
prompt as data, input tokens), never a success rate. The only paired quantity
is the number of discordant T/H-gold pairs, b + c, without b - c.

Truncations, empty sanitised programs, ``json_response`` and ``prompt_as_data``
are failures by construction: counted per condition next to b + c they would
bound the sign of H-gold - T. Per condition the report says only whether they
are present; their counts are pooled over all conditions (R-STA-05).

Some criteria cannot be read from a journal and its evaluation (calibration of
the harness, annotation time, the parser on the pilot prompts); they are listed
under ``not_computed_here`` with the artefact where they are read.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pstparser.empirical.analysis import OutcomeRecord
from pstparser.empirical.contracts import CONDITIONS

REPORT_VERSION: Final = "pilot-report-v2"
#: Fragments no key of the report may contain (case-insensitive).
FORBIDDEN_KEY_FRAGMENTS: Final = (
    "delta",
    "diff",
    "baseline_only",
    "treatment_only",
    "pass",
    "success",
    "effect",
)
TRUNCATION_THRESHOLD: Final = 0.02
EXTRACTION_THRESHOLD: Final = 0.15
AGREEMENT_THRESHOLD: Final = 0.99
CODEQL_THRESHOLD: Final = 0.95
BUDGET_MARGIN: Final = 1.25
DIAGNOSTIC_POLICY: Final = "python-single-block-v1"
PRIMARY_POLICY: Final = {
    "functional": "bigcodebench-sanitize",
    "security": "python-fenced-blocks-v2",
}
NOT_COMPUTED: Final = {
    "G7": "harness calibration: the reference and empty programs of check-bigcodebench "
    "and the exclusions of the calibration",
    "G12": "annotation time and correction cycles: the annotation timesheet",
    "G14": "parser on the pilot prompts: the parser generate, score and align runs",
}


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank quantile: the smallest value with at least ``quantile`` of the data."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(quantile * len(ordered))) - 1]


def _ordered(conditions: set[str]) -> list[str]:
    return sorted(conditions, key=lambda name: (CONDITIONS.index(name), name))


def _check_keys(value: Any) -> None:
    """Fail closed if any key of the report could carry a direction."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if any(fragment in str(key).lower() for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise AssertionError(f"directional key in the pilot report: {key}")
            _check_keys(item)
    elif isinstance(value, list):
        for item in value:
            _check_keys(item)


def _journal(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """G1 and G2: every planned request recorded once, with its frozen input."""
    planned = {request["request_id"]: request for request in manifest["requests"]}
    seen: Counter[str] = Counter(row["request_id"] for row in rows)
    recorded = [identifier for identifier in seen if identifier in planned]
    duplicates = sum(count - 1 for identifier, count in seen.items() if identifier in planned)
    unknown = sum(count for identifier, count in seen.items() if identifier not in planned)
    fingerprint = sum(1 for row in rows if row.get("fingerprint") != manifest.get("fingerprint"))
    inputs = sum(
        1
        for row in rows
        if row["request_id"] in planned
        and any(row.get(key) != value for key, value in planned[row["request_id"]].items())
    )
    unrecorded = len(planned) - len(recorded)
    return {
        "G1_journal_completeness": {
            "planned": len(planned),
            "recorded": len(recorded),
            "unrecorded": unrecorded,
            "duplicates": duplicates,
            "unknown": unknown,
            "go": unrecorded == 0 and duplicates == 0 and unknown == 0,
        },
        "G2_resume": {
            "fingerprint_mismatches": fingerprint,
            "input_mismatches": inputs,
            "forced_interruption": "not observable in the files: record the interruption "
            "and the resumed cell of the notebook",
        },
    }


def _by_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    return {name: groups[name] for name in _ordered(set(groups))}


def _generation(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """G3, G4, G5, G10, G11 and G13, read from the journal and its manifest."""
    configuration = manifest.get("configuration", {})
    decoding = configuration.get("decoding", {})
    runtime = configuration.get("runtime", {})
    budget = decoding.get("max_new_tokens")
    context = configuration.get("model", {}).get("max_seq_length")
    groups = _by_condition(rows)
    t_rows = groups.get("T", [])
    unequal = sum(1 for row in t_rows if row["prompt_hash"] != row["source_hash"])
    largest = max(row["input_tokens"] for row in rows)
    output_p99 = _nearest_rank([row["output_tokens"] for row in rows], 0.99)
    truncation = {}
    for name, members in groups.items():
        cut = sum(1 for row in members if row["finish_reason"] == "length")
        truncation[name] = {
            "requests": len(members),
            "present": cut > 0,
            "go": cut / len(members) <= TRUNCATION_THRESHOLD,
        }
    cut_total = sum(1 for row in rows if row["finish_reason"] == "length")
    elapsed = [row["elapsed_seconds"] for row in rows if "elapsed_seconds" in row]
    peaks = [row["peak_vram_bytes"] for row in rows if row.get("peak_vram_bytes") is not None]
    overhead: dict[str, Any] | None = None
    if "T" in groups:
        base = {(row["task_id"], row["replicate_id"]): row["input_tokens"] for row in t_rows}
        overhead = {}
        for name, members in groups.items():
            if name == "T":
                continue
            ratios = [
                row["input_tokens"] / base[row["task_id"], row["replicate_id"]]
                for row in members
                if (row["task_id"], row["replicate_id"]) in base
            ]
            overhead[name] = {
                "pairs": len(ratios),
                "median": statistics.median(ratios),
                "p10": _nearest_rank(ratios, 0.10),
                "p90": _nearest_rank(ratios, 0.90),
                "min": min(ratios),
                "max": max(ratios),
            }
    return {
        "G3_rendering": {
            "t_requests": len(t_rows),
            "t_unequal_to_source": unequal,
            "go": unequal == 0,
        },
        "G4_token_budget": {
            "max_seq_length": context,
            "max_new_tokens": budget,
            "max_input_tokens": largest,
            "max_input_plus_budget": largest + budget if budget is not None else None,
            "go": None if context is None or budget is None else largest + budget <= context,
        },
        "G5_truncation": {
            "by_condition": truncation,
            "length": cut_total,
            "rate": cut_total / len(rows),
            "threshold": TRUNCATION_THRESHOLD,
            "output_tokens_p99": output_p99,
            "output_tokens_max": max(row["output_tokens"] for row in rows),
            "max_new_tokens_if_raised": math.ceil(BUDGET_MARGIN * output_p99),
        },
        "G10_provenance": {
            "purpose": configuration.get("purpose"),
            "revision_verified": runtime.get("revision_verified"),
            "chat_template_matches_official": runtime.get("chat_template_matches_official"),
            "repetition_penalty": decoding.get("repetition_penalty"),
            "source_digest": manifest.get("source", {}).get("digest"),
            "go": runtime.get("revision_verified") is True
            and runtime.get("chat_template_matches_official") is True,
        },
        "G11_resources": {
            "elapsed_seconds_total": math.fsum(elapsed) if elapsed else None,
            "elapsed_seconds_median": statistics.median(elapsed) if elapsed else None,
            "elapsed_seconds_p95": _nearest_rank(elapsed, 0.95) if elapsed else None,
            "peak_vram_bytes": max(peaks) if peaks else None,
            "devices": configuration.get("environment", {}).get("accelerator", {}).get("devices"),
        },
        "G13_input_overhead": overhead,
    }


def _evaluation(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[OutcomeRecord],
    details: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """G6, G6b, G8, G9 and b + c, read from the evaluation of the recorded journal."""
    endpoints = {record.endpoint for record in outcomes}
    if len(endpoints) != 1:
        raise ValueError("the outcomes must hold exactly one endpoint")
    (endpoint,) = endpoints
    cells = Counter((row["task_id"], row["condition"], row["replicate_id"]) for row in rows)
    judged = Counter((r.task_id, r.condition, r.replicate_id) for r in outcomes)
    if judged != cells:
        raise ValueError("the outcomes do not cover the recorded journal exactly")
    if Counter(d["request_id"] for d in details) != Counter(r["request_id"] for r in rows):
        raise ValueError("the details do not cover the recorded journal exactly")
    groups = _by_condition(details)
    if endpoint == "functional":
        failed = [d for d in details if d["sanitized_lines"] == 0]
    else:
        failed = [d for d in details if d["extraction_status"] != "ok"]
    diagnostic = sum(1 for d in details if d["extraction_statuses"].get(DIAGNOSTIC_POLICY) != "ok")
    extraction = {
        "primary_policy": PRIMARY_POLICY[endpoint],
        "rows": len(details),
        "failures": len(failed),
        "rate": len(failed) / len(details),
        "threshold": EXTRACTION_THRESHOLD,
        "go": len(failed) / len(details) <= EXTRACTION_THRESHOLD,
        "by_condition": {
            name: {
                "rows": len(members),
                "present": any(d["condition"] == name for d in failed),
            }
            for name, members in groups.items()
        },
        "diagnostic": {
            DIAGNOSTIC_POLICY: {"failures": diagnostic, "rate": diagnostic / len(details)}
        },
    }
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "G6_extraction": extraction,
        "G6b_prompt_as_data": None,
        "G8_flakiness": None,
        "G9_codeql": None,
        "discordant_pairs": None,
    }
    if endpoint == "functional":
        result["G6b_prompt_as_data"] = {
            "by_condition": {
                name: {
                    "rows": len(members),
                    "json_response_present": any(d["reason"] == "json_response" for d in members),
                    "prompt_as_data_present": any(d["reason"] == "prompt_as_data" for d in members),
                }
                for name, members in groups.items()
            },
            "total": {
                "rows": len(details),
                "json_response": sum(1 for d in details if d["reason"] == "json_response"),
                "prompt_as_data": sum(1 for d in details if d["reason"] == "prompt_as_data"),
            },
        }
        executed = [d for d in details if d["executed"]]
        discordant = sum(
            1
            for d in executed
            if "missing" not in d["run_statuses"][:2]
            and d["run_statuses"][0] != d["run_statuses"][1]
        )
        agreement = 1 - discordant / len(executed) if executed else None
        result["G8_flakiness"] = {
            "executed": len(executed),
            "discordant_runs": discordant,
            "agreement": agreement,
            "flaky": sum(1 for d in details if d["flaky"]),
            "threshold": AGREEMENT_THRESHOLD,
            "go": agreement is not None and agreement >= AGREEMENT_THRESHOLD,
        }
        result["discordant_pairs"] = _discordant(outcomes)
    else:
        usable = [d for d in details if d["usable"]]
        analysed = sum(1 for d in usable if d["analysed"] and d["checks_complete"])
        rate = analysed / len(usable) if usable else None
        result["G9_codeql"] = {
            "measure": details[0]["measure"] if details else None,
            "usable": len(usable),
            "analysed": analysed,
            "rate": rate,
            "threshold": CODEQL_THRESHOLD,
            "go": rate is not None and rate >= CODEQL_THRESHOLD,
        }
    return result


def _discordant(outcomes: Sequence[OutcomeRecord]) -> dict[str, Any] | None:
    """Count b + c over the complete T/H-gold pairs of a single-replicate functional ledger."""
    replicates = {record.replicate_id for record in outcomes}
    arms: dict[str, dict[str, str]] = {}
    for record in outcomes:
        arms.setdefault(record.task_id, {})[record.condition] = record.outcome
    if len(replicates) != 1 or not all({"T", "H-gold"} <= set(cell) for cell in arms.values()):
        return None
    complete = [cell for cell in arms.values() if "missing" not in (cell["T"], cell["H-gold"])]
    total = sum(1 for cell in complete if cell["T"] != cell["H-gold"])
    return {
        "conditions": ["T", "H-gold"],
        "complete_pairs": len(complete),
        "discordant_total": total,
        "discordant_rate": total / len(complete) if complete else None,
    }


def pilot_report(
    manifest: Mapping[str, Any],
    journal: Sequence[Mapping[str, Any]],
    outcomes: Sequence[OutcomeRecord] | None = None,
    details: Sequence[Mapping[str, Any]] | None = None,
    *,
    restrict: bool = False,
) -> dict[str, Any]:
    """Compute the pilot criteria of one journal and, optionally, of its evaluation.

    Args:
        manifest: The ``empirical-generate`` manifest of the journal.
        journal: Its rows, as read; duplicates and unknown rows are counted.
        outcomes: The outcome ledger of one endpoint (one check run), or, with
            ``restrict``, the ledgers of the check runs of the merged files.
        details: The details written by the same check run(s).
        restrict: Keep only the outcomes of the journal's cells and the
            details of its requests (two journals in one session, C-26); the
            exact coverage check applies after the restriction, so a missing
            or doubly judged cell still stops the report.

    Returns:
        A JSON-compatible report with no success rate by condition and no
        difference between conditions.

    Raises:
        ValueError: If only one of outcomes and details is given, or they do
            not cover the recorded journal exactly.
    """
    if (outcomes is None) != (details is None):
        raise ValueError("outcomes and details must be given together")
    if restrict and outcomes is None:
        raise ValueError("restrict applies to an evaluation: give outcomes and details")
    planned = {request["request_id"] for request in manifest["requests"]}
    unique: dict[str, Mapping[str, Any]] = {}
    for row in journal:
        if row["request_id"] in planned:
            unique.setdefault(row["request_id"], row)
    rows = list(unique.values())
    if not rows:
        raise ValueError("the journal has no planned request recorded")
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "endpoint": None,
        "conditions": _ordered({request["condition"] for request in manifest["requests"]}),
        **_journal(manifest, journal),
        **_generation(manifest, rows),
        "G6_extraction": None,
        "G6b_prompt_as_data": None,
        "G8_flakiness": None,
        "G9_codeql": None,
        "discordant_pairs": None,
        "not_computed_here": dict(NOT_COMPUTED),
    }
    if outcomes is not None and details is not None:
        if restrict:
            cells = {(row["task_id"], row["condition"], row["replicate_id"]) for row in rows}
            outcomes = [r for r in outcomes if (r.task_id, r.condition, r.replicate_id) in cells]
            details = [d for d in details if d["request_id"] in unique]
        report.update(_evaluation(rows, outcomes, details))
    json.dumps(report, allow_nan=False)
    _check_keys(report)
    return report
