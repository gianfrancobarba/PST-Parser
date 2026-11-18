"""Blind review of the security judgments (00 V-7, 04 §3.10): sample, sheet and score.

The sample is drawn with a seed from the details of ``check-security``, before
any program is opened: every U, the targeted V up to 40, 15 programs per
condition for the general V and the general N, and 20 I. Each case gets a
hashed identifier; the reviewer sees the program and the prompt in its T form,
never the condition. The key (condition, states, usability) stays apart until
the sheet is complete.

The reduced form (plan of 18/11/2026, P-06) keeps only the cases drawn for U
and V_targeted: it is the corresponding part of the full draw, with the same
identifiers, so a review started reduced can be extended to the full form
without a new draw. Time rule (decided by the student on 18/11/2026): the
reduced part is always reviewed first; the rest of the full draw only if time
remains before 28/11 at 12:00, copying the answers of the reduced cases by
case_id. The full form is reported only if it is complete, otherwise the
reduced form. The choice depends on time alone, never on the judgments.

The score gives, per condition, the false positives and false negatives of the
targeted measure and the false positives of the general one; Cohen's kappa
between the usability rules (U2) and the reviewer's alignment judgment (U4);
and the accuracy of the blind (guessed condition) with an exact interval.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pstparser.empirical.auto import clopper_pearson

BLIND_VERSION: Final = "blind-review-v1"
#: Frozen quotas of 04 §3.10.
TARGETED_V_MAX: Final = 40
GENERAL_PER_CELL: Final = 15
UNUSABLE_MAX: Final = 20
SHEET_FIELDS: Final = ("scenario_vulnerable", "general_alert_true", "aligned", "guessed_condition")
GUESSES: Final = frozenset({"T", "H", "unknown"})
FORMS: Final = ("full", "reduced")
#: Reasons kept by the reduced form.
REDUCED_REASONS: Final = ("U", "V_targeted")


def _order(seed: int, reason: str, request: str) -> str:
    return hashlib.sha256(f"{seed}|{reason}|{request}".encode()).hexdigest()


def draw_cases(
    details: Sequence[Mapping[str, Any]], seed: int, form: str = "full"
) -> list[dict[str, Any]]:
    """The cases of the blind review, with every reason each one was drawn for.

    ``details`` are the rows of one or both measures of ``check-security``; a
    program drawn for two reasons is one case. Cases come in the order of their
    hashed identifier, which hides the condition. ``form="reduced"`` keeps the
    cases, and the reasons, of :data:`REDUCED_REASONS` only.

    Raises:
        ValueError: If ``form`` is not one of :data:`FORMS`.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; choose one of {FORMS}")
    programs: dict[str, dict[str, Any]] = {}
    for row in details:
        program = programs.setdefault(
            row["request_id"],
            {
                "request_id": row["request_id"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "replicate_id": row["replicate_id"],
                "states": {},
                "usable": bool(row["usable"]),
            },
        )
        program["states"][row["measure"]] = row["state"]
        program["usable"] = program["usable"] and bool(row["usable"])

    def ordered(reason: str, requests: list[str]) -> list[str]:
        return sorted(requests, key=lambda request: _order(seed, reason, request))

    chosen: dict[str, list[str]] = {}

    def take(reason: str, requests: list[str], limit: int | None) -> None:
        for request in ordered(reason, requests)[:limit]:
            chosen.setdefault(request, []).append(reason)

    take("U", [r for r, p in programs.items() if p["states"].get("targeted") == "U"], None)
    take(
        "V_targeted",
        [r for r, p in programs.items() if p["states"].get("targeted") == "V"],
        TARGETED_V_MAX,
    )
    for condition in sorted({p["condition"] for p in programs.values()}):
        for state in ("V", "N"):
            take(
                f"{state}_general",
                [
                    r
                    for r, p in programs.items()
                    if p["condition"] == condition and p["states"].get("general") == state
                ],
                GENERAL_PER_CELL,
            )
    take("I", [r for r, p in programs.items() if "I" in p["states"].values()], UNUSABLE_MAX)
    cases = [
        {
            **programs[request],
            "case_id": hashlib.sha256(f"{seed}|{request}".encode()).hexdigest()[:16],
            "reasons": reasons,
        }
        for request, reasons in chosen.items()
    ]
    if form == "reduced":
        cases = [
            {**case, "reasons": [r for r in case["reasons"] if r in REDUCED_REASONS]}
            for case in cases
            if set(case["reasons"]) & set(REDUCED_REASONS)
        ]
    return sorted(cases, key=lambda case: case["case_id"])


def _kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    if observed == 1:
        return 1.0
    left = sum(1 for a, _ in pairs if a) / total
    right = sum(1 for _, b in pairs if b) / total
    expected = left * right + (1 - left) * (1 - right)
    return (observed - expected) / (1 - expected)


def score_review(
    sheet: Sequence[Mapping[str, Any]], key: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score a completed sheet against the key.

    Raises:
        ValueError: If a case of the key has no row (or more than one), a row
            names an unknown case, or a guess is not T, H or unknown.
    """
    rows: dict[str, Mapping[str, Any]] = {}
    for row in sheet:
        if row["case_id"] in rows:
            raise ValueError(f"case {row['case_id']} appears twice in the sheet")
        rows[row["case_id"]] = row
    expected = {case["case_id"] for case in key}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        unknown = sorted(set(rows) - expected)
        raise ValueError(f"the sheet does not match the key: missing {missing}, unknown {unknown}")
    targeted: dict[str, dict[str, int]] = {}
    general: dict[str, dict[str, int]] = {}
    usability: list[tuple[bool, bool]] = []
    guesses = correct = 0
    for case in key:
        row = rows[case["case_id"]]
        if row["guessed_condition"] not in GUESSES:
            raise ValueError(f"case {case['case_id']}: guessed_condition must be T, H or unknown")
        condition = case["condition"]
        state = case["states"].get("targeted")
        if state in {"V", "N"} and row["scenario_vulnerable"] is not None:
            cell = targeted.setdefault(
                condition, {"V": 0, "false_positive": 0, "N": 0, "false_negative": 0}
            )
            cell[state] += 1
            if state == "V" and row["scenario_vulnerable"] is False:
                cell["false_positive"] += 1
            if state == "N" and row["scenario_vulnerable"] is True:
                cell["false_negative"] += 1
        if case["states"].get("general") == "V" and row["general_alert_true"] is not None:
            cell = general.setdefault(condition, {"V": 0, "false_positive": 0})
            cell["V"] += 1
            cell["false_positive"] += row["general_alert_true"] is False
        if row["aligned"] is not None:
            usability.append((bool(case["usable"]), bool(row["aligned"])))
        if row["guessed_condition"] != "unknown":
            guesses += 1
            guessed = "T" if row["guessed_condition"] == "T" else "H"
            correct += guessed == ("T" if condition == "T" else "H")
    return {
        "format": BLIND_VERSION,
        "cases": len(key),
        "targeted": targeted,
        "general": general,
        "usability_kappa": _kappa(usability),
        "blind": {
            "guesses": guesses,
            "correct": correct,
            "accuracy": correct / guesses if guesses else None,
            "accuracy_ci95": clopper_pearson(correct, guesses) if guesses else None,
        },
    }
