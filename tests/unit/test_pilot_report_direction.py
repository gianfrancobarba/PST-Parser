"""R-STA-05 (C-04): the pilot report must not bound the sign of H-gold - T.

Truncations, empty sanitised programs, ``json_response`` and ``prompt_as_data``
are failures by construction. Counted per condition, together with b + c, they
bound c - b; in the scenario below (from the independent review) the sign of
H-gold - T would be certain. Per condition the report may only say whether such
failures are present; their counts are pooled over all conditions.
"""

from typing import Any

from pstparser.empirical.analysis import OutcomeRecord
from pstparser.empirical.pilot_report import pilot_report

N = 10
T_PASS = {0, 1, 2, 3, 4}
H_PASS = {0}
H_REASON = {i: ("json_response" if i % 2 else "prompt_as_data") for i in range(1, N)}


def scenario() -> tuple[dict[str, Any], list[dict[str, Any]], list[OutcomeRecord], list[dict]]:
    requests, rows, outcomes, details = [], [], [], []
    for i in range(N):
        task = f"BigCodeBench/{i}"
        for condition in ("T", "H-gold"):
            request = {
                "request_id": f"r{i}-{condition}",
                "task_id": task,
                "condition": condition,
                "replicate_id": 0,
            }
            requests.append(request)
            rows.append(
                request
                | {
                    "fingerprint": "f",
                    "prompt_hash": "h",
                    "source_hash": "h",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "finish_reason": "eos",
                    "elapsed_seconds": 1.0,
                }
            )
            passed = i in (T_PASS if condition == "T" else H_PASS)
            reason = "pass" if passed else H_REASON[i] if condition == "H-gold" else "assertion"
            outcome = "pass" if passed else "fail"
            outcomes.append(OutcomeRecord("p", "functional", task, condition, 0, outcome, reason))
            details.append(
                request
                | {
                    "extraction_statuses": {"python-single-block-v1": "ok"},
                    "sanitized_lines": 3,
                    "executed": True,
                    "reason": reason,
                    "run_statuses": [outcome, outcome],
                    "flaky": False,
                }
            )
    manifest = {
        "requests": requests,
        "fingerprint": "f",
        "configuration": {"decoding": {"max_new_tokens": 1280}, "model": {"max_seq_length": 4096}},
    }
    return manifest, rows, outcomes, details


def per_condition_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *report["G5_truncation"]["by_condition"].values(),
        *report["G6_extraction"]["by_condition"].values(),
        *report["G6b_prompt_as_data"]["by_condition"].values(),
    ]


def test_no_per_condition_count_of_certain_failures() -> None:
    report = pilot_report(*scenario())
    for entry in per_condition_entries(report):
        for key, value in entry.items():
            if key in {"rows", "requests"}:
                continue
            assert isinstance(value, bool), (key, value)


def test_certain_failures_are_pooled_over_the_conditions() -> None:
    report = pilot_report(*scenario())
    assert report["G6b_prompt_as_data"]["total"] == {
        "rows": 20,
        "json_response": 5,
        "prompt_as_data": 4,
    }
    assert report["G6b_prompt_as_data"]["by_condition"]["H-gold"] == {
        "rows": 10,
        "json_response_present": True,
        "prompt_as_data_present": True,
    }


def test_the_sign_of_h_minus_t_cannot_be_deduced_from_the_report() -> None:
    report = pilot_report(*scenario())
    n = report["discordant_pairs"]["complete_pairs"]
    d = report["discordant_pairs"]["discordant_total"]
    # Lower bound of the certain failures of each condition: one per category present.
    sure = {
        name: int(entry["json_response_present"]) + int(entry["prompt_as_data_present"])
        for name, entry in report["G6b_prompt_as_data"]["by_condition"].items()
    }
    for name, entry in report["G5_truncation"]["by_condition"].items():
        sure[name] += int(entry["present"])
    low, high = d - 2 * (n - sure["T"]), 2 * (n - sure["H-gold"]) - d
    assert low <= 0 <= high
