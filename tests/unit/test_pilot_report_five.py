"""The pilot report with the five conditions of the pilot (plan of 18/11/2026, P-08, C-04).

T, H-gold, E, F and H-auto in one journal: every condition is listed, no key
reveals a rate or a difference, and two worlds with opposite directions give
the same report.
"""

from typing import Any

from test_pilot_report import keys

from pstparser.empirical.analysis import OutcomeRecord
from pstparser.empirical.auto import as_hauto
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.pilot_report import FORBIDDEN_KEY_FRAGMENTS, pilot_report
from pstparser.empirical.rendering import (
    render_envelope_variant,
    render_flat_variant,
    render_prompt_variants,
)

FIVE = ["T", "H-gold", "E", "F", "H-auto"]
TASKS = [f"BigCodeBench/{i}" for i in range(6)]
CONFIGURATION = {
    "model": {"max_seq_length": 4096},
    "decoding": {"max_new_tokens": 1280, "repetition_penalty": 1.0},
    "purpose": "pilot",
    "runtime": {"revision_verified": True, "chat_template_matches_official": True},
    "environment": {"accelerator": {"devices": ["NVIDIA L4"]}},
}


def variants() -> list:
    rows = []
    for task in TASKS:
        text = f"Solve {task}. Keep it short."
        cut = text.index("Keep")
        prompt = CanonicalPrompt(
            task,
            text,
            (
                SpanAnnotation("main_instruction", 0, cut - 1, text[: cut - 1]),
                SpanAnnotation("context.constraints", cut, len(text), text[cut:]),
            ),
            "v",
        )
        t, h = render_prompt_variants(prompt)
        rows += [t, h, render_envelope_variant(prompt), render_flat_variant(prompt), as_hauto(h)]
    return rows


def generation(request: dict) -> dict[str, Any]:
    return {
        "raw_output": "```python\nx = 1\n```",
        "input_tokens": 100,
        "output_tokens": 50,
        "finish_reason": "eos",
        "elapsed_seconds": 1.0,
        "peak_vram_bytes": 1000,
    }


def journal(tmp_path) -> tuple[dict, list[dict]]:
    import json

    directory = tmp_path / "journal"
    run_journal(request_plan(variants(), [0], 7319, FIVE), CONFIGURATION, directory, generation)
    manifest = json.loads((directory / "empirical-generate_manifest.json").read_text("utf-8"))
    rows = [
        json.loads(x) for x in (directory / "generations.jsonl").read_text("utf-8").split("\n") if x
    ]
    return manifest, rows


def evaluation(rows: list[dict], passing: dict[str, set[int]]):
    outcomes, details = [], []
    for row in rows:
        index = int(row["task_id"].split("/")[1])
        passed = index in passing[row["condition"]]
        status = "pass" if passed else "fail"
        reason = "pass" if passed else "assertion_failure"
        outcomes.append(
            OutcomeRecord(
                "pilot-bcb", "functional", row["task_id"], row["condition"], 0, status, reason
            )
        )
        details.append(
            {
                "request_id": row["request_id"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "replicate_id": 0,
                "extraction_policy": "bigcodebench-sanitize",
                "extraction_statuses": {
                    "python-single-block-v1": "ok",
                    "python-fenced-blocks-v2": "ok",
                    "python-longest-block-v2": "ok",
                },
                "sanitized_lines": 3,
                "executed": True,
                "outcome": status,
                "reason": reason,
                "run_statuses": [status, status],
                "flaky": False,
                "finish_reason": row["finish_reason"],
            }
        )
    return outcomes, details


def test_the_report_lists_five_conditions_and_hides_every_direction(tmp_path) -> None:
    manifest, rows = journal(tmp_path)
    world_a = {"T": {0, 1}, "H-gold": {0, 1, 2, 3}, "E": {0}, "F": {0, 1, 2}, "H-auto": {4}}
    world_b = {"T": {0, 1, 2, 3}, "H-gold": {0, 1}, "E": {0, 1, 2}, "F": {0}, "H-auto": {5}}
    first = pilot_report(manifest, rows, *evaluation(rows, world_a))
    second = pilot_report(manifest, rows, *evaluation(rows, world_b))
    assert first["conditions"] == ["T", "H-gold", "E", "H-auto", "F"]
    assert not [key for key in keys(first) if any(part in key for part in FORBIDDEN_KEY_FRAGMENTS)]
    assert first == second
