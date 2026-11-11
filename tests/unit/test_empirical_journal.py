import json

import pytest

from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.rendering import render_prompt_variants


def plan():
    return request_plan(
        render_prompt_variants(
            CanonicalPrompt(
                "task",
                "hello",
                (SpanAnnotation("main_instruction", 0, 5, "hello"),),
                "v1",
            )
        ),
        [0, 1],
        42,
    )


def output(request):
    return {"raw_output": "code", "input_tokens": 5, "output_tokens": 1, "finish_reason": "eos"}


def test_journal_resumes_without_repeating_completed_requests(tmp_path) -> None:
    requests = plan()
    calls = []

    def interrupted(request):
        calls.append(request["request_id"])
        if len(calls) == 2:
            raise RuntimeError("interruption")
        return output(request)

    with pytest.raises(RuntimeError):
        run_journal(requests, {"model": "fixed"}, tmp_path, interrupted)
    assert len((tmp_path / "generations.jsonl").read_text().splitlines()) == 1
    resumed = []

    def callback(request):
        resumed.append(request["request_id"])
        return output(request)

    result = run_journal(requests, {"model": "fixed"}, tmp_path, callback)
    assert len(result) == 4 and len(resumed) == 3
    assert calls[0] not in resumed
    assert run_journal(requests, {"model": "fixed"}, tmp_path, callback) == result
    assert len(resumed) == 3


def test_resume_rejects_changed_configuration(tmp_path) -> None:
    run_journal(plan(), {"model": "fixed"}, tmp_path, output)
    with pytest.raises(ValueError, match="resume refused"):
        run_journal(plan(), {"model": "changed"}, tmp_path, output)


def test_corrupt_journal_fails_closed(tmp_path) -> None:
    run_journal(plan(), {}, tmp_path, output)
    with (tmp_path / "generations.jsonl").open("a") as handle:
        handle.write('{"broken')
    with pytest.raises(json.JSONDecodeError):
        run_journal(plan(), {}, tmp_path, output)


def test_resume_rejects_missing_output_field(tmp_path) -> None:
    run_journal(plan(), {}, tmp_path, output)
    path = tmp_path / "generations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[0]["raw_output"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="raw_output"):
        run_journal(plan(), {}, tmp_path, output)
