"""Verified union of the two journals of one generation session (`protocol/analysis/plan.md`).

Journal A holds {T, H-gold, E, H-auto} on S_parse, journal B {T, H-gold, E} on the
other tasks. The union gives the core file (the conditions of every journal, on
exactly the sample) and the extra file (H-auto on S_parse), checked before
check-bigcodebench: no duplicate, sample coverage, same configuration and GPU.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.auto import S_PARSE_FORMAT, as_hauto, tasks_digest
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.journal import merge_journals, request_plan, run_journal
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants

GPU = "NVIDIA L4"


def canonical(index: int) -> CanonicalPrompt:
    text = f"Do thing {index}.\n\ndef task_func():\n    pass\n"
    cut = text.index("def")
    return CanonicalPrompt(
        f"BigCodeBench/{index}",
        text,
        (
            SpanAnnotation("main_instruction", 0, cut - 2, text[: cut - 2]),
            SpanAnnotation("context.data", cut, len(text) - 1, text[cut:-1]),
        ),
        "gold-v2.0",
    )


ENVIRONMENT = "e" * 64


def variants(indices, with_hauto: bool):
    out = []
    for index in indices:
        t, h = render_prompt_variants(canonical(index))
        out += [t, h, render_envelope_variant(canonical(index))]
        if with_hauto:
            out.append(as_hauto(h))
    return out


def configuration(conditions, **changes):
    return {
        "gpu": GPU,
        "conditions": list(conditions),
        "variant_file_sha256": "v" * 64 if "H-auto" in conditions else "w" * 64,
        "decoding": {"do_sample": False, "max_new_tokens": 1280},
        "seed": 7319,
        **changes,
    }


def journal(directory: Path, indices, with_hauto: bool, gpu: str = GPU, **changes) -> Path:
    conditions = ("T", "H-gold", "E", "H-auto") if with_hauto else ("T", "H-gold", "E")
    requests = request_plan(variants(indices, with_hauto), [0], 7319, conditions)

    def generate(request):
        return {
            "raw_output": f"out {request['task_id']} {request['condition']}",
            "input_tokens": 3,
            "output_tokens": 2,
            "finish_reason": "eos",
            "session_id": directory.name,
            "gpu_name": gpu,
            "environment_digest": ENVIRONMENT,
        }

    run_journal(requests, configuration(conditions, **changes), directory, generate)
    return directory


SAMPLE = [f"BigCodeBench/{i}" for i in range(7)]
S_PARSE = SAMPLE[:3]


def pair(tmp_path: Path, **changes) -> list[Path]:
    return [
        journal(tmp_path / "a", range(3), True),
        journal(tmp_path / "b", range(3, 7), False, **changes),
    ]


def test_the_union_splits_the_core_conditions_from_h_auto(tmp_path: Path) -> None:
    merged = merge_journals(pair(tmp_path), SAMPLE)
    assert merged["core_conditions"] == ["T", "H-gold", "E"]
    assert merged["extra_conditions"] == ["H-auto"]
    assert len(merged["core"]) == 21
    assert {row["task_id"] for row in merged["core"]} == set(SAMPLE)
    assert {row["condition"] for row in merged["core"]} == {"T", "H-gold", "E"}
    assert {row["task_id"] for row in merged["extra"]} == set(S_PARSE)
    assert merged["sessions"] == ["a", "b"]


def test_a_task_in_both_journals_is_refused(tmp_path: Path) -> None:
    paths = [journal(tmp_path / "a", range(3), True), journal(tmp_path / "b", range(2, 7), False)]
    with pytest.raises(ValueError, match="more than one journal"):
        merge_journals(paths, SAMPLE)


def test_the_core_must_cover_exactly_the_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample"):
        merge_journals(pair(tmp_path), [*SAMPLE, "BigCodeBench/99"])
    with pytest.raises(ValueError, match="sample"):
        merge_journals(pair(tmp_path / "x"), SAMPLE[:-1])


def test_a_different_configuration_is_refused(tmp_path: Path) -> None:
    paths = pair(tmp_path, decoding={"do_sample": False, "max_new_tokens": 512})
    with pytest.raises(ValueError, match="configuration"):
        merge_journals(paths, SAMPLE)


def test_a_row_on_another_gpu_is_refused(tmp_path: Path) -> None:
    paths = [
        journal(tmp_path / "a", range(3), True),
        journal(tmp_path / "b", range(3, 7), False, gpu="NVIDIA A100"),
    ]
    with pytest.raises(ValueError, match="GPU"):
        merge_journals(paths, SAMPLE)


def test_an_incomplete_journal_is_refused(tmp_path: Path) -> None:
    paths = pair(tmp_path)
    lines = (paths[1] / "generations.jsonl").read_text(encoding="utf-8").splitlines(True)
    (paths[1] / "generations.jsonl").write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        merge_journals(paths, SAMPLE)


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _invoke(tmp_path: Path, paths: list[Path], s_parse_tasks: list[str]):
    sample = _write(tmp_path / "sample.json", {"tasks": [{"task_id": t} for t in SAMPLE]})
    s_parse = _write(
        tmp_path / "s-parse.json",
        {
            "format": S_PARSE_FORMAT,
            "tasks": s_parse_tasks,
            "tasks_sha256": tasks_digest(s_parse_tasks),
        },
    )
    arguments = ["merge-journals", *map(str, paths), "--sample", str(sample)]
    arguments += ["--s-parse", str(s_parse), "--out", str(tmp_path / "merged")]
    return CliRunner().invoke(app, arguments)


def test_the_command_writes_both_files_and_checks_h_auto_against_s_parse(tmp_path: Path) -> None:
    result = _invoke(tmp_path, pair(tmp_path), S_PARSE)
    assert result.exit_code == 0, result.output
    merged = tmp_path / "merged"
    assert len((merged / "generations-core.jsonl").read_text("utf-8").splitlines()) == 21
    assert len((merged / "generations-extra.jsonl").read_text("utf-8").splitlines()) == 3
    manifest = json.loads((merged / "empirical-merge-journals_manifest.json").read_text("utf-8"))
    assert manifest["core_conditions"] == ["T", "H-gold", "E"]
    assert len(manifest["journals"]) == 2

    refused = _invoke(tmp_path / "other", pair(tmp_path / "other"), S_PARSE[:2])
    assert isinstance(refused.exception, ValueError)
    assert "S_parse" in str(refused.exception)


def test_s_parse_is_read_as_the_sample_of_the_h_auto_rows(tmp_path: Path) -> None:
    from pstparser.empirical.cli import _sample_ids

    s_parse = _write(
        tmp_path / "s-parse.json",
        {"format": S_PARSE_FORMAT, "tasks": S_PARSE, "tasks_sha256": tasks_digest(S_PARSE)},
    )
    assert _sample_ids(s_parse, None) == S_PARSE
    _write(s_parse, {"format": S_PARSE_FORMAT, "tasks": S_PARSE, "tasks_sha256": "0" * 64})
    with pytest.raises(ValueError, match="hash"):
        _sample_ids(s_parse, None)


# Finding of the pilot ds-pilot-01: in a pilot each journal points at the render
# record of its own variant file; in a final run every journal points at one freeze.


def _pair(tmp_path: Path, purpose: str, first: str, second: str) -> list[Path]:
    return [
        journal(tmp_path / "a", range(3), True, purpose=purpose, review_record_sha256=first),
        journal(tmp_path / "b", range(3, 7), False, purpose=purpose, review_record_sha256=second),
    ]


def test_a_pilot_merges_journals_with_their_own_render_records(tmp_path: Path) -> None:
    merged = merge_journals(_pair(tmp_path, "pilot", "1" * 64, "2" * 64), SAMPLE)
    assert len(merged["core"]) == 21


def test_a_final_run_needs_one_freeze_for_every_journal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configuration"):
        merge_journals(_pair(tmp_path, "final", "1" * 64, "2" * 64), SAMPLE)
    merged = merge_journals(_pair(tmp_path / "same", "final", "3" * 64, "3" * 64), SAMPLE)
    assert len(merged["core"]) == 21


def test_journals_of_a_pilot_and_of_a_final_run_are_not_merged(tmp_path: Path) -> None:
    paths = [
        journal(tmp_path / "a", range(3), True, purpose="pilot", review_record_sha256="1" * 64),
        journal(tmp_path / "b", range(3, 7), False, purpose="final", review_record_sha256="2" * 64),
    ]
    with pytest.raises(ValueError, match="configuration"):
        merge_journals(paths, SAMPLE)
