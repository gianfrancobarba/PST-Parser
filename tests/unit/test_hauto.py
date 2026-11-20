"""H-auto (M19): parser predictions into spans, S_parse, the two H3 estimates and the gate."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    analyze_paired,
    read_outcomes,
    write_outcomes,
)
from pstparser.empirical.auto import (
    ATTACH,
    STRICT,
    auto_annotate,
    clopper_pearson,
    hauto_ledgers,
    leaf_path,
    load_s_parse,
    macro_f1,
    parser_gate,
    predicted_spans,
)
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CanonicalPrompt
from pstparser.empirical.rendering import render_prompt_variants
from pstparser.pst.taxonomy import LEAF_PATHS, assemble

TEXT = "Write a function.\nReturn the list.\n\nNote: keep it short."
SENTENCES = ("Write a function.", "Return the list.", "Note: keep it short.")


def tree(**leaves: list[str]) -> str:
    values = {path: [] for path in LEAF_PATHS}
    for name, phrases in leaves.items():
        values[name.replace("__", ".")] = phrases
    return json.dumps(assemble(values))


EXACT = tree(
    main_instruction=[SENTENCES[0]],
    context__format=[SENTENCES[1]],
    context__constraints=[SENTENCES[2]],
)


def reconstructed(text: str, spans: Any, policy: str) -> str:
    plain, _ = render_prompt_variants(CanonicalPrompt("t", text, spans, policy))
    return plain.text


@pytest.mark.parametrize("policy", [STRICT, ATTACH])
def test_an_exact_prediction_becomes_source_spans(policy: str) -> None:
    result = predicted_spans(TEXT, EXACT, policy=policy)
    assert result.status == "ok" and result.cause is None
    assert [(s.path, s.phrase) for s in result.spans] == [
        ("main_instruction", SENTENCES[0]),
        ("context.format", SENTENCES[1]),
        ("context.constraints", SENTENCES[2]),
    ]
    assert reconstructed(TEXT, result.spans, policy) == TEXT
    assert result.diagnostics["located"] == 3 and result.diagnostics["non_exact"] == 0


def test_an_omitted_sentence_fails_strict_and_is_attached_to_the_previous_span() -> None:
    omitted = tree(main_instruction=[SENTENCES[0]], context__constraints=[SENTENCES[2]])
    strict = predicted_spans(TEXT, omitted, policy=STRICT)
    assert (strict.status, strict.cause) == ("parser_failure", "content_gap")
    attach = predicted_spans(TEXT, omitted, policy=ATTACH)
    assert attach.status == "ok"
    assert [(s.path, s.phrase) for s in attach.spans] == [
        ("main_instruction", "Write a function.\nReturn the list."),
        ("context.constraints", SENTENCES[2]),
    ]
    assert attach.diagnostics["attached_gaps"] == 1
    assert reconstructed(TEXT, attach.spans, ATTACH) == TEXT


def test_a_prefix_gap_extends_the_first_span() -> None:
    late = tree(context__format=[SENTENCES[1]], context__constraints=[SENTENCES[2]])
    attach = predicted_spans(TEXT, late, policy=ATTACH)
    assert attach.spans[0].path == "context.format"
    assert attach.spans[0].phrase == "Write a function.\nReturn the list."


def test_an_invented_phrase_fails_strict_and_is_dropped_by_attach() -> None:
    invented = tree(
        main_instruction=[SENTENCES[0], "Use recursion."],
        context__format=[SENTENCES[1]],
        context__constraints=[SENTENCES[2]],
    )
    strict = predicted_spans(TEXT, invented, policy=STRICT)
    assert (strict.status, strict.cause) == ("parser_failure", "absent_phrase")
    assert strict.diagnostics["absent"] == 1
    attach = predicted_spans(TEXT, invented, policy=ATTACH)
    assert attach.status == "ok" and attach.diagnostics["dropped_phrases"] == 1
    assert reconstructed(TEXT, attach.spans, ATTACH) == TEXT


def test_a_phrase_claimed_twice_is_contested() -> None:
    twice = tree(
        main_instruction=[SENTENCES[0]],
        context__format=[SENTENCES[1]],
        context__constraints=[SENTENCES[2]],
        context__data=[SENTENCES[1]],
    )
    strict = predicted_spans(TEXT, twice, policy=STRICT)
    assert (strict.status, strict.cause) == ("parser_failure", "contested_phrase")
    attach = predicted_spans(TEXT, twice, policy=ATTACH)
    assert attach.status == "ok" and attach.diagnostics["dropped_phrases"] == 1


def test_a_collapsed_match_takes_the_phrase_from_the_source() -> None:
    text = "Write  a\nfunction. Return it."
    prediction = tree(main_instruction=["Write a function."], context__format=["Return it."])
    result = predicted_spans(text, prediction, policy=STRICT)
    assert result.status == "ok" and result.diagnostics["non_exact"] == 1
    assert result.spans[0].phrase == "Write  a\nfunction."
    assert reconstructed(text, result.spans, STRICT) == text


@pytest.mark.parametrize("policy", [STRICT, ATTACH])
def test_broken_json_fails_with_both_policies(policy: str) -> None:
    result = predicted_spans(TEXT, EXACT[:-3], policy=policy)
    assert (result.status, result.cause) == ("parser_failure", "invalid_output")


def test_attach_fails_only_without_any_located_phrase() -> None:
    nothing = tree(main_instruction=["Something else entirely."])
    assert predicted_spans(TEXT, nothing, policy=ATTACH).cause == "no_located_phrase"


def test_blank_leaves_are_ignored() -> None:
    blank = tree(
        main_instruction=[SENTENCES[0], "  "],
        context__format=[SENTENCES[1]],
        context__constraints=[SENTENCES[2]],
    )
    result = predicted_spans(TEXT, blank, policy=STRICT)
    assert result.status == "ok" and result.diagnostics["blank_phrases"] == 1


def test_attach_is_deterministic() -> None:
    omitted = tree(main_instruction=[SENTENCES[0]])
    assert predicted_spans(TEXT, omitted, policy=ATTACH) == predicted_spans(
        TEXT, omitted, policy=ATTACH
    )


def test_leaf_paths_lose_the_root_and_the_index() -> None:
    assert leaf_path("prompt.context.data[0]") == "context.data"
    assert leaf_path("prompt.main_instruction[3]") == "main_instruction"
    with pytest.raises(ValueError):
        leaf_path("other.main_instruction[0]")


def test_unknown_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="policy"):
        predicted_spans(TEXT, EXACT, policy="auto-v0")


def test_clopper_pearson_by_reference_values() -> None:
    # Beta quantiles: 1 of 4 -> (B(0.025; 1, 4), B(0.975; 2, 3)); 0 of 5 -> (0, 1 - 0.025^(1/5)).
    low, high = clopper_pearson(1, 4)
    assert low == pytest.approx(0.0063094632, abs=1e-9)
    assert high == pytest.approx(0.8058795503, abs=1e-9)
    assert clopper_pearson(0, 5) == pytest.approx((0.0, 1 - 0.025 ** (1 / 5)), abs=1e-9)


def sources() -> list[dict[str, Any]]:
    return [
        {"task_id": "BigCodeBench/1", "text": TEXT},
        {"task_id": "BigCodeBench/2", "text": "Do it. Then stop."},
        {"task_id": "BigCodeBench/3", "text": "Only one sentence."},
        {"task_id": "BigCodeBench/4", "text": "First part. Second part."},
    ]


def predictions(targets: str = "{}") -> list[dict[str, Any]]:
    outputs = [
        EXACT,
        tree(main_instruction=["Do it."]),  # omits "Then stop."
        "not json",
        tree(main_instruction=["First part."], context__format=["Second part."]),
    ]
    return [
        {"index": i, "prompt": row["text"], "target": targets, "prediction": output}
        for i, (row, output) in enumerate(zip(sources(), outputs, strict=True))
    ]


ID_MAP = {"format": "id-map-v1", "tasks": [row["task_id"] for row in sources()]}


def test_the_conversion_writes_s_parse_with_its_hash_and_the_failure_rate() -> None:
    conversion = auto_annotate(predictions(), ID_MAP, sources(), policy=STRICT)
    document = conversion.s_parse
    assert document["tasks"] == ["BigCodeBench/1", "BigCodeBench/4"]
    assert load_s_parse(document) == document["tasks"]
    assert (document["n"], document["failures"], document["failure_rate"]) == (4, 2, 0.5)
    assert document["causes"] == {"invalid_output": 1, "content_gap": 1}
    assert document["failure_rate_ci95_clopper_pearson"] == list(clopper_pearson(2, 4))
    assert [f["task_id"] for f in conversion.failures] == ["BigCodeBench/2", "BigCodeBench/3"]
    attach = auto_annotate(predictions(), ID_MAP, sources(), policy=ATTACH)
    assert attach.s_parse["tasks"] == ["BigCodeBench/1", "BigCodeBench/2", "BigCodeBench/4"]


def test_the_conversion_never_reads_the_gold_target() -> None:
    gold = tree(context__data=["anything"])
    for policy in (STRICT, ATTACH):
        assert auto_annotate(predictions(), ID_MAP, sources(), policy=policy) == auto_annotate(
            predictions(gold), ID_MAP, sources(), policy=policy
        )


def test_the_conversion_refuses_a_prediction_of_another_text() -> None:
    rows = predictions()
    rows[0]["prompt"] = "changed"
    with pytest.raises(ValueError, match="source text"):
        auto_annotate(rows, ID_MAP, sources(), policy=STRICT)
    with pytest.raises(ValueError, match="one prediction"):
        auto_annotate(predictions()[:3], ID_MAP, sources(), policy=STRICT)


def test_a_tampered_s_parse_list_is_refused() -> None:
    document = auto_annotate(predictions(), ID_MAP, sources(), policy=STRICT).s_parse
    with pytest.raises(ValueError, match="hash"):
        load_s_parse({**document, "tasks": [*document["tasks"], "BigCodeBench/2"]})


def ledger(endpoint: str = "functional") -> list[OutcomeRecord]:
    """T on all four tasks; H-auto on S_parse (1 and 4); H-gold everywhere, left out."""
    good, bad = ("pass", "fail") if endpoint == "functional" else ("N", "V")
    t = {
        "BigCodeBench/1": bad,
        "BigCodeBench/2": good,
        "BigCodeBench/3": good,
        "BigCodeBench/4": bad,
    }
    rows = [OutcomeRecord("a", endpoint, task, "T", 0, value) for task, value in t.items()]
    rows += [OutcomeRecord("a", endpoint, task, "H-gold", 0, good) for task in t]
    rows += [
        OutcomeRecord("a", endpoint, "BigCodeBench/1", "H-auto", 0, good),
        OutcomeRecord("a", endpoint, "BigCodeBench/4", "H-auto", 0, good),
    ]
    return rows


FAILURES = [
    {"task_id": "BigCodeBench/2", "cause": "content_gap"},
    {"task_id": "BigCodeBench/3", "cause": "invalid_output"},
]
S_PARSE = ["BigCodeBench/1", "BigCodeBench/4"]


def test_the_primary_estimate_is_on_s_parse_and_the_end_to_end_is_separate() -> None:
    primary, end_to_end = hauto_ledgers(ledger(), S_PARSE, FAILURES)
    assert {r.task_id for r in primary} == set(S_PARSE)
    assert {r.condition for r in primary} == {"T", "H-auto"}
    injected = [r for r in end_to_end if r.reason.startswith("parser_failure")]
    assert [(r.task_id, r.outcome, r.reason) for r in injected] == [
        ("BigCodeBench/2", "fail", "parser_failure:content_gap"),
        ("BigCodeBench/3", "fail", "parser_failure:invalid_output"),
    ]
    design = AnalysisDesign(
        "a-s-parse", "functional", tuple(TaskSpec(t) for t in S_PARSE), conditions=("T", "H-auto")
    )
    # On S_parse H-auto passes both tasks T fails: +1.
    assert analyze_paired(design, primary, bootstrap_samples=50).delta == 1.0
    everything = tuple(TaskSpec(f"BigCodeBench/{i}") for i in range(1, 5))
    e2e = AnalysisDesign("a-end-to-end", "functional", everything, conditions=("T", "H-auto"))
    # End-to-end: +1, -1, -1, +1 -> 0.
    assert analyze_paired(e2e, end_to_end, bootstrap_samples=50).delta == 0.0


def test_security_failures_become_i_for_every_replicate() -> None:
    rows = ledger("security")
    rows += [
        OutcomeRecord("a", "security", r.task_id, r.condition, 1, r.outcome) for r in list(rows)
    ]
    _, end_to_end = hauto_ledgers(rows, S_PARSE, FAILURES)
    injected = sorted((r.task_id, r.replicate_id, r.outcome) for r in end_to_end if r.reason)
    assert injected == [
        ("BigCodeBench/2", 0, "I"),
        ("BigCodeBench/2", 1, "I"),
        ("BigCodeBench/3", 0, "I"),
        ("BigCodeBench/3", 1, "I"),
    ]


def test_inconsistent_h3_ledgers_are_refused() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        hauto_ledgers(ledger(), [*S_PARSE, "BigCodeBench/2"], FAILURES)
    with pytest.raises(ValueError, match="H-auto must"):
        hauto_ledgers(
            ledger(),
            S_PARSE[:1],
            [*FAILURES, {"task_id": "BigCodeBench/4", "cause": "content_gap"}],
        )


# The bindings of the gate (R-HAU-02): the conversion and the score read the same predictions.
BOUND = {"predictions_sha256": "p" * 64, "id_map_sha256": "m" * 64}
BIND = {"id_map": ID_MAP, "id_map_sha256": "m" * 64, "sample_tasks": ID_MAP["tasks"]}
SCORE = {
    "total": 4,
    "provenance": {"predictions_sha256": "p" * 64, "indices": [0, 1, 2, 3]},
    "field_f1": {
        "prompt.main_instruction": {"precision": 1, "recall": 1, "f1": 0.9, "support": 10},
        "prompt.context.format": {"precision": 1, "recall": 1, "f1": 0.7, "support": 5},
        "prompt.context.role": {"precision": 0, "recall": 0, "f1": None, "support": 0},
    },
}


def test_macro_f1_averages_the_leaves_with_gold_support() -> None:
    value, leaves = macro_f1(SCORE["field_f1"])
    assert value == pytest.approx(0.8)
    assert leaves == ["prompt.context.format", "prompt.main_instruction"]


@pytest.mark.parametrize(
    ("thresholds", "status", "reasons"),
    [
        ({"max_strict_failure": 0.5, "min_macro_f1": 0.8}, "pass", 0),  # 0.5 <= 0.5, 0.8 >= 0.8
        ({"max_strict_failure": 0.49, "min_macro_f1": 0.8}, "fail", 1),
        ({"max_strict_failure": 0.5, "min_macro_f1": 0.81}, "fail", 1),
        ({"max_strict_failure": 0.25, "min_macro_f1": 0.9}, "fail", 2),
    ],
)
def test_the_gate_above_and_below_each_threshold(
    thresholds: dict[str, float], status: str, reasons: int
) -> None:
    s_parse = auto_annotate(predictions(), ID_MAP, sources(), policy=STRICT, inputs=BOUND).s_parse
    gate = parser_gate(s_parse, SCORE, thresholds, **BIND)
    assert gate["status"] == status and len(gate["reasons"]) == reasons
    assert gate["h3"] == ("confirmatory" if status == "pass" else "descriptive")
    assert gate["strict_failure_rate"] == 0.5 and gate["macro_f1"] == pytest.approx(0.8)


def test_the_gate_refuses_missing_thresholds_and_a_non_strict_conversion() -> None:
    s_parse = auto_annotate(predictions(), ID_MAP, sources(), policy=STRICT, inputs=BOUND).s_parse
    with pytest.raises(ValueError, match="min_macro_f1"):
        parser_gate(s_parse, SCORE, {"max_strict_failure": 0.25}, **BIND)
    attach = auto_annotate(predictions(), ID_MAP, sources(), policy=ATTACH, inputs=BOUND).s_parse
    with pytest.raises(ValueError, match="strict"):
        parser_gate(attach, SCORE, {"max_strict_failure": 0.25, "min_macro_f1": 0.8}, **BIND)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def test_the_cli_converts_renders_s_parse_and_gates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    preds = write_jsonl(tmp_path / "predictions.jsonl", predictions())
    id_map = tmp_path / "id-map.json"
    id_map.write_text(json.dumps(ID_MAP), encoding="utf-8")
    srcs = write_jsonl(tmp_path / "sources.jsonl", sources())
    out = tmp_path / "hauto"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "auto-annotate",
            str(preds),
            str(id_map),
            str(srcs),
            "--policy",
            STRICT,
            "--out",
            str(out / "hauto.canonical.jsonl"),
            "--failures",
            str(out / "parser_failures.jsonl"),
            "--s-parse",
            str(out / "s_parse.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "empirical-auto-annotate_manifest.json").is_file()
    rendered = tmp_path / "hauto.variants.jsonl"
    result = runner.invoke(
        app,
        [
            "render",
            str(out / "hauto.canonical.jsonl"),
            str(rendered),
            "--condition",
            "H-auto",
            "--no-traditional",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in rendered.read_text("utf-8").splitlines()]
    assert [(r["task_id"], r["condition"]) for r in rows] == [
        ("BigCodeBench/1", "H-auto"),
        ("BigCodeBench/4", "H-auto"),
    ]
    gold = write_jsonl(
        tmp_path / "gold.jsonl",
        [
            {
                "task_id": row["task_id"],
                "text": row["text"],
                "annotation_version": "gold-v2.0",
                "annotations": [
                    {
                        "path": "main_instruction",
                        "start": 0,
                        "end": len(row["text"]),
                        "phrase": row["text"],
                    }
                ],
            }
            for row in sources()
        ],
    )
    only = tmp_path / "gold-s-parse.variants.jsonl"
    result = runner.invoke(
        app, ["render", str(gold), str(only), "--tasks", str(out / "s_parse.json")]
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in only.read_text("utf-8").splitlines()]
    assert {r["task_id"] for r in rows} == {"BigCodeBench/1", "BigCodeBench/4"}
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"max_strict_failure": 0.25, "min_macro_f1": 0.8}))
    score = tmp_path / "results.json"
    provenance = {"predictions_sha256": hashlib.sha256(preds.read_bytes()).hexdigest()}
    score.write_text(json.dumps({**SCORE, "provenance": {**SCORE["provenance"], **provenance}}))
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"tasks": [{"task_id": t} for t in ID_MAP["tasks"]]}))
    result = runner.invoke(
        app,
        [
            "parser-gate",
            str(out / "s_parse.json"),
            str(score),
            "--thresholds",
            str(thresholds),
            "--id-map",
            str(id_map),
            "--sample",
            str(sample),
            "--out",
            str(tmp_path / "gate" / "gate.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    gate = json.loads((tmp_path / "gate" / "gate.json").read_text("utf-8"))
    assert gate["status"] == "fail" and gate["h3"] == "descriptive"
    assert set(gate["inputs"]) == {
        "s_parse_sha256",
        "score_sha256",
        "thresholds_sha256",
        "id_map_sha256",
        "sample_sha256",
        "c22_sha256",
    }
    assert gate["inputs"]["c22_sha256"] is None and gate["policy"] == "auto-strict-v1"


def test_the_cli_writes_the_two_h3_ledgers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    outcomes = write_outcomes(ledger(), tmp_path / "outcomes.jsonl")
    document = auto_annotate(predictions(), ID_MAP, sources(), policy=STRICT).s_parse
    s_parse = tmp_path / "s_parse.json"
    s_parse.write_text(json.dumps(document), encoding="utf-8")
    failures = write_jsonl(tmp_path / "failures.jsonl", FAILURES)
    result = CliRunner().invoke(
        app,
        [
            "hauto-ledgers",
            str(outcomes),
            "--s-parse",
            str(s_parse),
            "--failures",
            str(failures),
            "--directory",
            str(tmp_path / "h3"),
        ],
    )
    assert result.exit_code == 0, result.output
    primary = read_outcomes(tmp_path / "h3" / "outcomes-s-parse.jsonl")
    end_to_end = read_outcomes(tmp_path / "h3" / "outcomes-end-to-end.jsonl")
    assert {r.analysis_id for r in primary} == {"a-s-parse"}
    assert {r.analysis_id for r in end_to_end} == {"a-end-to-end"}
    assert len(primary) == 4 and len(end_to_end) == 8
    every = read_outcomes(tmp_path / "h3" / "outcomes-s-parse-all.jsonl")
    assert {r.analysis_id for r in every} == {"a-s-parse-all"}
    assert {r.task_id for r in every} == {r.task_id for r in primary}


# Finding of the dress rehearsal (19/11/2026): RQ3c (H-auto - H-gold) and the
# decomposition of Delta_auto are estimated on S_parse, so a third ledger keeps
# every condition of the S_parse tasks.


def test_the_s_parse_ledger_keeps_every_condition_of_the_s_parse_tasks() -> None:
    from pstparser.empirical.auto import s_parse_ledger

    records = [
        OutcomeRecord("a", "functional", task, condition, 0, "pass")
        for task in ("BigCodeBench/1", "BigCodeBench/2")
        for condition in ("T", "H-gold", "E")
    ] + [OutcomeRecord("a", "functional", "BigCodeBench/1", "H-auto", 0, "fail")]
    kept = s_parse_ledger(records, ["BigCodeBench/1"])
    assert {r.analysis_id for r in kept} == {"a-s-parse-all"}
    assert sorted(r.condition for r in kept) == ["E", "H-auto", "H-gold", "T"]
    with pytest.raises(ValueError, match="H-auto"):
        s_parse_ledger(records, ["BigCodeBench/1", "BigCodeBench/2"])
