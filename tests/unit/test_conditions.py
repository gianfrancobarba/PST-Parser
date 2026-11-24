"""N conditions in contracts, request plan and paired analysis, and renderer E."""

import hashlib
import json
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    analyze_paired,
    write_outcomes,
)
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CONDITIONS, CanonicalPrompt, PromptVariant, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.rendering import (
    ENVELOPE_RENDERER_VERSION,
    RENDERER_VERSION,
    render_envelope_variant,
    render_prompt_variants,
)
from pstparser.pst.taxonomy import LEAF_PATHS


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def gold_variants(count: int = 7) -> list[PromptVariant]:
    return [variant for i in range(count) for variant in render_prompt_variants(canonical(i))]


def core_variants(count: int = 7) -> list[PromptVariant]:
    return [
        variant
        for i in range(count)
        for variant in (
            *render_prompt_variants(canonical(i)),
            render_envelope_variant(canonical(i)),
        )
    ]


def plan_digest(plan: list[dict]) -> str:
    return sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


# --- Contracts -----------------------------------------------------------------------------


def test_registered_conditions() -> None:
    assert CONDITIONS == ("T", "H-gold", "E", "H-auto", "F", "H-explained")


def test_an_unregistered_condition_is_refused() -> None:
    variant = render_prompt_variants(canonical(0))[0]
    with pytest.raises(ValueError, match="unregistered condition"):
        replace(variant, condition="H")
    assert replace(variant, condition="H-auto").condition == "H-auto"


# --- Renderer E (envelope-json-v1) ---------------------------------------------------------

AWKWARD = 'Città "quoted" \\ back\\slash\n\tdef f(x):\r\n    return {"a": [1]}\u2028end/€ 😀'


def test_envelope_holds_the_source_text_exactly() -> None:
    prompt = CanonicalPrompt("x/1", AWKWARD, (), "unused")
    envelope = render_envelope_variant(prompt)
    assert json.loads(envelope.text) == {"prompt": AWKWARD}
    assert envelope.condition == "E"
    assert envelope.source_hash == sha256(AWKWARD)
    assert envelope.prompt_hash == sha256(envelope.text)
    assert envelope.renderer_version == ENVELOPE_RENDERER_VERSION == "envelope-json-v1"
    assert envelope.annotation_hash == "" and envelope.annotation_version == ""


def test_envelope_uses_the_serialisation_of_h() -> None:
    envelope = render_envelope_variant(CanonicalPrompt("x/1", AWKWARD, (), "unused"))
    # indent=2, real UTF-8 characters, JSON escapes only for quotes, backslash and controls.
    assert envelope.text == (
        '{\n  "prompt": "Città \\"quoted\\" \\\\ back\\\\slash\\n\\tdef f(x):\\r\\n'
        '    return {\\"a\\": [1]}\u2028end/€ 😀"\n}'
    )
    cut = AWKWARD.index("def")
    spans = (
        SpanAnnotation("main_instruction", 0, cut, AWKWARD[:cut]),
        SpanAnnotation("context.data", cut, len(AWKWARD), AWKWARD[cut:]),
    )
    _, hierarchical = render_prompt_variants(CanonicalPrompt("x/1", AWKWARD, spans, "v"))
    code = json.dumps(AWKWARD[cut:], ensure_ascii=False)
    assert f'"phrase": {code}' in hierarchical.text
    assert envelope.text.endswith(code[1:] + "\n}")


def test_envelope_has_no_pst_label_and_ignores_annotations() -> None:
    annotated = canonical(3)
    bare = replace(annotated, annotations=(), annotation_version="")
    envelope = render_envelope_variant(annotated)
    assert envelope == render_envelope_variant(bare) == render_envelope_variant(annotated)
    assert list(json.loads(envelope.text)) == ["prompt"]
    labels = {part for path in LEAF_PATHS for part in path.split(".")}
    assert not any(f'"{label}"' in envelope.text for label in labels)
    traditional, hierarchical = render_prompt_variants(annotated)
    assert envelope.source_hash == traditional.source_hash == hierarchical.source_hash
    assert envelope.task_id == traditional.task_id


@pytest.mark.parametrize("text", ["", "  \n\t"])
def test_envelope_refuses_an_empty_prompt(text) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        render_envelope_variant(CanonicalPrompt("x/1", text, (), ""))


def test_envelope_refuses_an_empty_task() -> None:
    with pytest.raises(ValueError, match="task_id"):
        render_envelope_variant(CanonicalPrompt(" ", "text", (), ""))


# --- Request plan --------------------------------------------------------------------------


def test_two_condition_plan_is_byte_identical_to_the_previous_code() -> None:
    # Digests of request_plan at commit dcda9c5, on the same input.
    variants = gold_variants()
    assert plan_digest(request_plan(variants, [0], 7319)) == (
        "5b16f9a0b8861517fcefe209e49c52c9854cd924ed52380e175d4efdff518481"
    )
    assert plan_digest(request_plan(variants, [0, 1, 2, 3, 4], 42)) == (
        "f65bce6851c22a1d9a612f4973274293f6d2a47837a3630f4118e3dbf2521653"
    )
    assert request_plan(variants, [0], 7319) == request_plan(
        variants, [0], 7319, conditions=("H-gold", "T")
    )


def test_three_condition_plan_interleaves_every_condition_in_each_block() -> None:
    conditions = ("T", "H-gold", "E")
    plan = request_plan(core_variants(), [0, 1], 7319, conditions=conditions)
    assert len(plan) == 7 * 2 * 3
    blocks = [plan[i : i + 3] for i in range(0, len(plan), 3)]
    orders = set()
    for block in blocks:
        assert len({(r["task_id"], r["replicate_id"]) for r in block}) == 1
        assert sorted(r["condition"] for r in block) == sorted(conditions)
        orders.add(tuple(r["condition"] for r in block))
    assert len(orders) > 1  # the order within a block is randomised
    assert len({(r["task_id"], r["replicate_id"]) for r in plan}) == 14
    assert len({r["request_id"] for r in plan}) == len(plan)
    for request in plan:
        assert request["seed"] == 7319 + request["replicate_id"]
        assert request["request_id"] == sha256(
            json.dumps([request["task_id"], request["condition"], request["replicate_id"]])
        )


def test_three_condition_join_is_by_task_and_source() -> None:
    plan = request_plan(core_variants(), [0], 7, conditions=("T", "H-gold", "E"))
    by_task: dict[str, dict[str, dict]] = {}
    for request in plan:
        by_task.setdefault(request["task_id"], {})[request["condition"]] = request
    for arms in by_task.values():
        assert len({arm["source_hash"] for arm in arms.values()}) == 1
        assert arms["T"]["prompt_hash"] == arms["T"]["source_hash"]
        assert json.loads(arms["E"]["text"]) == {"prompt": arms["T"]["text"]}
        assert arms["E"]["renderer_version"] == "envelope-json-v1"
        assert arms["H-gold"]["renderer_version"] == RENDERER_VERSION


def test_three_condition_plan_is_deterministic_and_seed_dependent() -> None:
    first = request_plan(core_variants(), [0], 1, conditions=("T", "H-gold", "E"))
    assert first == request_plan(
        list(reversed(core_variants())), [0], 1, conditions=("T", "H-gold", "E")
    )
    assert first != request_plan(core_variants(), [0], 2, conditions=("T", "H-gold", "E"))
    assert first == request_plan(core_variants(), [0], 1, conditions=("E", "T", "H-gold"))


def test_a_plan_with_different_annotation_hashes_per_arm_is_accepted() -> None:
    traditional, hierarchical = render_prompt_variants(canonical(0))
    automatic = replace(
        hierarchical,
        condition="H-auto",
        annotation_hash="a" * 64,
        annotation_version="auto-strict-v1",
        renderer_version="pst-gold-json-v1",
    )
    plan = request_plan([traditional, automatic], [0], 3, conditions=("T", "H-auto"))
    arms = {r["condition"]: r for r in plan}
    assert arms["T"]["text"] == canonical(0).text
    assert arms["T"]["prompt_hash"] == arms["T"]["source_hash"]
    assert arms["H-auto"]["annotation_hash"] != arms["T"]["annotation_hash"]


@pytest.mark.parametrize(
    ("conditions", "message"),
    [
        (("T",), "at least two distinct"),
        (("T", "T"), "at least two distinct"),
        (("T", "H"), "unregistered condition"),
        (("T", "E"), "each task requires exactly the conditions"),
        (("T", "H-gold", "E", "F"), "each task requires exactly the conditions"),
    ],
)
def test_invalid_condition_sets_are_refused(conditions, message) -> None:
    with pytest.raises(ValueError, match=message):
        request_plan(core_variants(2), [0], 1, conditions=conditions)


def test_a_missing_arm_is_refused() -> None:
    variants = [v for v in core_variants(2) if (v.task_id, v.condition) != ("BigCodeBench/1", "E")]
    with pytest.raises(ValueError, match="each task requires exactly the conditions"):
        request_plan(variants, [0], 1, conditions=("T", "H-gold", "E"))


def test_arms_must_share_the_source() -> None:
    variants = core_variants(1)
    other = render_envelope_variant(canonical(5))
    variants[2] = replace(other, task_id="BigCodeBench/0")
    with pytest.raises(ValueError, match="do not share source_hash"):
        request_plan(variants, [0], 1, conditions=("T", "H-gold", "E"))


def test_t_and_h_gold_still_share_their_annotation() -> None:
    traditional, hierarchical = render_prompt_variants(canonical(0))
    altered = replace(hierarchical, annotation_hash="b" * 64)
    with pytest.raises(ValueError, match="do not share annotation_hash"):
        request_plan([traditional, altered], [0], 1)


def test_an_envelope_that_is_not_the_source_is_refused() -> None:
    traditional, hierarchical = render_prompt_variants(canonical(0))
    envelope = render_envelope_variant(canonical(0))
    compact = json.dumps({"prompt": canonical(0).text}, ensure_ascii=False)
    for forged in (
        replace(envelope, text=compact, prompt_hash=sha256(compact)),
        replace(envelope, annotation_hash=traditional.annotation_hash),
        replace(envelope, renderer_version=RENDERER_VERSION),
    ):
        with pytest.raises(ValueError, match="envelope"):
            request_plan(
                [traditional, hierarchical, forged], [0], 1, conditions=("T", "H-gold", "E")
            )


def test_a_three_condition_journal_resumes_and_a_two_condition_one_still_does(tmp_path) -> None:
    def output(request):
        return {"raw_output": "x", "input_tokens": 1, "output_tokens": 1, "finish_reason": "eos"}

    calls = []

    def counted(request):
        calls.append(request["request_id"])
        return output(request)

    core = request_plan(core_variants(2), [0], 5, conditions=("T", "H-gold", "E"))
    first = run_journal(core, {"model": "m"}, tmp_path / "core", output)
    assert run_journal(core, {"model": "m"}, tmp_path / "core", counted) == first
    pair = request_plan(gold_variants(2), [0], 5)
    written = run_journal(pair, {"model": "m"}, tmp_path / "pair", output)
    manifest = json.loads((tmp_path / "pair" / "empirical-generate_manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert run_journal(pair, {"model": "m"}, tmp_path / "pair", counted) == written
    assert calls == []


# --- Paired analysis from a ledger with more conditions ------------------------------------

OUTCOMES = {
    "T": ["pass", "fail", "fail", "pass", "fail", "missing"],
    "H-gold": ["pass", "pass", "pass", "fail", "pass", "pass"],
    "E": ["fail", "pass", "fail", "fail", "pass", "pass"],
}


def ledger(conditions=("T", "H-gold", "E")) -> list[OutcomeRecord]:
    return [
        OutcomeRecord(
            "core",
            "functional",
            str(i),
            condition,
            0,
            outcome,
            reason="harness_unavailable" if outcome == "missing" else "",
        )
        for condition in conditions
        for i, outcome in enumerate(OUTCOMES[condition])
    ]


def design(pair=("T", "H-gold")) -> AnalysisDesign:
    return AnalysisDesign(
        "core", "functional", tuple(TaskSpec(str(i)) for i in range(6)), (0,), pair
    )


# Only the pairs with T: the one missing cell is in T, so filtering the ledger by hand
# keeps the same complete tasks (every contrast drops the tasks with a missing arm).
@pytest.mark.parametrize("pair", [("T", "H-gold"), ("T", "E")])
def test_a_pair_from_a_three_condition_ledger_equals_a_ledger_filtered_by_hand(pair) -> None:
    full = analyze_paired(
        design(pair), ledger(), bootstrap_samples=200, ledger_conditions=("T", "H-gold", "E")
    )
    filtered = analyze_paired(design(pair), ledger(pair), bootstrap_samples=200)
    expected = filtered.as_dict()
    expected.update(ledger_conditions=("T", "H-gold", "E"))
    (ignored,) = set(("T", "H-gold", "E")) - set(pair)
    expected.update(ignored_records={ignored: 6})
    assert full.as_dict() == expected


def test_the_known_contrasts_of_the_fixture() -> None:
    options = {"bootstrap_samples": 200, "ledger_conditions": ("T", "H-gold", "E")}
    h1 = analyze_paired(design(("T", "H-gold")), ledger(), **options)
    assert h1.included_tasks == 5 and h1.excluded_tasks == {"5": ["T/0: harness_unavailable"]}
    assert h1.mcnemar_table == {
        "both_fail": 0,
        "baseline_only": 1,
        "treatment_only": 3,
        "both_pass": 1,
    }
    assert h1.delta == pytest.approx(0.4)
    # The task with T missing leaves every contrast of the ledger.
    envelope = analyze_paired(design(("E", "H-gold")), ledger(), **options)
    assert envelope.included_tasks == 5 and envelope.delta == pytest.approx(2 / 5)
    assert envelope.excluded_tasks == {"5": ["T/0: harness_unavailable"]}


def test_without_ledger_conditions_extra_rows_are_still_refused() -> None:
    with pytest.raises(ValueError, match="Unexpected record key"):
        analyze_paired(design(), ledger(), bootstrap_samples=200)
    report = analyze_paired(design(), ledger(("T", "H-gold")), bootstrap_samples=200)
    assert "ignored_records" not in report.as_dict()
    assert "ledger_conditions" not in report.as_dict()


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        (("T", "E"), "ledger conditions must include"),
        (("T", "H-gold", "T"), "distinct"),
        (("T", "H-gold", ""), "non-empty"),
    ],
)
def test_invalid_ledger_conditions_are_refused(declared, message) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_paired(design(), ledger(), bootstrap_samples=200, ledger_conditions=declared)


def test_an_undeclared_ledger_condition_is_still_refused() -> None:
    with pytest.raises(ValueError, match="Unexpected record key"):
        analyze_paired(design(), ledger(), bootstrap_samples=200, ledger_conditions=("T", "H-gold"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "Duplicate record key"),
        ("unknown_task", "Unexpected record key"),
        ("scope", "Mixed analysis scope"),
    ],
)
def test_ignored_rows_are_still_checked(mutation, message) -> None:
    records = ledger()
    extra = next(r for r in records if r.condition == "E")
    if mutation == "duplicate":
        records.append(extra)
    elif mutation == "unknown_task":
        records.append(replace(extra, task_id="99"))
    else:
        records.append(replace(extra, analysis_id="other"))
    with pytest.raises(ValueError, match=message):
        analyze_paired(
            design(), records, bootstrap_samples=200, ledger_conditions=("T", "H-gold", "E")
        )


# --- Commands ------------------------------------------------------------------------------


def test_render_with_the_envelope_writes_three_variants_per_task(tmp_path) -> None:
    source = tmp_path / "canonical.jsonl"
    rows = []
    for i in (0, 1):
        prompt = canonical(i)
        rows.append(
            {
                "task_id": prompt.task_id,
                "text": prompt.text,
                "annotation_version": prompt.annotation_version,
                "annotations": [vars(span) for span in prompt.annotations],
            }
        )
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    plain = CliRunner().invoke(app, ["render", str(source), str(tmp_path / "pair.jsonl")])
    assert plain.exit_code == 0, plain.exception
    full = CliRunner().invoke(
        app, ["render", str(source), str(tmp_path / "core.jsonl"), "--envelope"]
    )
    assert full.exit_code == 0, full.exception
    pair = (tmp_path / "pair.jsonl").read_text(encoding="utf-8").split("\n")[:-1]
    core = (tmp_path / "core.jsonl").read_text(encoding="utf-8").split("\n")[:-1]
    assert [json.loads(line)["condition"] for line in pair] == ["T", "H-gold"] * 2
    assert [json.loads(line)["condition"] for line in core] == ["T", "H-gold", "E"] * 2
    assert core[0:2] == pair[0:2] and core[3:5] == pair[2:4]
    assert json.loads(core[2]) == render_envelope_variant(canonical(0)).as_dict()


def test_analyze_command_accepts_ledger_conditions(tmp_path) -> None:
    write_outcomes(ledger(), tmp_path / "outcomes.jsonl")
    payload = {
        "analysis_id": "core",
        "endpoint": "functional",
        "tasks": [{"task_id": str(i)} for i in range(6)],
        "conditions": ["T", "E"],
    }
    (tmp_path / "design.json").write_text(json.dumps(payload), encoding="utf-8")
    arguments = ["analyze", str(tmp_path / "design.json"), str(tmp_path / "outcomes.jsonl")]
    refused = CliRunner().invoke(app, [*arguments, str(tmp_path / "a" / "report.json")])
    assert isinstance(refused.exception, ValueError)
    result = CliRunner().invoke(
        app,
        [*arguments, str(tmp_path / "b" / "report.json"), "--ledger-conditions", "T,H-gold,E"],
    )
    assert result.exit_code == 0, result.exception
    report = json.loads((tmp_path / "b" / "report.json").read_text(encoding="utf-8"))
    assert report["conditions"] == ["T", "E"]
    assert report["ignored_records"] == {"H-gold": 6}
    manifest = json.loads((tmp_path / "b" / "empirical-analysis_manifest.json").read_text())
    assert manifest["ledger_conditions"] == ["T", "H-gold", "E"]
