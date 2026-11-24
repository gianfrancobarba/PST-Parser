"""Extension F (pst-flat-json-v1) and the TOST with the four-way reading."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.analysis import TaskEstimate
from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.inference import (
    cluster_interval_at,
    cluster_t_interval,
    four_way_reading,
    tost,
)
from pstparser.empirical.journal import request_plan
from pstparser.empirical.rendering import (
    FLAT_RENDERER_VERSION,
    render_flat_variant,
    render_prompt_variants,
)
from pstparser.empirical.tables import paired_tables
from pstparser.pst.taxonomy import LEAF_PATHS

# Unicode, a quote, a backslash, a tab and a code block with newlines.
TEXT = 'Écris "f".\nUse a\\b\there.\n```python\ndef f():\n    return 1\n```'


def canonical() -> CanonicalPrompt:
    first = TEXT.index("Use")
    code = TEXT.index("```")
    return CanonicalPrompt(
        "BigCodeBench/1",
        TEXT,
        (
            SpanAnnotation("main_instruction", 0, first - 1, TEXT[: first - 1]),
            SpanAnnotation("context.constraints", first, code - 1, TEXT[first : code - 1]),
            SpanAnnotation("context.data", code, len(TEXT), TEXT[code:]),
        ),
        "gold-v2.0",
    )


def test_f_by_hand() -> None:
    flat = render_flat_variant(canonical())
    first = TEXT.index("Use")
    code = TEXT.index("```")
    expected = {
        "prompt": {
            **{path: [] for path in LEAF_PATHS},
            "main_instruction": [{"position": 0, "phrase": TEXT[:first]}],
            "context.constraints": [{"position": 1, "phrase": TEXT[first:code]}],
            "context.data": [{"position": 2, "phrase": TEXT[code:]}],
        }
    }
    ordered = {"prompt": {path: expected["prompt"][path] for path in LEAF_PATHS}}
    assert flat.text == json.dumps(ordered, ensure_ascii=False, indent=2)
    assert list(json.loads(flat.text)["prompt"]) == list(LEAF_PATHS)
    assert (flat.condition, flat.renderer_version) == ("F", FLAT_RENDERER_VERSION)


def test_f_reconstructs_the_source_with_the_pairs_of_h() -> None:
    _, hierarchical = render_prompt_variants(canonical())
    flat = render_flat_variant(canonical())
    tree = json.loads(hierarchical.text)["prompt"]
    pairs_h = sorted(
        (item["position"], item["phrase"])
        for node in tree.values()
        for leaf in (node.values() if isinstance(node, dict) else [node])
        for item in leaf
    )
    pairs_f = sorted(
        (item["position"], item["phrase"])
        for leaf in json.loads(flat.text)["prompt"].values()
        for item in leaf
    )
    assert pairs_f == pairs_h
    assert "".join(phrase for _, phrase in pairs_f) == TEXT


def test_f_escapes_as_h_does() -> None:
    _, hierarchical = render_prompt_variants(canonical())
    flat = render_flat_variant(canonical())
    for item in json.loads(flat.text)["prompt"]["context.data"]:
        literal = json.dumps(item["phrase"], ensure_ascii=False)
        assert literal in flat.text and literal in hierarchical.text
    assert "É" in flat.text and "\\n" in flat.text and '\\"' in flat.text


def test_f_is_deterministic_and_shares_the_annotation_metadata() -> None:
    plain, hierarchical = render_prompt_variants(canonical())
    flat = render_flat_variant(canonical())
    assert flat == render_flat_variant(canonical())
    for field in ("task_id", "source_hash", "annotation_hash", "annotation_version"):
        assert getattr(flat, field) == getattr(hierarchical, field)
    plan = request_plan([plain, hierarchical, flat], [0], 7, ("T", "H-gold", "F"))
    assert sorted(r["condition"] for r in plan) == ["F", "H-gold", "T"]


def test_f_refuses_what_h_refuses() -> None:
    gap = CanonicalPrompt(
        "t", "One. Two.", (SpanAnnotation("main_instruction", 0, 4, "One."),), "gold-v2.0"
    )
    with pytest.raises(ValueError, match="Unannotated"):
        render_flat_variant(gap)


def test_render_writes_f_after_the_hierarchical_variant(tmp_path: Path) -> None:
    prompt = canonical()
    source = tmp_path / "gold.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": prompt.task_id,
                "text": prompt.text,
                "annotation_version": prompt.annotation_version,
                "annotations": [span.__dict__ for span in prompt.annotations],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["render", str(source), str(tmp_path / "v.jsonl"), "--flat"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (tmp_path / "v.jsonl").read_text("utf-8").splitlines()]
    assert [row["condition"] for row in rows] == ["T", "H-gold", "F"]


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ((-0.04, 0.049), True),
        ((-0.05, 0.01), False),  # the margin itself is outside the open interval
        ((-0.01, 0.05), False),
        ((0.01, 0.06), False),
        ((-0.06, -0.051), False),
        ((0.0, 0.0), True),
    ],
)
def test_tost_on_known_intervals(interval: tuple[float, float], expected: bool) -> None:
    assert tost(interval) is expected


def test_tost_with_another_margin_and_invalid_inputs() -> None:
    assert tost((-0.09, 0.09), margin=0.10) and not tost((-0.09, 0.09), margin=0.03)
    with pytest.raises(ValueError):
        tost((0.02, -0.02))
    with pytest.raises(ValueError):
        tost((-0.01, 0.01), margin=0.0)


@pytest.mark.parametrize(
    ("p", "interval", "reading"),
    [
        (0.001, (0.06, 0.12), "E1"),  # different, not equivalent
        (0.001, (0.01, 0.03), "E2"),  # different but negligible
        (0.40, (-0.02, 0.03), "E3"),  # equivalent
        (0.40, (-0.08, 0.09), "E4"),  # inconclusive
        (0.05, (0.0, 0.06), "E1"),  # p = alpha counts as different
    ],
)
def test_four_way_reading(p: float, interval: tuple[float, float], reading: str) -> None:
    assert four_way_reading(p, interval) == reading


def test_the_cr1_interval_at_another_confidence_matches_a_direct_computation() -> None:
    estimates = [
        TaskEstimate(f"t{i}", f"f{i % 4}", 0.0, value, value)
        for i, value in enumerate([0.2, -0.4, 0.6, 0.0, 0.8, -0.2, 0.4, 0.2])
    ]
    at_95 = cluster_t_interval(estimates, confidence=0.95)
    at_90 = cluster_t_interval(estimates, confidence=0.90)
    derived = cluster_interval_at(
        at_95.interval, at_95.standard_error, at_95.degrees_of_freedom, 0.90
    )
    assert derived == pytest.approx(at_90.interval, abs=1e-12)


def report(analysis_id: str, p: float, interval_90: tuple[float, float]) -> dict:
    return {
        "analysis_id": analysis_id,
        "endpoint": "functional",
        "conditions": ["T", "H-gold"],
        "planned_tasks": 800,
        "included_tasks": 800,
        "mcnemar_table": {"baseline_only": 40, "treatment_only": 44},
        "delta": 0.005,
        "midp_p": p,
        "mcnemar_p": p,
        "tango_ci": [-0.02, 0.03],
        "tango_ci_90": list(interval_90),
        "sensitivity": {},
    }


def test_the_paired_table_carries_the_tost_and_the_reading() -> None:
    main, _ = paired_tables(
        {
            "a": report("equivalent", 0.6, (-0.015, 0.025)),
            "b": report("inconclusive", 0.6, (-0.07, 0.08)),
        }
    )
    columns = main.columns
    rows = {row[0]: dict(zip(columns, row, strict=True)) for row in main.rows}
    assert (rows["equivalent"]["tost_5pp"], rows["equivalent"]["reading_5pp"]) == ("yes", "E3")
    assert (rows["inconclusive"]["tost_5pp"], rows["inconclusive"]["reading_5pp"]) == ("no", "E4")
    assert rows["equivalent"]["equivalence90_low"] == -0.015
