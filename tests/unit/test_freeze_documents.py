"""The written parts of the freeze F4 agree with the code and the frozen files.

The gold guidelines, the prompt-as-data rule and the analysis plan are frozen as
text: their versions, thresholds and counts must be those the code and the
frozen files use, and no versioned file may point into the unversioned docs.
"""

import json
import re
from pathlib import Path

import yaml

from pstparser.empirical import annotations, bigcodebench, covariates, usability
from pstparser.empirical.freeze import read_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "protocol" / "freeze-spec.yaml"
GUIDELINES = ROOT / "protocol" / "guidelines" / "gold-v2.0.md"
RULES = ROOT / "protocol" / "rules" / "prompt-as-data.md"
PLAN = ROOT / "protocol" / "analysis" / "plan.md"


def test_the_freeze_lists_the_written_documents_and_the_model_digests() -> None:
    items, _ = read_spec(SPEC)
    expected = {
        "protocol/guidelines/gold-v2.0.md",
        "protocol/rules/prompt-as-data.md",
        "protocol/analysis/plan.md",
        "protocol/model.json",
    }
    assert expected <= set(items)


def test_no_protocol_document_points_into_docs() -> None:
    for path in (ROOT / "protocol").rglob("*.md"):
        assert "docs/" not in path.read_text("utf-8"), path


def test_the_guidelines_carry_the_version_of_the_gold() -> None:
    text = GUIDELINES.read_text("utf-8")
    assert annotations.GUIDELINES_VERSION == "gold-v2.0"
    assert f"`{annotations.GUIDELINES_VERSION}`" in text
    assert covariates.JUDGEMENT_UNITS_RULE in text
    for rule in [f"G{n}" for n in range(1, 5)] + [f"B{n}" for n in range(1, 15)]:
        assert re.search(rf"\b{rule}\b", text), rule


def test_the_rule_document_names_the_rules_the_code_applies() -> None:
    text = RULES.read_text("utf-8")
    assert bigcodebench.JSON_RESPONSE_RULE in text
    assert bigcodebench.PROMPT_AS_DATA_RULE in text
    assert "categorize" in text


def test_the_plan_states_the_frozen_thresholds_and_counts() -> None:
    text = PLAN.read_text("utf-8")
    gate = json.loads((ROOT / "protocol/hauto/gate-thresholds.json").read_text("utf-8"))
    assert f"≤ {gate['max_strict_failure']}" in text
    assert f"≥ {gate['min_macro_f1']:.2f}" in text
    contamination = json.loads((ROOT / "protocol/analysis/contamination.json").read_text("utf-8"))
    assert f"{contamination['n']}-gram" in text
    assert f"threshold {contamination['threshold']}" in text
    assert usability.USABILITY_VERSION in text
    cards = yaml.safe_load((ROOT / "protocol/security/cards.yaml").read_text("utf-8"))
    states = {row["family_id"]: row["state"] for row in cards["families"]}
    ids = json.loads((ROOT / "protocol/samples/llmseceval-ids.json").read_text("utf-8"))
    s2 = [row for row in ids["records"] if "S2" in row["selections"]]
    calibrated = [row for row in s2 if states.get(row["family_id"]) == "calibrated"]
    widened = [
        row
        for row in s2
        if states.get(row["family_id"]) in {"calibrated", "calibrated-with-false-positives"}
    ]
    assert f"{len(s2)} prompts in {len({r['family_id'] for r in s2})} families" in text
    assert f"{len({r['family_id'] for r in calibrated})} `calibrated` families" in text
    assert f"({len(calibrated)} prompts)" in text
    assert f"({len(widened)} prompts, {len({r['family_id'] for r in widened})} families)" in text
    model = json.loads((ROOT / "protocol/model.json").read_text("utf-8"))
    assert model["revision"] in text
