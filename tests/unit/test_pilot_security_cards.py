"""Security oracle for the pilot families (declared in ``protocol/deviations.md``).

The 12 LLMSecEval prompts of the pilot come from four C-origin memory families,
disjoint from the S2 sample. They have no target query and no family rule: the
pilot cards mark them ``no-query`` and the usability check applies only the
generic rule to them, so that check-security can measure G9 (general measure)
and the CodeQL time of CP3-R on generated programs. The cards and rules of the
final families are unchanged.
"""

import json
from pathlib import Path

import pytest

from pstparser.empirical.codeql import (
    NO_QUERY,
    CodeQLConfig,
    read_cards,
    require_calibration,
)
from pstparser.empirical.usability import (
    FAMILY_RULES,
    PILOT_GENERIC_FAMILIES,
    USABILITY_VERSION,
    check_usable,
)

ROOT = Path(__file__).resolve().parents[2]
PILOT_CARDS = ROOT / "protocol" / "security" / "pilot-cards.yaml"
CARDS = ROOT / "protocol" / "security" / "cards.yaml"
PILOT_SAMPLE = ROOT / "protocol" / "samples" / "pilot-llmseceval.json"
CALIBRATION = ROOT / "protocol" / "oracles" / "codeql-calibration.json"
CODEQL = ROOT / "protocol" / "oracles" / "codeql.json"

WORKING = """
def copy(buffer, data):
    for index, value in enumerate(data):
        buffer[index] = value
    return buffer

print(copy([0] * 4, [1, 2]))
"""

STUB = """
def copy(buffer, data):
    pass
"""


def pilot_families() -> set[str]:
    sample = json.loads(PILOT_SAMPLE.read_text("utf-8"))
    return set(sample["families"])


def test_the_pilot_cards_hold_exactly_the_pilot_families_as_no_query() -> None:
    cards = read_cards(PILOT_CARDS)
    assert set(cards) == pilot_families() == set(PILOT_GENERIC_FAMILIES)
    assert all(card.state == NO_QUERY and not card.target_rules for card in cards.values())


def test_the_pilot_families_are_outside_the_final_cards_and_rules() -> None:
    assert set(PILOT_GENERIC_FAMILIES).isdisjoint(read_cards(CARDS))
    assert set(PILOT_GENERIC_FAMILIES).isdisjoint(FAMILY_RULES)
    assert USABILITY_VERSION == "usability-rules-v2"


@pytest.mark.parametrize("family", sorted(PILOT_GENERIC_FAMILIES))
def test_a_pilot_family_gets_the_generic_rule_only(family: str) -> None:
    assert check_usable(WORKING, family) == (True, ())
    usable, reasons = check_usable(STUB, family)
    assert not usable and reasons
    assert check_usable("not python (", family) == (False, ("not valid Python",))


def test_a_family_without_any_rule_is_still_refused() -> None:
    with pytest.raises(ValueError, match="no usability rule"):
        check_usable(WORKING, "416-UAF-1")


def test_the_frozen_calibration_backs_the_pilot_cards() -> None:
    calibration = json.loads(CALIBRATION.read_text("utf-8"))
    # The settings only, without binding a local bundle (read_config would check it).
    settings = json.loads(CODEQL.read_text("utf-8"))["settings"]
    config = CodeQLConfig(bundle_dir=ROOT, **settings)
    require_calibration(calibration, config, read_cards(PILOT_CARDS))
