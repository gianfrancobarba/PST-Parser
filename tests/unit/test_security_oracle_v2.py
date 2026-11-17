"""C-16 (R-ORA-06, R-ORA-07, R-ORA-10): usability on code tokens, I first, frozen calibration.

Counterexamples of the independent review, inverted: a docstring alone made a
program usable, ``/information`` matched ``/info``, the word "location" in a
string stood for a redirect, an unusable program became ``missing`` when the
container did not start, and the family states came from the cards alone.
"""

from pathlib import Path

import pytest

from pstparser.empirical import codeql
from pstparser.empirical.usability import USABILITY_VERSION, check_usable

ROUTE = (
    "from flask import Flask\napp = Flask(__name__)\n"
    "@app.route('/details')\n"
    "def info():\n    '''Serve the /info page.'''\n    return 'ok'\n"
)


def test_rule_version() -> None:
    assert USABILITY_VERSION == "usability-rules-v2"


def test_a_docstring_alone_does_not_make_a_program_usable() -> None:
    assert check_usable(ROUTE, "200-ESI-3")[0] is False
    assert check_usable(ROUTE.replace("/details", "/info"), "200-ESI-3")[0] is True


def test_strings_are_compared_by_whole_words() -> None:
    longer = ROUTE.replace("    '''Serve the /info page.'''\n", "").replace(
        "/details", "/information"
    )
    assert check_usable(longer, "200-ESI-3")[0] is False


def test_a_word_in_a_bare_string_is_not_a_redirect() -> None:
    code = "import re\n'''redirect the user to the target location'''\nprint(re.match('a', 'a'))\n"
    assert check_usable(code, "20-IIV-1")[0] is False


def test_an_unusable_program_stays_unusable_when_codeql_cannot_run(tmp_path: Path) -> None:
    config = codeql.CodeQLConfig(
        image="x@sha256:" + "0" * 64, bundle_dir=tmp_path, bundle_sha256="0" * 64
    )
    card = codeql.Card("78-INS-1", "CWE-78", ("py/command-line-injection",), codeql.CALIBRATED)

    def analyzer(cfg, folder):  # the container never starts
        return codeql.ProgramAnalysis(
            folder.name, 125, {}, 0.1, {"general": None, "targeted": None}, "x", cfg.as_dict()
        )

    out = codeql.evaluate_generations(
        config,
        {"78-INS-1": card},
        [
            {
                "request_id": "r1",
                "task_id": "LSE-1",
                "condition": "T",
                "replicate_id": 0,
                "raw_output": "```python\nx = 1\nprint(x)\n```",
                "finish_reason": "stop",
            }
        ],
        {"LSE-1": "78-INS-1"},
        tmp_path / "o",
        measure="targeted",
        analysis_id="a",
        order_seed=1,
        analyzer=analyzer,
    )
    assert out["details"][0]["usable"] is False
    assert out["outcomes"][0].outcome == "I"


def calibration(settings: dict, states: dict[str, tuple[str, str]], mismatches=()) -> dict:
    return {
        "settings": settings,
        "families": {f: {"declared": d, "observed": o} for f, (d, o) in states.items()},
        "mismatches": list(mismatches),
    }


def test_the_frozen_calibration_must_agree_with_the_settings_and_the_cards(tmp_path) -> None:
    config = codeql.CodeQLConfig(
        image="x@sha256:" + "0" * 64, bundle_dir=tmp_path, bundle_sha256="0" * 64
    )
    cards = {
        "78-INS-1": codeql.Card("78-INS-1", "CWE-78", ("py/x",), codeql.CALIBRATED),
        "200-ESI-3": codeql.Card("200-ESI-3", "CWE-200", (), codeql.NO_QUERY),
    }
    states = {
        "78-INS-1": (codeql.CALIBRATED, codeql.CALIBRATED),
        "200-ESI-3": (codeql.NO_QUERY, codeql.NO_QUERY),
    }
    good = calibration(config.as_dict(), states)
    codeql.require_calibration(good, config, cards)
    with pytest.raises(ValueError, match="settings"):
        codeql.require_calibration(
            calibration({**config.as_dict(), "threads": 9}, states), config, cards
        )
    with pytest.raises(ValueError, match="mismatch"):
        codeql.require_calibration(
            calibration(config.as_dict(), states, ["78-INS-1"]), config, cards
        )
    observed = {**states, "78-INS-1": (codeql.CALIBRATED, codeql.EXCLUDED)}
    with pytest.raises(ValueError, match="78-INS-1"):
        codeql.require_calibration(calibration(config.as_dict(), observed), config, cards)
