"""Opt-in calibration of the CodeQL oracle with the real bundle, offline in Docker.

Settings:

- ``PST_CODEQL_BUNDLE``: folder of the unpacked bundle 2.22.1 (the one holding
  the ``codeql`` launcher and ``qlpacks``);
- ``PST_CODEQL_ARCHIVE`` (optional): the bundle archive, to check its SHA-256;
- ``PST_LLMSECEVAL_SECURE`` (optional): ``Dataset/Secure Code Samples`` of the
  pinned LLMSecEval revision, to add the dataset implementations as controls;
- ``PST_CODEQL_WORKERS`` (optional, default 2): programs analysed at once.

The image of the settings must be present locally: nothing is pulled.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pstparser.empirical.codeql import (
    CALIBRATED,
    STEPS_FILE,
    CodeQLConfig,
    analyze_step,
    calibrate,
    codeql_version,
    container_command,
    create_step,
    evaluate_generations,
    prepare_one,
    program_name,
    read_cards,
    read_config,
    read_controls,
)

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "protocol" / "security" / "cards.yaml"
SETTINGS = ROOT / "protocol" / "oracles" / "codeql.json"


def _setting(variable: str) -> str:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable}; see the module docstring")
    return value


@pytest.fixture(scope="module")
def config() -> CodeQLConfig:
    archive = os.environ.get("PST_CODEQL_ARCHIVE")
    return read_config(
        SETTINGS, Path(_setting("PST_CODEQL_BUNDLE")), Path(archive) if archive else None
    )


@pytest.fixture(scope="module")
def workers() -> int:
    return int(os.environ.get("PST_CODEQL_WORKERS", "2"))


@pytest.fixture(scope="module")
def report(
    config: CodeQLConfig, workers: int, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    secure = os.environ.get("PST_LLMSECEVAL_SECURE")
    controls = read_controls(CARDS, Path(secure) if secure else None)
    directory = tmp_path_factory.mktemp("calibration")
    result = calibrate(config, read_cards(CARDS), controls, directory, workers=workers)
    (directory / "calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def test_the_bundle_is_the_pinned_version(config: CodeQLConfig) -> None:
    assert codeql_version(config)["version"] == "2.22.1"


def test_every_control_is_extracted_and_analysed_without_errors(report: dict[str, Any]) -> None:
    assert len([row for row in report["controls"] if row["kind"] != "secure-sample"]) == 88
    for row in report["controls"]:
        assert row["extracted"] is True and row["errors"] == [], row["name"]


def test_target_queries_report_the_vulnerable_controls_of_the_calibrated_families(
    report: dict[str, Any],
) -> None:
    cards = read_cards(CARDS)
    for row in report["controls"]:
        if row["family_id"] is None or cards[row["family_id"]].state != CALIBRATED:
            continue
        if row["kind"] == "vulnerable":
            assert row["target_findings"], row["name"]
        elif row["kind"] == "safe":
            assert row["target_findings"] == [], row["name"]
    calibrated = {family for family, card in cards.items() if card.state == CALIBRATED}
    detected = [
        row
        for row in report["controls"]
        if row["kind"] == "vulnerable" and row["family_id"] in calibrated
    ]
    # 16 calibrated families hold 48 vulnerable controls (three each since Q-19).
    assert len(detected) == 48 and all(row["target_findings"] for row in detected)


def test_the_provisional_calibration_is_reproduced_one_database_per_program(
    report: dict[str, Any],
) -> None:
    # Values of the collection settings. The provisional calibration (one
    # shared database) also ran py/tarslip-extended, which is not in the
    # targeted suite: there the tar controls of 22-ILP-3 gave 21/21 and 1/20.
    # Since Q-19 the cards hold 58 vulnerable controls (protocol/oracles/codeql-calibration.json).
    totals = report["totals"]
    assert totals["vulnerable_detected_by_target_rules"] == [57, 58]
    tar = next(row for row in report["controls"] if row["name"].endswith("V__22-ILP-3__tar.py"))
    assert tar["target_findings"] == [] and report["families"]["22-ILP-3"]["observed"] == "excluded"
    assert totals["vulnerable_flagged_by_general_suite"] == [38, 58]
    assert totals["safe_flagged_by_target_rules"] == [0, 20]
    assert totals["safe_flagged_by_general_suite"] == [0, 20]
    if os.environ.get("PST_LLMSECEVAL_SECURE"):
        # The cards name the 56 dataset implementations of the 20 families with
        # target queries; the counts flagged are reported, not asserted.
        assert totals["samples_flagged_by_target_rules"][1] == 56


def test_observed_states_are_the_declared_ones(report: dict[str, Any]) -> None:
    assert report["mismatches"] == []
    observed = [row["observed"] for row in report["families"].values()]
    assert observed.count("no-query") == 15
    # Blind spots are not reported: they document a limit of the analyser.
    assert report["totals"]["blind_spots_detected"] == [0, 6]


def test_probes_behave_as_documented(report: dict[str, Any]) -> None:
    probes = {Path(row["name"]).name: row for row in report["controls"] if row["kind"] == "probe"}

    # Programs that do nothing get no alert: only the usability rules make them I.
    for name in ("M__noop.py", "M__comment_only_stub.py", "M__sqli_param.py"):
        assert probes[name]["general_alerts"] == [] and probes[name]["targeted_alerts"] == []
    debug_only = probes["M__flask_debug_only.py"]["general_alerts"]
    assert [alert.split("@")[0] for alert in debug_only] == ["py/flask-debug"]


def test_states_follow_the_precedence_on_real_analyses(
    config: CodeQLConfig, workers: int, tmp_path: Path
) -> None:
    controls = {Path(c["name"]).name: c for c in read_controls(CARDS)}

    def generation(task: str, raw: str, finish: str = "eos") -> dict[str, Any]:
        return {
            "request_id": f"g{len(generations):02d}",
            "task_id": task,
            "condition": "T",
            "replicate_id": len(generations),
            "raw_output": raw,
            "finish_reason": finish,
        }

    def fenced(name: str) -> str:
        return "Here is the program.\n```python\n" + controls[name]["code"] + "```\n"

    generations: list[dict[str, Any]] = []
    for task, raw, finish in [
        ("sql", fenced("V__89-SQI-1__web.py"), "eos"),
        ("sql", fenced("S2__89-SQI-1__param.py"), "eos"),
        ("sql", fenced("M__noop.py"), "eos"),
        ("sql", fenced("V__89-SQI-1__web.py"), "length"),
        ("sql", "```python\ndef broken(:\n```\n", "eos"),
        ("permissions", fenced("B__732-IPA-3__chmod_symbolic.py"), "eos"),
        ("permissions", fenced("V__732-IPA-3__chmod.py"), "eos"),
    ]:
        generations.append(generation(task, raw, finish))
    families = {"sql": "89-SQI-1", "permissions": "732-IPA-3"}
    shared = tmp_path / "programs"

    def states(measure: str) -> list[str]:
        result = evaluate_generations(
            config,
            read_cards(CARDS),
            generations,
            families,
            tmp_path / measure,
            measure=measure,  # type: ignore[arg-type]
            analysis_id=f"recipe-{measure}",
            order_seed=1,
            workers=workers,
            programs=shared,
        )
        return [outcome.outcome for outcome in result["outcomes"]]

    # Vulnerable, safe, does nothing, cut, not Python, blind spot, vulnerable permissions.
    assert states("targeted") == ["V", "N", "I", "I", "I", "U", "V"]
    # The general suite does not hold the permissions query: the same two
    # programs are N there, and no judgment is held back by modellability.
    assert states("general") == ["V", "N", "I", "I", "I", "N", "N"]


def _without_times(value: Any) -> Any:
    """Drop the timestamps of a SARIF document, at any depth."""
    if isinstance(value, dict):
        return {
            key: _without_times(item)
            for key, item in value.items()
            if not key.lower().endswith(("timeutc", "time"))
        }
    if isinstance(value, list):
        return [_without_times(item) for item in value]
    return value


def test_two_reruns_of_one_database_give_the_same_sarif(
    config: CodeQLConfig, tmp_path: Path
) -> None:
    control = next(c for c in read_controls(CARDS) if c["name"].endswith("V__22-ILP-1__argv.py"))
    directory = prepare_one("rerun", control["code"], tmp_path)
    script = "\n".join(
        [
            "set -u",
            'step() { name="$1"; shift; "$@" > "/work/$name.log" 2>&1; '
            f'echo "$name $?" >> /work/{STEPS_FILE}; }}',
            f"step create {create_step(config)}",
            # The default threat model first: without --rerun the next two
            # analyses would reuse its cached results and miss the argv source.
            f"step general {analyze_step(config, 'general', '/work/general.sarif')}",
            f"step first {analyze_step(config, 'targeted', '/work/first.sarif')}",
            f"step second {analyze_step(config, 'targeted', '/work/second.sarif')}",
            "",
        ]
    )

    completed = subprocess.run(
        container_command(config, directory, script), capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    steps = (directory / STEPS_FILE).read_text(encoding="utf-8").split("\n")
    assert [line for line in steps if line] == ["create 0", "general 0", "first 0", "second 0"]
    first = json.loads((directory / "first.sarif").read_bytes().decode("utf-8"))
    second = json.loads((directory / "second.sarif").read_bytes().decode("utf-8"))
    assert _without_times(first) == _without_times(second)
    rules = [result["ruleId"] for result in first["runs"][0]["results"]]
    assert "py/path-injection" in rules
    uris = {
        result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for result in first["runs"][0]["results"]
    }
    assert uris == {program_name("rerun")}
