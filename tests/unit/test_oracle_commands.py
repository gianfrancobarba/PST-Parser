"""Commands of the two oracles: BigCodeBench harness and CodeQL, with fake containers."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import test_bigcodebench_adapter as bcb
import test_codeql_oracle as sec
from typer.testing import CliRunner

from pstparser.empirical import bigcodebench, codeql
from pstparser.empirical.analysis import read_outcomes
from pstparser.empirical.bigcodebench import write_harness_dataset
from pstparser.empirical.cli import app
from pstparser.empirical.codeql import read_cards
from pstparser.repro.manifest import digest_directory


@pytest.fixture
def parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return bcb.pinned_parquet(tmp_path, monkeypatch)


@pytest.fixture
def nltk_dir(tmp_path: Path) -> Path:
    return bcb.nltk_data(tmp_path)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return sec.fake_bundle(tmp_path)


# ------------------------------------------------------------------ check-bigcodebench


def command(
    tmp_path: Path,
    parquet: Path,
    nltk_dir: Path,
    sample_tasks: list[str],
    excluded: list[str],
    **calibrated_with: Any,
) -> list[str]:
    reference = tmp_path / "calibration-data" / "BigCodeBench.jsonl"
    harness = {
        "image": bcb.IMAGE,
        "dataset_sha256": write_harness_dataset(parquet, bcb.REVISION, bcb.SPLIT, reference),
        "nltk_sha256": digest_directory(nltk_dir),
        "parallel": 8,
        "calibrated": True,
        "network": "none",
        **calibrated_with,
    }
    journal = tmp_path / "generations.jsonl"
    journal.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bcb.planned()),
        encoding="utf-8",
    )
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps({"tasks": [{"task_id": t} for t in sample_tasks]}), encoding="utf-8"
    )
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(
        json.dumps({"harness": harness, "excluded": [{"task_id": task} for task in excluded]}),
        encoding="utf-8",
    )
    return [
        "check-bigcodebench",
        str(journal),
        "--parquet",
        str(parquet),
        "--revision",
        bcb.REVISION,
        "--split",
        bcb.SPLIT,
        "--image",
        bcb.IMAGE,
        "--nltk-dir",
        str(nltk_dir),
        "--sample",
        str(sample),
        "--exclusions",
        str(exclusions),
        "--directory",
        str(tmp_path / "check"),
        "--analysis-id",
        "pilot-01",
        "--order-seed",
        "7",
        # The fixture has one discordant program out of seven (C-15: accepted explicitly).
        "--accept-discordance",
    ]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> bcb.FakeHarness:
    harness = bcb.FakeHarness()
    monkeypatch.setattr(bigcodebench, "run_harness", harness)
    monkeypatch.setattr(bigcodebench, "sanitize_in_image", bcb.fake_sanitize)
    monkeypatch.setattr(bigcodebench, "harness_version", lambda image: "0.2.4")
    monkeypatch.setattr(bigcodebench, "docker_resources", lambda: {"cpus": 28})
    return harness


def test_command_writes_the_ledger_the_details_and_the_manifest(
    parquet: Path, nltk_dir: Path, tmp_path: Path, offline: bcb.FakeHarness
) -> None:
    arguments = command(tmp_path, parquet, nltk_dir, bcb.TASKS, [])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "9 generations, 7 executed, 1 discordant" in result.output
    assert "accepted: more than 1%" in result.output
    directory = tmp_path / "check"
    outcomes = read_outcomes(directory / "outcomes.jsonl")
    assert len(outcomes) == 9
    assert sum(o.outcome == "pass" for o in outcomes) == 6
    details = [
        json.loads(line)
        for line in (directory / "details.jsonl").read_text("utf-8").split("\n")
        if line
    ]
    assert {d["extraction_policy"] for d in details} == {"bigcodebench-sanitize"}
    manifest = json.loads((directory / "empirical-check-bcb_manifest.json").read_text("utf-8"))
    assert manifest["harness"]["image"] == bcb.IMAGE and manifest["harness_version"] == "0.2.4"
    assert manifest["extraction_policy"] == "bigcodebench-sanitize"
    assert manifest["rules"] == ["json-response-v2", "prompt-as-data-v2"]
    assert manifest["order_seed"] == 7 and manifest["flaky_tasks"] == [bcb.TASKS[1]]
    assert (
        manifest["outcomes_sha256"]
        == hashlib.sha256((directory / "outcomes.jsonl").read_bytes()).hexdigest()
    )
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


@pytest.mark.parametrize(
    "sample_tasks,excluded,message",
    [
        (bcb.TASKS[:2], [], "do not cover exactly the tasks of the sample"),
        ([*bcb.TASKS, "BigCodeBench/9"], [], "do not cover exactly the tasks of the sample"),
        (bcb.TASKS, [bcb.TASKS[1]], "generations for excluded tasks"),
    ],
)
def test_command_refuses_generations_outside_the_frozen_sample(
    sample_tasks: list[str],
    excluded: list[str],
    message: str,
    parquet: Path,
    nltk_dir: Path,
    tmp_path: Path,
    offline: bcb.FakeHarness,
) -> None:
    result = CliRunner().invoke(app, command(tmp_path, parquet, nltk_dir, sample_tasks, excluded))

    assert result.exit_code != 0
    assert message in str(result.exception)
    assert offline.calls == [] and not (tmp_path / "check").exists()


@pytest.mark.parametrize(
    "calibrated_with,differing",
    [
        ({"parallel": 4}, "['parallel']"),
        ({"image": bcb.IMAGE.replace("a" * 64, "b" * 64)}, "['image']"),
        (
            {"nltk_sha256": "0" * 64, "dataset_sha256": "0" * 64},
            "['dataset_sha256', 'nltk_sha256']",
        ),
        ({"calibrated": False}, "['calibrated']"),
    ],
)
def test_command_refuses_settings_other_than_those_of_the_calibration(
    calibrated_with: dict[str, Any],
    differing: str,
    parquet: Path,
    nltk_dir: Path,
    tmp_path: Path,
    offline: bcb.FakeHarness,
) -> None:
    arguments = command(tmp_path, parquet, nltk_dir, bcb.TASKS, [], **calibrated_with)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert f"differ from the calibration of the exclusions: {differing}" in str(result.exception)
    assert offline.calls == [] and not (tmp_path / "check" / "outcomes.jsonl").exists()


@pytest.mark.parametrize("damage", ["drop", "repeat"])
def test_command_refuses_an_incomplete_or_repeated_journal(
    damage: str, parquet: Path, nltk_dir: Path, tmp_path: Path, offline: bcb.FakeHarness
) -> None:
    arguments = command(tmp_path, parquet, nltk_dir, bcb.TASKS, [])
    journal = Path(arguments[1])
    lines = [line for line in journal.read_text(encoding="utf-8").split("\n") if line]
    # One condition of one task is missing, or one request appears twice.
    lines = lines[:-1] if damage == "drop" else [*lines, lines[0]]
    journal.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert "each condition and replicate exactly once" in str(result.exception)
    assert offline.calls == [] and not (tmp_path / "check").exists()


# ------------------------------------------------- check-security, calibrate-security


@pytest.fixture
def analyser(monkeypatch: pytest.MonkeyPatch) -> sec.FakeAnalyzer:
    analyzer = sec.FakeAnalyzer()
    monkeypatch.setattr(codeql, "analyze_program", analyzer)
    monkeypatch.setattr(codeql, "codeql_version", lambda config: {"version": "2.22.1"})
    return analyzer


def security_command(tmp_path: Path, bundle: Path, measure: str, *extra: str) -> list[str]:
    journal = tmp_path / "generations.jsonl"
    if not journal.exists():
        rows = [{**g, "task_id": g["task_id"].replace("LSE-", "LSE-00")} for g in sec.planned()]
        # Tasks of three known families of the sample.
        tasks = {"LSE-001": "LSE-141", "LSE-002": "LSE-107", "LSE-003": "LSE-039"}
        for row in rows:
            row["task_id"] = tasks[row["task_id"]]
        journal.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    # A frozen calibration that backs every card (C-16, R-ORA-10).
    calibration = tmp_path / "codeql-calibration.json"
    if not calibration.exists():
        cards = read_cards(sec.CARDS)
        document = {
            "settings": codeql.read_config(sec.SETTINGS, bundle).as_dict(),
            "families": {
                family: {"declared": card.state, "observed": card.state}
                for family, card in cards.items()
            },
            "mismatches": [],
        }
        calibration.write_text(json.dumps(document), encoding="utf-8")
    return [
        "check-security",
        str(journal),
        "--codeql",
        str(sec.SETTINGS),
        "--bundle-dir",
        str(bundle),
        "--cards",
        str(sec.CARDS),
        "--calibration",
        str(calibration),
        "--id-map",
        str(sec.ID_MAP),
        "--measure",
        measure,
        "--directory",
        str(tmp_path / measure),
        "--programs",
        str(tmp_path / "programs"),
        "--analysis-id",
        f"pilot-{measure}",
        "--order-seed",
        "7",
        *extra,
    ]


def test_check_command_writes_ledger_details_and_manifest(
    tmp_path: Path, bundle: Path, analyser: sec.FakeAnalyzer
) -> None:
    records = {r["task_id"]: r for r in json.loads(sec.ID_MAP.read_text("utf-8"))["records"]}
    families = [records[task]["family_id"] for task in ("LSE-141", "LSE-107", "LSE-039")]
    cards = read_cards(sec.CARDS)
    # One calibrated SQL family, one command family with false positives, one without query.
    assert [cards[f].state for f in families] == [
        "calibrated",
        "calibrated-with-false-positives",
        "no-query",
    ]
    assert families == ["89-SQI-1", "78-INS-1", "200-ESI-3"]

    result = CliRunner().invoke(app, security_command(tmp_path, bundle, "general"))

    assert result.exit_code == 0, result.output
    assert "6 generations, 5 analysed" in result.output
    outcomes = read_outcomes(tmp_path / "general" / "outcomes.jsonl")
    assert len(outcomes) == 6 and {o.analysis_id for o in outcomes} == {"pilot-general"}
    manifest = json.loads(
        (tmp_path / "general" / "empirical-check-security_manifest.json").read_text("utf-8")
    )
    assert manifest["measure"] == "general" and manifest["family_states"] is None
    assert manifest["codeql"]["cli_version"] == "2.22.1" and manifest["order_seed"] == 7
    assert manifest["oracle_version"].endswith("python-code-scanning/remote/usability-rules-v2")
    assert manifest["cards_sha256"] == hashlib.sha256(sec.CARDS.read_bytes()).hexdigest()
    assert (
        manifest["outcomes_sha256"]
        == hashlib.sha256((tmp_path / "general" / "outcomes.jsonl").read_bytes()).hexdigest()
    )

    targeted = CliRunner().invoke(app, security_command(tmp_path, bundle, "targeted"))
    assert targeted.exit_code == 0, targeted.output
    assert len(read_outcomes(tmp_path / "targeted" / "outcomes.jsonl")) == 2
    wider = security_command(tmp_path, bundle, "targeted", "--sensitivity")
    wider[wider.index("--directory") + 1] = str(tmp_path / "targeted-fp")
    assert CliRunner().invoke(app, wider).exit_code == 0
    assert len(read_outcomes(tmp_path / "targeted-fp" / "outcomes.jsonl")) == 4
    manifest = json.loads(
        (tmp_path / "targeted-fp" / "empirical-check-security_manifest.json").read_text("utf-8")
    )
    assert manifest["family_states"] == ["calibrated", "calibrated-with-false-positives"]

    again = CliRunner().invoke(app, security_command(tmp_path, bundle, "general"))
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


@pytest.mark.parametrize(
    "measure,extra,message",
    [
        ("bandit", (), "measure must be one of"),
        ("general", ("--sensitivity",), "applies to the targeted measure only"),
    ],
)
def test_check_command_refuses_unknown_measures(
    tmp_path: Path,
    bundle: Path,
    analyser: sec.FakeAnalyzer,
    measure: str,
    extra: tuple[str, ...],
    message: str,
) -> None:
    result = CliRunner().invoke(app, security_command(tmp_path, bundle, measure, *extra))

    assert result.exit_code != 0 and message in str(result.exception)
    assert analyser.order == [] and not (tmp_path / measure).exists()


def test_check_command_refuses_tasks_outside_the_identifier_map(
    tmp_path: Path, bundle: Path, analyser: sec.FakeAnalyzer
) -> None:
    arguments = security_command(tmp_path, bundle, "general")
    journal = Path(arguments[1])
    journal.write_text(
        json.dumps(sec.generation("LSE-999", "T", sec.fenced(sec.SAFE_SQL))) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert "outside the identifier map: ['LSE-999']" in str(result.exception)
    assert analyser.order == []


def test_calibration_command_reports_and_fails_on_a_mismatch(
    tmp_path: Path, bundle: Path, analyser: sec.FakeAnalyzer
) -> None:
    # The fake analyser reports nothing for the real controls: every family
    # with controls is observed as excluded, which the command must not hide.
    arguments = [
        "calibrate-security",
        "--codeql",
        str(sec.SETTINGS),
        "--bundle-dir",
        str(bundle),
        "--cards",
        str(sec.CARDS),
        "--directory",
        str(tmp_path / "calibration"),
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 1
    assert "47 controls; target queries report 0 of 21 vulnerable controls" in result.output
    assert "families whose observed state differs" in result.output
    report = json.loads((tmp_path / "calibration" / "calibration.json").read_text("utf-8"))
    assert len(report["controls"]) == 47
    assert len(report["mismatches"]) == 19
    manifest = json.loads(
        (tmp_path / "calibration" / "empirical-calibrate-security_manifest.json").read_text("utf-8")
    )
    assert manifest["controls"] == 47 and manifest["secure_samples_included"] is False
    assert (
        manifest["calibration_sha256"]
        == hashlib.sha256((tmp_path / "calibration" / "calibration.json").read_bytes()).hexdigest()
    )


def test_check_command_refuses_an_incomplete_journal(
    tmp_path: Path, bundle: Path, analyser: sec.FakeAnalyzer
) -> None:
    arguments = security_command(tmp_path, bundle, "general")
    journal = Path(arguments[1])
    lines = [line for line in journal.read_text(encoding="utf-8").split("\n") if line]
    journal.write_text("".join(line + "\n" for line in lines[:-1]), encoding="utf-8")

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert "each condition and replicate exactly once" in str(result.exception)
    assert analyser.order == [] and not (tmp_path / "general").exists()
