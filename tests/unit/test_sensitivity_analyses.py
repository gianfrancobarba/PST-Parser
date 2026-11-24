"""The prespecified sensitivity analyses kept by the user, each with a test.

Leave-one-family-out and the equally weighted family mean (with the 2 pp
rule) for the security endpoint; S-FLAKY and "fail if the first two runs
disagree" for correctness; the four-way reading at 3 and 10 pp; LOC and
findings per 100 LOC; the 5% determinism check (selection and comparison).
"""

import json
import math
import statistics
from pathlib import Path

import pytest
from scipy import stats
from typer.testing import CliRunner

from pstparser.empirical.analysis import AnalysisDesign, OutcomeRecord, TaskSpec, analyze_paired
from pstparser.empirical.cli import app
from pstparser.empirical.tables import loc_tables, paired_tables

# --- Security: leave-one-family-out and Delta_fam ------------------------------------------

FAMILIES = {"A": [1, 1, 0], "B": [0, -1], "C": [1, 0, 0, 0]}  # task differences H - T


def security_report():
    tasks, records = [], []
    for family, differences in FAMILIES.items():
        for i, difference in enumerate(differences):
            task = f"{family}{i}"
            tasks.append(TaskSpec(task, family))
            t, h = ("V", "N") if difference == 1 else ("N", "V") if difference == -1 else ("N", "N")
            records += [
                OutcomeRecord("sec", "security", task, "T", 0, t),
                OutcomeRecord("sec", "security", task, "H-gold", 0, h),
            ]
    design = AnalysisDesign("sec", "security", tuple(tasks))
    return analyze_paired(design, records, bootstrap_samples=50)


def test_leave_one_family_out_and_the_family_mean_by_hand() -> None:
    report = security_report()
    family = report.family_sensitivity
    # Outcome N counts 1 (not vulnerable), V counts 0: the difference is H - T in N.
    means = {f: statistics.fmean(d) for f, d in FAMILIES.items()}
    assert family["delta_fam"] == pytest.approx(statistics.fmean(means.values()))
    se = statistics.stdev(means.values()) / math.sqrt(3)
    q = stats.t.ppf(0.975, 2)
    assert family["delta_fam_ci"] == pytest.approx(
        (family["delta_fam"] - q * se, family["delta_fam"] + q * se)
    )
    every = [d for values in FAMILIES.values() for d in values]
    assert report.delta == pytest.approx(statistics.fmean(every))
    assert family["delta_fam_differs_2pp"] is (abs(family["delta_fam"] - report.delta) > 0.02)
    without = {
        f: statistics.fmean([d for g, v in FAMILIES.items() if g != f for d in v]) for f in FAMILIES
    }
    assert family["leave_one_family_out"] == pytest.approx(without)
    assert (family["lofo_min"], family["lofo_max"]) == pytest.approx(
        (min(without.values()), max(without.values()))
    )


# --- Correctness: S-FLAKY --------------------------------------------------------------------


def test_s_flaky_excludes_flaky_tasks_and_fails_discordant_programs() -> None:
    pairs = [("fail", "pass"), ("pass", "pass"), ("fail", "fail"), ("fail", "pass")]
    tasks = tuple(TaskSpec(f"t{i}") for i in range(4))
    records = [
        OutcomeRecord("c", "functional", f"t{i}", cond, 0, out)
        for i, pair in enumerate(pairs)
        for cond, out in zip(("T", "H-gold"), pair, strict=True)
    ]
    # t0 H-gold passed only by majority (pass, fail, pass); t1 T was stable.
    details = [
        {
            "task_id": f"t{i}",
            "condition": cond,
            "replicate_id": 0,
            "flaky": False,
            "run_statuses": [out, out],
        }
        for i, pair in enumerate(pairs)
        for cond, out in zip(("T", "H-gold"), pair, strict=True)
    ]
    details[1] = {**details[1], "flaky": True, "run_statuses": ["pass", "fail", "pass"]}
    report = analyze_paired(
        AnalysisDesign("c", "functional", tasks), records, bootstrap_samples=50, details=details
    )
    assert report.delta == pytest.approx(2 / 4)
    # Without t0: differences 0, 0, 1.
    assert report.sensitivity["s_flaky_excluded"] == pytest.approx(1 / 3)
    # t0 H-gold becomes a fail: differences 0, 0, 0, 1.
    assert report.sensitivity["s_flaky_fail_if_discordant"] == pytest.approx(1 / 4)


# --- Readings at 3 and 10 pp -----------------------------------------------------------------


def test_the_four_way_reading_is_also_given_at_3_and_10_pp() -> None:
    report = {
        "analysis_id": "x",
        "endpoint": "functional",
        "conditions": ["T", "E"],
        "planned_tasks": 800,
        "included_tasks": 800,
        "mcnemar_table": None,
        "delta": 0.04,
        "midp_p": 0.001,
        "mcnemar_p": None,
        "tango_ci": None,
        "tango_ci_90": [0.02, 0.06],
        "newcombe_ci": None,
        "conservative_ci": None,
        "signflip_p": None,
        "cr1_ci": None,
        "sensitivity": {},
    }
    (main, _) = paired_tables({"a": report})
    row = dict(zip(main.columns, main.rows[0], strict=True))
    assert (row["reading_3pp"], row["reading_5pp"], row["reading_10pp"]) == ("E1", "E1", "E2")
    assert (row["tost_3pp"], row["tost_10pp"]) == ("no", "yes")


# --- LOC and findings per 100 LOC ------------------------------------------------------------


def test_loc_table_by_hand() -> None:
    details = [
        {
            "task_id": "L1",
            "family_id": "F",
            "condition": "T",
            "replicate_id": 0,
            "usable": True,
            "loc": 10,
            "alerts": ["a@1"],
        },
        {
            "task_id": "L1",
            "family_id": "F",
            "condition": "H-gold",
            "replicate_id": 0,
            "usable": True,
            "loc": 20,
            "alerts": ["a@1", "b@2"],
        },
        {
            "task_id": "L2",
            "family_id": "G",
            "condition": "T",
            "replicate_id": 0,
            "usable": True,
            "loc": 5,
            "alerts": [],
        },
        {
            "task_id": "L2",
            "family_id": "G",
            "condition": "H-gold",
            "replicate_id": 0,
            "usable": False,
            "loc": 0,
            "alerts": [],
        },
    ]
    (table,) = loc_tables(details)
    rows = {row[0]: row for row in table.rows}
    # Usable programs only: T has 15 LOC and 1 finding, H-gold 20 LOC and 2 findings.
    assert rows["T"][1:] == (2, 7.5, 1, pytest.approx(100 / 15))
    assert rows["H-gold"][1:] == (1, 20.0, 2, pytest.approx(10.0))


# --- Determinism at 5% -----------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def test_determinism_selects_whole_tasks_and_compares_byte_for_byte(tmp_path: Path) -> None:
    tasks = [f"BigCodeBench/{i}" for i in range(30)]
    variants = write_jsonl(
        tmp_path / "variants.jsonl",
        [{"task_id": t, "condition": c, "text": f"{t}{c}"} for t in tasks for c in ("T", "H-gold")],
    )
    journal = write_jsonl(
        tmp_path / "run" / "generations.jsonl",
        [
            {
                "request_id": f"{t}-{c}",
                "task_id": t,
                "condition": c,
                "replicate_id": 0,
                "raw_output": f"out {t} {c}",
            }
            for t in tasks
            for c in ("T", "H-gold")
        ],
    )
    out = tmp_path / "det" / "variants.jsonl"
    result = CliRunner().invoke(
        app,
        ["determinism-sample", str(variants), str(journal), "--seed", "7319", "--out", str(out)],
    )
    assert result.exit_code == 0, result.exception
    chosen = [json.loads(line) for line in out.read_text(encoding="utf-8").split("\n") if line]
    picked = {row["task_id"] for row in chosen}
    # 60 greedy requests: at least max(20, 5%) = 20 requests, so 10 whole tasks.
    assert len(picked) == 10 and len(chosen) == 20
    again = tmp_path / "det2" / "variants.jsonl"
    CliRunner().invoke(
        app,
        ["determinism-sample", str(variants), str(journal), "--seed", "7319", "--out", str(again)],
    )
    assert again.read_bytes() == out.read_bytes()
    regenerated = [
        {"request_id": f"{t}-{c}", "raw_output": f"out {t} {c}" if t != min(picked) else "other"}
        for t in sorted(picked)
        for c in ("T", "H-gold")
    ]
    regen = write_jsonl(tmp_path / "regen" / "generations.jsonl", regenerated)
    report_path = tmp_path / "cmp" / "determinism.json"
    result = CliRunner().invoke(
        app, ["determinism-compare", str(journal), str(regen), "--out", str(report_path)]
    )
    assert result.exit_code == 0, result.exception
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (report["compared"], report["identical"]) == (20, 18)
    assert report["identical_share"] == pytest.approx(0.9)
    assert sorted(report["differing"]) == [f"{min(picked)}-H-gold", f"{min(picked)}-T"]


# --- S-NEARDUP --------------------------------------------------------------


def test_s_neardup_keeps_one_task_per_group() -> None:
    from pstparser.empirical.contamination import near_duplicate_groups

    prompts = {
        "BigCodeBench/1120": "Same words here.",
        "BigCodeBench/1121": "Same words here.",
        "BigCodeBench/7": "Something else entirely.",
    }
    groups = near_duplicate_groups(prompts)
    assert groups == [["BigCodeBench/1120", "BigCodeBench/1121"]]
    pairs = {
        "BigCodeBench/1120": ("fail", "pass"),
        "BigCodeBench/1121": ("fail", "pass"),
        "BigCodeBench/7": ("pass", "pass"),
    }
    tasks = tuple(TaskSpec(task) for task in pairs)
    records = [
        OutcomeRecord("c", "functional", task, cond, 0, out)
        for task, pair in pairs.items()
        for cond, out in zip(("T", "H-gold"), pair, strict=True)
    ]
    report = analyze_paired(
        AnalysisDesign("c", "functional", tasks),
        records,
        bootstrap_samples=50,
        near_duplicate_groups=groups,
    )
    assert report.delta == pytest.approx(2 / 3)
    assert report.sensitivity["s_neardup_one_per_group"] == pytest.approx(1 / 2)
