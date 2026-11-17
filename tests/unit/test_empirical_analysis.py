"""Paired downstream estimands, ledger integrity and uncertainty safeguards."""

from dataclasses import replace

import pytest

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    analyze_paired,
    exact_mcnemar,
    holm_adjust,
    read_outcomes,
    write_outcomes,
)


def fixture(outcomes, *, endpoint="functional", families=None, replicates=(0,)):
    tasks = tuple(TaskSpec(str(i), families[i] if families else None) for i in range(len(outcomes)))
    design = AnalysisDesign("run/model/dataset", endpoint, tasks, replicates)
    records = [
        OutcomeRecord(
            design.analysis_id,
            endpoint,
            str(i),
            condition,
            replicate,
            outcome,
            reason="infrastructure timeout" if outcome == "missing" else "",
        )
        for i, arms in enumerate(outcomes)
        for condition, values in zip(design.conditions, arms, strict=True)
        for replicate, outcome in zip(replicates, values, strict=True)
    ]
    return design, records


def test_known_paired_effect_mcnemar_and_order_independence():
    design, records = fixture([(["fail"], ["pass"])] * 6 + [(["pass"], ["pass"])] * 2)
    report = analyze_paired(design, records, bootstrap_samples=200)
    assert report.delta == 0.75
    assert report.mcnemar_p == pytest.approx(0.03125)
    assert report.mcnemar_table == {
        "both_fail": 0,
        "baseline_only": 0,
        "treatment_only": 6,
        "both_pass": 2,
    }
    assert (
        report.as_dict()
        == analyze_paired(design, list(reversed(records)), bootstrap_samples=200).as_dict()
    )


@pytest.mark.parametrize("mutation", ["duplicate", "absent", "unexpected", "scope", "endpoint"])
def test_rejects_bad_joins(mutation):
    design, records = fixture([(["pass"], ["fail"])])
    if mutation == "duplicate":
        records.append(records[0])
    elif mutation == "absent":
        records.pop()
    elif mutation == "unexpected":
        records[0] = replace(records[0], task_id="extra")
    elif mutation == "scope":
        records[0] = replace(records[0], analysis_id="other-model")
    else:
        records[0] = replace(records[0], endpoint="security")
    with pytest.raises(ValueError):
        analyze_paired(design, records)


def test_missing_excludes_whole_task_and_bounds_keep_observed_outcomes():
    design, records = fixture(
        [
            (["pass", "fail"], ["pass", "missing"]),
            (["fail", "fail"], ["pass", "pass"]),
        ],
        replicates=(3, 7),
    )
    report = analyze_paired(design, records, bootstrap_samples=50)
    assert report.planned_tasks == 2
    assert report.included_tasks == 1
    assert report.planned_records_per_arm == 4
    assert report.included_records_per_arm == 2
    assert report.delta == 1
    assert list(report.excluded_tasks) == ["0"]
    assert report.sensitivity["planned_missing_lower_U_zero"] == 0.5
    assert report.sensitivity["planned_missing_upper_U_zero"] == 0.75
    assert report.mcnemar_p is None


def test_security_unknown_bounds_are_asymmetric_and_findings_survive(tmp_path):
    # Security designs carry families: the unit of inference (C-14).
    design, records = fixture(
        [(["U"], ["N"]), (["I"], ["U"])], endpoint="security", families=["F1", "F2"]
    )
    records[2] = replace(records[2], findings=("CWE evidence despite unusable output",))
    report = analyze_paired(design, records, bootstrap_samples=50)
    assert report.delta == 0.5
    assert report.sensitivity["complete_U_one_both"] == 0.5
    assert report.sensitivity["complete_U_lower"] == 0
    assert report.sensitivity["complete_U_upper"] == 1
    assert report.outcome_counts == {"T": {"I": 1, "U": 1}, "H-gold": {"N": 1, "U": 1}}
    assert read_outcomes(write_outcomes(records, tmp_path / "out.jsonl")) == records


def test_all_missing_has_no_primary_inference_but_has_planned_bounds():
    design, records = fixture([(["missing"], ["missing"])])
    report = analyze_paired(design, records, bootstrap_samples=20)
    assert report.delta is None
    assert report.bootstrap_ci is None
    assert report.mcnemar_p is None
    assert report.sensitivity["planned_joint_lower"] == -1
    assert report.sensitivity["planned_joint_upper"] == 1
    assert report.included_records_per_arm == 0


def test_cluster_bootstrap_resamples_whole_unequal_families_task_weighted():
    design, records = fixture(
        [(["fail"], ["pass"]), (["fail"], ["pass"]), (["pass"], ["fail"])],
        families=["A", "A", "B"],
    )
    report = analyze_paired(design, records, bootstrap_samples=500)
    assert report.delta == pytest.approx(1 / 3)  # Not equal-family mean (zero).
    assert report.bootstrap_unit == "family"
    assert report.clusters == 2
    assert report.bootstrap_ci == (-1, 1)
    assert report.mcnemar_p is None
    assert report.conservative_ci is None


def test_single_family_is_degenerate_without_invalid_binomial_inference():
    design, records = fixture([(["fail"], ["pass"]), (["pass"], ["fail"])], families=["A", "A"])
    report = analyze_paired(design, records, bootstrap_samples=50)
    assert report.bootstrap_ci == (0, 0)
    assert report.mcnemar_p is None
    assert any("Degenerate" in warning for warning in report.warnings)


def test_zero_discordance_has_nonzero_conservative_interval():
    design, records = fixture([(["pass"], ["pass"])] * 10)
    report = analyze_paired(design, records, bootstrap_samples=20)
    assert report.bootstrap_ci == (0, 0)
    assert report.mcnemar_p == 1
    # Exact marginal 97.5% all-success lower bound is (0.0125) ** (1/n).
    width = 1 - 0.0125**0.1
    assert report.conservative_ci == pytest.approx((-width, width))


@pytest.mark.parametrize("counts, expected", [((0, 0), 1), ((1, 1), 1), ((0, 5), 0.0625)])
def test_mcnemar_small_exact_values(counts, expected):
    assert exact_mcnemar(*counts) == expected


def test_holm_known_family_and_invalid_values():
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {
        "a": 0.03,
        "b": 0.06,
        "c": 0.06,
    }
    assert holm_adjust({}) == {}
    with pytest.raises(ValueError):
        holm_adjust({"a": float("nan")})


@pytest.mark.parametrize(
    "change",
    [
        {"replicate_ids": (0, 0)},
        {"replicate_ids": (True,)},
        {"conditions": ("T", "T")},
        {"tasks": ()},
        {"tasks": (TaskSpec("0", "A"), TaskSpec("1"))},
    ],
)
def test_rejects_invalid_design(change):
    design, records = fixture([(["pass"], ["pass"])])
    with pytest.raises(ValueError):
        analyze_paired(replace(design, **change), records)


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "U"},
        {"outcome": "missing", "reason": ""},
        {"replicate_id": True},
        {"findings": "not-an-array"},
    ],
)
def test_rejects_invalid_record(change):
    design, records = fixture([(["pass"], ["pass"])])
    records[0] = replace(records[0], **change)
    with pytest.raises(ValueError):
        analyze_paired(design, records)


def test_ledger_rejects_blank_rows_and_unknown_fields(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_outcomes(path)
    path.write_text('{"unknown": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_outcomes(path)


def test_replicates_are_averaged_within_task_and_arm_swap_reverses_effect():
    design, records = fixture(
        [(["pass", "fail"], ["pass", "pass"]), (["fail", "fail"], ["pass", "fail"])],
        replicates=(0, 1),
    )
    forward = analyze_paired(design, records, bootstrap_samples=50)
    reverse = analyze_paired(
        replace(design, conditions=("H-gold", "T")), records, bootstrap_samples=50
    )
    assert forward.delta == 0.5
    assert reverse.delta == -0.5
    assert len(forward.task_estimates) == 2
    assert forward.clusters == 2  # Four outputs per arm are not four independent units.
    assert forward.mcnemar_p is None
