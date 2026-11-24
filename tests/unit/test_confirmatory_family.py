"""The confirmatory family comes from the parser gate, and its reports are checked.

Before the correction, ``--confirmatory`` alone decided whether H3 stayed in
the family (and so whether H1 was read at 0.04 or 0.05), a security report was
accepted as H1 and any report as H3, and a failed gate did not stop H3.
"""

from typing import Any

import pytest

from pstparser.empirical.tables import paired_tables


def report(
    analysis_id: str,
    p: float,
    *,
    endpoint: str = "functional",
    conditions: tuple[str, str] = ("T", "H-gold"),
    planned: int = 800,
    signflip: bool = False,
) -> dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "endpoint": endpoint,
        "conditions": list(conditions),
        "planned_tasks": planned,
        "included_tasks": planned,
        "mcnemar_table": None,
        "delta": 0.01,
        "midp_p": None if signflip else p,
        "mcnemar_p": None,
        "tango_ci": None,
        "tango_ci_90": [-0.01, 0.03],
        "newcombe_ci": None,
        "conservative_ci": None,
        "signflip_p": p if signflip else None,
        "cr1_ci": None,
        "sensitivity": {},
    }


def gate(h3: str, s_parse: int = 640) -> dict[str, Any]:
    return {
        "format": "parser-gate-v1",
        "status": "pass" if h3 == "confirmatory" else "fail",
        "h3": h3,
        "s_parse_size": s_parse,
    }


H1 = report("bcb-h1", 0.03)
H3 = report("bcb-h3-s-parse", 0.2, conditions=("T", "H-auto"), planned=640)
REPORTS = {"a": H1, "b": H3}
FAMILY = {"H1": "bcb-h1", "H3": "bcb-h3-s-parse"}


def test_a_valid_family_is_decided_with_the_weighted_holm() -> None:
    main, _ = paired_tables(REPORTS, FAMILY, gate("confirmatory"))
    assert {row[0]: row[-1] for row in main.rows} == {
        "bcb-h1": "H1 rejected",
        "bcb-h3-s-parse": "H3 not_rejected",
    }


def test_the_family_needs_the_gate_report() -> None:
    with pytest.raises(ValueError, match="gate"):
        paired_tables(REPORTS, FAMILY)


def test_h3_cannot_enter_after_a_failed_gate() -> None:
    with pytest.raises(ValueError, match="gate"):
        paired_tables(REPORTS, FAMILY, gate("descriptive"))


def test_h3_cannot_leave_after_a_passed_gate() -> None:
    with pytest.raises(ValueError, match="gate"):
        paired_tables(REPORTS, {"H1": "bcb-h1"}, gate("confirmatory"))


def test_a_failed_gate_leaves_h1_alone_at_five_percent() -> None:
    main, _ = paired_tables({"a": report("bcb-h1", 0.045)}, {"H1": "bcb-h1"}, gate("descriptive"))
    assert main.rows[0][-1] == "H1 rejected"


@pytest.mark.parametrize(
    "h1",
    [
        report("bcb-h1", 0.03, endpoint="security", signflip=True),
        report("bcb-h1", 0.03, conditions=("T", "E")),
    ],
)
def test_h1_must_be_the_functional_t_versus_h_gold_report(h1: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="H1"):
        paired_tables({"a": h1, "b": H3}, FAMILY, gate("confirmatory"))


@pytest.mark.parametrize(
    "h3, family",
    [
        (report("bcb-h3-s-parse", 0.2, planned=640), FAMILY),  # T vs H-gold
        (
            report("bcb-h3-end-to-end", 0.2, conditions=("T", "H-auto"), planned=800),
            {"H1": "bcb-h1", "H3": "bcb-h3-end-to-end"},
        ),  # not the S_parse estimate
        (report("bcb-h3-s-parse", 0.2, conditions=("T", "H-auto"), planned=600), FAMILY),
    ],
)
def test_h3_must_be_the_functional_t_versus_h_auto_report_on_s_parse(
    h3: dict[str, Any], family: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="H3"):
        paired_tables({"a": H1, "b": h3}, family, gate("confirmatory"))


# --- The four-way reading of a confirmatory row follows the Holm decision.


def tabled(analysis_id: str, p: float, conditions=("T", "H-gold"), planned: int = 800) -> dict:
    row = report(analysis_id, p, conditions=conditions, planned=planned)
    row["mcnemar_table"] = {"both_pass": 300, "treatment_only": 60, "baseline_only": 40}
    row["tango_ci_90"] = [0.001, 0.06]  # not equivalent at 5 pp
    return row


def columns(main) -> dict[str, dict]:
    return {row[0]: dict(zip(main.columns, row, strict=True)) for row in main.rows}


def test_a_confirmatory_row_reads_its_difference_at_the_holm_level() -> None:
    from pstparser.empirical.inference import tango_interval

    reports = {
        "a": tabled("bcb-h1", 0.045),
        "b": tabled("bcb-h3-s-parse", 0.5, ("T", "H-auto"), 640),
    }
    main, _ = paired_tables(reports, FAMILY, gate("confirmatory"))
    h1 = columns(main)["bcb-h1"]
    assert h1["decision"] == "H1 not_rejected"
    assert h1["holm_alpha"] == pytest.approx(0.04)
    assert h1["reading_5pp"] == "E4"  # not "different from zero" at 0.04
    assert (h1["holm_ci_low"], h1["holm_ci_high"]) == pytest.approx(
        tango_interval(40, 60, 800, 0.96)
    )
    h3 = columns(main)["bcb-h3-s-parse"]
    assert h3["holm_alpha"] == pytest.approx(0.01)


def test_recycled_alpha_after_the_other_rejection() -> None:
    reports = {
        "a": tabled("bcb-h1", 0.045),
        "b": tabled("bcb-h3-s-parse", 0.005, ("T", "H-auto"), 640),
    }
    main, _ = paired_tables(reports, FAMILY, gate("confirmatory"))
    h1 = columns(main)["bcb-h1"]
    assert h1["decision"] == "H1 rejected"
    assert h1["holm_alpha"] == pytest.approx(0.05)
    assert h1["reading_5pp"] == "E1"


def test_exploratory_rows_keep_five_percent_and_no_holm_columns() -> None:
    main, _ = paired_tables({"a": tabled("bcb-e", 0.045, ("T", "E"))})
    row = columns(main)["bcb-e"]
    assert row["holm_alpha"] is None and row["holm_ci_low"] is None
    assert row["reading_5pp"] == "E1"


def test_holm_levels_agree_with_the_decisions_on_a_grid() -> None:
    from pstparser.empirical.inference import (
        CONFIRMATORY_WEIGHTS,
        weighted_holm,
        weighted_holm_levels,
    )

    grid = [i / 1000 for i in range(0, 81)] + [0.5, 1.0]
    for p1 in grid:
        for p3 in [*grid, None]:
            p = {"H1": p1, "H3": p3}
            decisions = weighted_holm(p, CONFIRMATORY_WEIGHTS)
            levels = weighted_holm_levels(p, CONFIRMATORY_WEIGHTS)
            for name, value in p.items():
                if value is None:
                    assert levels[name] is None
                else:
                    assert (value <= levels[name]) == (decisions[name] == "rejected")


# Prespecified time cut (`protocol/analysis/plan.md`): without a complete parse
# by the deadline, a frozen document fixes the family to H1 alone, read at 0.05;
# H3 is then exploratory whatever the parser gate says.
PROTOCOL = {
    "format": "confirmatory-family-v1",
    "family": ["H1"],
    "alpha": 0.05,
    "h3": "exploratory",
}


def test_the_protocol_family_reads_h1_alone_at_five_percent_without_a_gate() -> None:
    at_0045 = report("bcb-h1", 0.045)
    main, _ = paired_tables({"a": at_0045, "b": H3}, {"H1": "bcb-h1"}, PROTOCOL)
    decisions = {row[0]: row[-1] for row in main.rows}
    assert decisions["bcb-h1"] == "H1 rejected"
    assert decisions["bcb-h3-s-parse"] == "exploratory"


def test_the_protocol_family_refuses_h3_even_after_a_passed_gate() -> None:
    with pytest.raises(ValueError, match="H3 exploratory"):
        paired_tables(REPORTS, FAMILY, PROTOCOL)


@pytest.mark.parametrize(
    "changed",
    [
        {**PROTOCOL, "alpha": 0.04},
        {**PROTOCOL, "family": ["H1", "H3"]},
        {**PROTOCOL, "h3": "confirmatory"},
    ],
)
def test_a_protocol_family_other_than_h1_alone_is_refused(changed: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="only protocol family"):
        paired_tables(REPORTS, {"H1": "bcb-h1"}, changed)
