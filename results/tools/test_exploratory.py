# ruff: noqa: D103
"""Tests of the exploratory tools, on synthetic designs and ledgers."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pstparser.empirical.analysis import (
    AnalysisDesign,
    OutcomeRecord,
    TaskSpec,
    analyze_paired,
    write_outcomes,
)


def _tool() -> ModuleType:
    path = Path(__file__).with_name("exploratory.py")
    spec = importlib.util.spec_from_file_location("exploratory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _covariates(n: int) -> dict[str, dict[str, Any]]:
    rows = {}
    for i in range(n):
        leaves = 1 + i % 4
        rows[f"t{i}"] = {
            "task_id": f"t{i}",
            "n_leaves": leaves,
            "n_judgement_units": i % 5,
            "has_code": i % 2 == 0,
            "log_ratio_HT": i / n,
            "reorder": i % 3,
            "single_leaf": int(leaves == 1),
            "reordered": int(i % 3 > 0),
        }
    return rows


def _functional(n: int, seed: int = 1) -> tuple[AnalysisDesign, list[OutcomeRecord]]:
    """T and H-gold on n tasks; H wins more often when log_ratio_HT is high."""
    rng = random.Random(seed)
    design = AnalysisDesign("syn", "functional", tuple(TaskSpec(f"t{i}") for i in range(n)))
    records = []
    for i in range(n):
        t_pass = rng.random() < 0.5
        h_pass = rng.random() < (0.2 + 0.6 * i / n)
        for condition, passed in (("T", t_pass), ("H-gold", h_pass)):
            reason = "pass" if passed else rng.choice(["assertion_failure", "test_exception"])
            records.append(
                OutcomeRecord(
                    "syn", "functional", f"t{i}", condition, 0, "pass" if passed else "fail", reason
                )
            )
    return design, records


def test_restrict_cuts_the_design_and_relabels_the_ledger() -> None:
    design, records = _functional(30)
    relabelled = [r.__class__(**{**r.__dict__, "analysis_id": "full"}) for r in records]
    cut, kept = tool.restrict(design, relabelled, ["t1", "t2"])
    assert [t.task_id for t in cut.tasks] == ["t1", "t2"]
    assert {r.analysis_id for r in kept} == {"syn"} and len(kept) == 4
    with pytest.raises(ValueError, match="outside the design"):
        tool.restrict(design, records, ["zz"])


def test_restrict_refuses_a_mixed_ledger() -> None:
    design, records = _functional(4)
    mixed = [*records[:2], records[2].__class__(**{**records[2].__dict__, "analysis_id": "x"})]
    with pytest.raises(ValueError, match="mixes"):
        tool.restrict(design, mixed)


def test_subgroups_use_the_frozen_analysis_and_report_counts_under_twenty() -> None:
    design, records = _functional(120)
    covariates = _covariates(120)
    report = tool.subgroups(design, records, covariates, hard={f"t{i}" for i in range(10)})
    hard = {row["level"]: row for row in report["hard"]["levels"]}
    assert hard["true"]["reported"] == "counts only" and hard["true"]["delta"] is None
    leaves = {row["level"]: row for row in report["leaves"]["levels"]}
    members = [f"t{i}" for i in range(120) if covariates[f"t{i}"]["n_leaves"] == 1]
    cut, kept = tool.restrict(design, records, members)
    assert leaves["1"]["delta"] == analyze_paired(cut, kept).delta
    assert leaves["1"]["ci95"] is not None
    assert report["judgement_units"]["left_out_tasks"] == 24  # 0 units: no level
    assert {row["level"] for row in report["log_ratio_tercile"]["levels"]} == {"T1", "T2", "T3"}


def test_discordant_pairs_skip_missing_arms() -> None:
    design, records = _functional(10)
    first = records[0]
    records[0] = first.__class__(**{**first.__dict__, "outcome": "missing", "reason": "infra"})
    pairs = tool.discordant_pairs(design, records)
    assert all(pair["task_id"] != "t0" for pair in pairs)
    for pair in pairs:
        assert pair["winner"] in ("T", "H-gold")


def test_logistic_needs_forty_pairs_and_finds_the_planted_effect() -> None:
    design, records = _functional(40)
    pairs = tool.discordant_pairs(design, records)
    assert len(pairs) < 40
    assert tool.logistic(design, pairs, _covariates(40))["read"] is False
    design, records = _functional(600, seed=3)
    pairs = tool.discordant_pairs(design, records)
    result = tool.logistic(design, pairs, _covariates(600))
    ratio = result["one_covariate"][0]
    assert ratio["read"] and ratio["terms"]["log_ratio_HT"]["log_odds"] > 0
    assert max(result["vif"].values()) <= 5 and "terms" in result["joint"]


def test_the_joint_model_is_not_read_with_a_high_vif() -> None:
    design, records = _functional(600, seed=3)
    pairs = tool.discordant_pairs(design, records)
    covariates = _covariates(600)
    for row in covariates.values():
        row["copy"] = float(row["log_ratio_HT"]) * 2 + 0.001 * float(row["reorder"])
    result = tool.logistic(design, pairs, covariates, ("log_ratio_HT", "copy"))
    assert result["joint"] == {"read": False, "reason": "a VIF above 5.0"}


def test_qualitative_takes_twenty_per_direction_blind_and_seeded() -> None:
    design, records = _functional(600, seed=3)
    pairs = tool.discordant_pairs(design, records)
    assert len(pairs) > 40
    sample, key = tool.qualitative(design, pairs)
    again, _ = tool.qualitative(design, pairs)
    assert sample == again and len(sample) == 40
    winners = {pair["task_id"]: pair["winner"] for pair in pairs}
    assert Counter(winners[item["task_id"]] for item in key) == {"T": 20, "H-gold": 20}
    assert all({entry["A"], entry["B"]} == {"T", "H-gold"} for entry in key)
    assert all(set(item) == {"item", "task_id"} for item in sample)
    few = pairs[:7]
    assert len(tool.qualitative(design, few)[0]) == 7


def test_taxonomy_counts_categories_and_transitions() -> None:
    design, records = _functional(50)
    pairs = tool.discordant_pairs(design, records)
    result = tool.taxonomy(design, records, pairs)
    assert sum(result["per_condition"]["T"].values()) == 50
    assert sum(result["transitions"].values()) == len(pairs)


def test_run_writes_every_report_and_security_gets_no_model(tmp_path: Path) -> None:
    design, records = _functional(60)
    (tmp_path / "d.json").write_text(
        json.dumps(
            {
                "analysis_id": "syn",
                "endpoint": "functional",
                "tasks": [{"task_id": t.task_id} for t in design.tasks],
            }
        ),
        encoding="utf-8",
    )
    write_outcomes(records, tmp_path / "l.jsonl")
    covariates = tmp_path / "c.jsonl"
    covariates.write_text(
        "\n".join(json.dumps(row) for row in _covariates(60).values()) + "\n", encoding="utf-8"
    )
    names = tool.run(tmp_path / "d.json", tmp_path / "l.jsonl", covariates, tmp_path / "out")
    assert set(names) == {
        "subgroups.json",
        "taxonomy.json",
        "logistic.json",
        "qualitative-sample.json",
        "qualitative-key.json",
    }
    rng = random.Random(5)
    families = [f"{20 + i % 9}-X-{i % 2}" for i in range(36)]
    security = AnalysisDesign(
        "sec",
        "security",
        tuple(TaskSpec(f"t{i}", families[i]) for i in range(36)),
        (1, 2, 3),
        ("T", "H-gold"),
    )
    rows = [
        OutcomeRecord("sec", "security", f"t{i}", c, r, rng.choice(["N", "V", "I"]))
        for i in range(36)
        for c in ("T", "H-gold")
        for r in (1, 2, 3)
    ]
    report = tool.subgroups(security, rows, _covariates(36))
    assert "cwe" in report
    with pytest.raises(ValueError, match="one functional replicate"):
        tool.discordant_pairs(security, rows)


def test_a_constant_covariate_is_named_and_left_out_of_the_joint_model() -> None:
    design, records = _functional(600, seed=3)
    pairs = tool.discordant_pairs(design, records)
    covariates = _covariates(600)
    for row in covariates.values():
        row["single_leaf"] = 0
    result = tool.logistic(design, pairs, covariates)
    assert result["constant_covariates"] == ["single_leaf"]
    assert set(result["vif"]) == {"log_ratio_HT", "reordered"}
    assert set(result["joint"]["terms"]) == {"log_ratio_HT", "reordered"}
    json.dumps(result, allow_nan=False)
