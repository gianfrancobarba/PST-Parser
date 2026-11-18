"""The parser gate thresholds are fixed in the protocol before any benchmark parse (P-04)."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = ROOT / "protocol" / "hauto" / "gate-thresholds.json"


def test_the_frozen_thresholds_are_the_planned_ones() -> None:
    document = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    assert document == {"max_strict_failure": 0.25, "min_macro_f1": 0.80}


def test_the_thresholds_are_a_freeze_item() -> None:
    spec = yaml.safe_load((ROOT / "protocol" / "freeze-spec.yaml").read_text(encoding="utf-8"))
    assert "protocol/hauto/gate-thresholds.json" in spec["items"]
