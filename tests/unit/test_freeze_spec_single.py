"""The freeze specification describes the single freeze F4 (plan of 18/11/2026, P-09, C-07).

One internal freeze before the final generations, one OSF deposit under embargo
at the freeze; the supervisor acknowledgement is the export of messages M1/M2.
"""

from pathlib import Path

import yaml

from pstparser.empirical.freeze import read_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "protocol" / "freeze-spec.yaml"


def test_the_spec_reads_and_every_item_exists() -> None:
    items, _ = read_spec(SPEC)
    assert items and all((ROOT / item).is_file() for item in items)


def test_the_spec_lists_the_configurations_and_the_parser_identity() -> None:
    items, _ = read_spec(SPEC)
    configs = sorted(
        str(p.relative_to(ROOT).as_posix()) for p in (ROOT / "protocol/configs").rglob("*.json")
    )
    assert configs and set(configs) <= set(items)
    assert {"protocol/hauto/parser.json", "protocol/hauto/gate-thresholds.json"} <= set(items)


def test_the_metadata_describe_one_freeze_and_one_osf_deposit() -> None:
    _, meta = read_spec(SPEC)
    assert meta["preregistration"] == {
        "registry": "OSF",
        "deposit": "single, under embargo, at the freeze F4",
        "osf_url": None,
        "embargo_until": None,
    }
    assert meta["supervisor_ack"]["record"] is None
    assert meta["supervisor_ack"]["messages"] == ["M1", "M2"]
    assert meta["conditions"]["bigcodebench"] == ["T", "H-gold", "E"]
    assert meta["conditions"]["bigcodebench_s_parse"] == ["H-auto"]
    assert meta["conditions"]["bigcodebench_if_cp3_f"] == ["F"]
    assert meta["renderer_version"] == {
        "H-gold": "pst-gold-json-v1",
        "E": "envelope-json-v1",
        "F": "pst-flat-json-v1",
        "H-auto": "pst-gold-json-v1",
    }


def test_no_separate_h_auto_freeze_or_two_phase_osf_is_left() -> None:
    text = SPEC.read_text(encoding="utf-8")
    for stale in ("F8", "H-auto freeze", "phase1", "phase2", "e-mail"):
        assert stale not in text, stale
    assert yaml.safe_load(text)["format"] == "freeze-spec-v1"
