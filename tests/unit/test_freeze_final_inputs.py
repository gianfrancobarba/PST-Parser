"""The inputs of the final collection enter the freeze F4 (A4/A5, 19/11/2026).

The journal variants, their renderings with records, the covariates and the
analysis designs are listed; every H-gold variant of a journal comes from a listed
rendering of the frozen gold, as ``generate`` requires for a final run.
"""

import hashlib
import json
from pathlib import Path

from pstparser.empirical.freeze import _require_frozen_gold, read_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "protocol" / "freeze-spec.yaml"
JOURNALS = ("bcb-hauto", "bcb-rest", "lse")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listed() -> set[str]:
    items, _ = read_spec(SPEC)
    return set(items)


def test_the_freeze_lists_the_journal_variants_renderings_and_covariates() -> None:
    files = {
        p.relative_to(ROOT).as_posix()
        for folder in ("protocol/variants", "protocol/covariates")
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
    }
    assert {f"protocol/variants/{name}.variants.jsonl" for name in JOURNALS} <= files
    assert files <= _listed()


def test_the_freeze_lists_every_design() -> None:
    designs = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "protocol/analysis/designs").rglob("*")
        if p.is_file()
    }
    assert len([d for d in designs if d.endswith("design.json")]) == 14
    assert designs <= _listed()


def test_every_h_gold_variant_comes_from_a_listed_rendering_of_frozen_gold() -> None:
    document = {"items": {name: _sha(ROOT / name) for name in _listed()}}
    for name in JOURNALS:
        _require_frozen_gold(document, ROOT, ROOT / f"protocol/variants/{name}.variants.jsonl")


def test_the_s_parse_journal_joins_its_two_renderings() -> None:
    parts = ROOT / "protocol/variants/parts"
    joined = (parts / "bcb-sparse/variants.jsonl").read_bytes() + (
        parts / "bcb-hauto/variants.jsonl"
    ).read_bytes()
    assert joined == (ROOT / "protocol/variants/bcb-hauto.variants.jsonl").read_bytes()
    record = json.loads((ROOT / "protocol/variants/copy-record.json").read_text("utf-8"))
    assert record["flat"] is True


def test_the_pilot_block_names_the_pilot_summary_and_change_log() -> None:
    _, meta = read_spec(SPEC)
    pilot = meta["pilot"]
    assert pilot["run_ids"] == ["ds-pilot-01"]
    assert pilot["report_sha256"] == _sha(ROOT / "protocol/pilot/summary.json")
    assert pilot["changelog_sha256"] == _sha(ROOT / "protocol/pilot-changelog.md")
    files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "protocol/pilot").iterdir()}
    assert files | {"protocol/pilot-changelog.md"} <= _listed()


def test_the_plan_has_no_value_left_to_fill() -> None:
    text = (ROOT / "protocol/analysis/plan.md").read_text("utf-8")
    assert "TO FILL" not in text
    summary = json.loads((ROOT / "protocol/pilot/summary.json").read_text("utf-8"))
    assert summary["cp3"] == "go"
    assert f"`{_sha(ROOT / 'protocol/pilot/summary.json')}`" in text
