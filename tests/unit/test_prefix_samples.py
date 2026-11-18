"""The correctness sample is a prefix of the frozen permutation (plan of 18/11/2026, P-02, P-03).

S_gold is the 600-task prefix; the 400-task prefix is the prespecified fallback,
chosen only for time. Both are read from the files in ``protocol/``.
"""

import hashlib
import json
from pathlib import Path

import pytest

from pstparser.empirical.cli import S_GOLD_FALLBACK_FILE, S_GOLD_FILE

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "protocol" / "samples"


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("name", "n"), [(S_GOLD_FILE, 600), (S_GOLD_FALLBACK_FILE, 400)])
def test_the_sample_file_is_the_prefix_of_the_frozen_permutation(name: str, n: int) -> None:
    permutation_path = SAMPLES / "bcb-permutation.json"
    permutation = _document(permutation_path)
    document = _document(SAMPLES / name)
    expected = hashlib.sha256(permutation_path.read_bytes()).hexdigest()
    assert document["permutation_sha256"] == expected
    assert document["n"] == n
    assert document["seed"] == permutation["seed"] == 7319
    tasks = [row["task_id"] for row in document["tasks"]]
    assert tasks == [row["task_id"] for row in permutation["order"][:n]]


def test_the_fallback_is_nested_in_s_gold_and_both_leave_out_the_pilot() -> None:
    s_gold = [row["task_id"] for row in _document(SAMPLES / S_GOLD_FILE)["tasks"]]
    fallback = [row["task_id"] for row in _document(SAMPLES / S_GOLD_FALLBACK_FILE)["tasks"]]
    pilot = {row["task_id"] for row in _document(SAMPLES / "pilot-bcb.json")["tasks"]}
    assert fallback == s_gold[:400]
    assert not pilot & set(s_gold)


def test_the_hard_census_is_no_longer_in_the_protocol() -> None:
    assert not (SAMPLES / "bcb-hard.json").exists()
    assert not (SAMPLES / "empirical-sample-hard_manifest.json").exists()


def test_the_protocol_does_not_fix_the_family_so_the_parser_gate_decides_h3() -> None:
    # P-03: Holm on {H1, H3}, with H3 in the family only after a passed gate. The
    # confirmatory-family-v1 document is kept in the code for the prespecified
    # time cut (P-04, CP2) and enters the protocol only if that cut is applied.
    assert not (ROOT / "protocol" / "analysis" / "confirmatory-family.json").exists()
