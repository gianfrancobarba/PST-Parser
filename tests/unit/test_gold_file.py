"""The gold file of S_gold covers the 600-task prefix exactly, in its order (18/11/2026).

The 800 drafts of ``bcb-final`` stay as they are, as the draft side of
``gold-diff``; the gold is reviewed and frozen in ``bcb-gold``, whose rows are
the first 600 lines of the drafts, so ``canonicalize --sample`` with the prefix
covers it exactly. The nested 400-task fallback is its first 400 rows.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "annotated" / "downstream" / "bcb-gold" / "annotations.jsonl"
DRAFTS = ROOT / "data" / "annotated" / "downstream" / "bcb-final" / "annotations.jsonl"
PREFIX = ROOT / "protocol" / "samples" / "prefix-600" / "bcb-prefix-600.json"
PREFIX_400 = ROOT / "protocol" / "samples" / "prefix-400" / "bcb-prefix-400.json"


def task_ids(path: Path) -> list[str]:
    text = path.read_bytes().decode("utf-8")
    return [json.loads(line)["task_id"] for line in text.split("\n") if line.strip()]


def sample_ids(path: Path) -> list[str]:
    return [row["task_id"] for row in json.loads(path.read_text(encoding="utf-8"))["tasks"]]


def test_the_gold_file_is_the_prefix_in_its_order() -> None:
    assert task_ids(GOLD) == sample_ids(PREFIX)


def test_the_400_fallback_is_the_head_of_the_gold_file() -> None:
    assert task_ids(GOLD)[:400] == sample_ids(PREFIX_400)


def test_the_gold_rows_come_from_the_drafts_of_the_same_tasks() -> None:
    assert task_ids(DRAFTS)[:600] == task_ids(GOLD)


def test_the_gold_file_has_lf_endings_and_no_bom() -> None:
    raw = GOLD.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
