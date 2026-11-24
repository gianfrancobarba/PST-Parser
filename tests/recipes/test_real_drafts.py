"""Every real draft renders, and lints clean, with the local benchmark sources.

The drafts of ``data/annotated/downstream`` are versioned; the sources are local
copies made by ``import-bigcodebench`` and are not. Set ``PST_BCB_SOURCES`` to
the sources file (``sources.jsonl``) to run this recipe.
"""

import os
from pathlib import Path

import pytest

from pstparser.empirical.annotations import STATUSES, canonicalize, lint, load_annotations
from pstparser.empirical.benchmarks import read_sources
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants

pytestmark = pytest.mark.slow

DRAFTS = Path(__file__).resolve().parents[2] / "data" / "annotated" / "downstream"


#: bcb-final keeps the rule drafts; the pilot and the prefix gold (bcb-gold) are
#: reviewed by the student, so their rows may be draft, reviewed or frozen while
#: the review is under way.
EXPECTED_STATES = {
    "bcb-final": {"draft"},
    "bcb-pilot": set(STATUSES),
    "bcb-gold": set(STATUSES),
}


@pytest.mark.parametrize("sample", sorted(EXPECTED_STATES))
def test_every_real_draft_renders_and_lints_clean(sample: str) -> None:
    location = os.environ.get("PST_BCB_SOURCES")
    if not location:
        pytest.skip("set PST_BCB_SOURCES; see the module documentation")
    sources = {record.task_id: record for record in read_sources(Path(location))}
    rows = load_annotations(DRAFTS / sample / "annotations.jsonl")
    assert rows and {row.status for row in rows} <= EXPECTED_STATES[sample]
    for row in rows:
        assert lint(sources[row.task_id], row) == [], row.task_id
    prompts = []
    for status in sorted({row.status for row in rows}):
        group = [row for row in rows if row.status == status]
        prompts += canonicalize(
            [sources[row.task_id] for row in group], group, require_status=status
        )
    for prompt in prompts:
        t, h = render_prompt_variants(prompt)
        envelope = render_envelope_variant(prompt)
        assert t.text == prompt.text
        assert t.source_hash == h.source_hash == envelope.source_hash
