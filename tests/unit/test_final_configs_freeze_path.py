"""Final configurations reach the repository freeze (finding of the dress rehearsal, 19/11/2026).

``generate`` reads ``review_record`` relative to the configuration and verifies the
freeze against the repository two levels above it. A final configuration kept in
``protocol/configs/final/`` must therefore point at ``protocol/protocol-freeze.json``,
and the notebook must run the repository copy of the configuration, not a copy
inside ``inputs.zip``, whose freeze would resolve outside the repository.
"""

import json
from pathlib import Path

from pstparser.empirical.freeze import default_root

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "protocol" / "configs" / "final"
NOTEBOOK = ROOT / "notebooks" / "downstream.ipynb"


def test_every_final_configuration_points_at_the_repository_freeze() -> None:
    configs = sorted(FINAL.glob("*.json"))
    assert configs
    for path in configs:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["purpose"] == "final", path.name
        record = (path.parent / document["review_record"]).resolve()
        assert record == (ROOT / "protocol" / "protocol-freeze.json").resolve(), path.name
        assert default_root(record) == ROOT.resolve(), path.name


def test_the_notebook_runs_the_repository_configuration_in_a_final_run() -> None:
    notebook = json.loads(NOTEBOOK.read_bytes().decode("utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    generate = next(c for c in cells if '"generate"' in c)
    assert 'PROJECT / "protocol" / "configs" / "final"' in generate
    assert "sha256" in generate and 'if PURPOSE == "final":' in generate
