"""Q-25 (confirmed 17/11/2026): the S-PREFILL prefill goes in front of the continuation.

The journal of a prefilled run records only what the model generated; before
extraction and the harness the prefill is put back, as the start of the
assistant turn, so the program judged is the whole answer.
"""

import json
from pathlib import Path

from pstparser.empirical.cli import _with_prefill


def journal(directory: Path, prefill: str | None) -> Path:
    directory.mkdir(parents=True)
    configuration = {} if prefill is None else {"assistant_prefill": prefill}
    (directory / "empirical-generate_manifest.json").write_text(
        json.dumps({"configuration": configuration}), encoding="utf-8"
    )
    return directory / "generations.jsonl"


def test_the_prefill_is_prepended_to_every_output(tmp_path: Path) -> None:
    path = journal(tmp_path / "p", "```python\n")
    rows, prefill = _with_prefill([{"raw_output": "x = 1\n```"}], path)
    assert prefill == "```python\n"
    assert rows == [{"raw_output": "```python\nx = 1\n```"}]


def test_without_a_prefill_the_outputs_are_unchanged(tmp_path: Path) -> None:
    path = journal(tmp_path / "n", None)
    rows, prefill = _with_prefill([{"raw_output": "x"}], path)
    assert (rows, prefill) == ([{"raw_output": "x"}], None)
