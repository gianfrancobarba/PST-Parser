"""The alignment file derived from a set of predictions.

Alignment is a deterministic function of a prompt and a tree, both of which the
prediction file already holds, so this file is derived rather than authoritative:
it can be thrown away and rebuilt in seconds. It exists so that the positional
artefact the framework describes can be read, diffed and inspected on its own.

Like the prediction format, it lives with the other data formats rather than with
the code that produces it, so that reading it pulls in nothing heavy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALIGNMENTS_FILE = "alignments.jsonl"


@dataclass(frozen=True)
class AlignmentRecord:
    """One prediction and its reference, both located in the source prompt.

    Attributes:
        index: Position of the record in the corpus.
        prompt: The raw prompt the phrases were located in.
        target: The reference tree with positional leaves, serialised, or
            ``None`` when the reference could not be read.
        prediction: The predicted tree with positional leaves, serialised, or
            ``None`` when the prediction could not be read.
        target_alignment: Counters and located leaves of the reference, when it
            could be read.
        prediction_alignment: The same for the prediction.
    """

    index: int
    prompt: str
    target: str | None
    prediction: str | None
    target_alignment: dict[str, Any] | None
    prediction_alignment: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the record."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "target": self.target,
            "prediction": self.prediction,
            "target_alignment": self.target_alignment,
            "prediction_alignment": self.prediction_alignment,
        }


def write_alignments(records: Iterable[AlignmentRecord], destination: str | Path) -> Path:
    """Write alignments as one JSON object per line.

    Args:
        records: The records to write, in the order they should appear.
        destination: File to write. Parent directories are created.

    Returns:
        The path that was written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False))
            handle.write("\n")
    return destination


def load_alignments(source: str | Path) -> list[AlignmentRecord]:
    """Read alignments previously written by :func:`write_alignments`.

    Args:
        source: File holding one JSON object per line.

    Returns:
        The records, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"alignments file not found: {source}")

    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            AlignmentRecord(
                index=int(payload["index"]),
                prompt=str(payload["prompt"]),
                target=_optional_text(payload["target"]),
                prediction=_optional_text(payload["prediction"]),
                target_alignment=_optional_mapping(payload["target_alignment"]),
                prediction_alignment=_optional_mapping(payload["prediction_alignment"]),
            )
        )
    return records


def _optional_text(value: Any) -> str | None:
    """Read a field that is absent whenever its tree could not be parsed."""
    return None if value is None else str(value)


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    """Read an alignment block that is absent whenever its tree was unreadable."""
    return None if value is None else dict(value)
