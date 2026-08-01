"""The prediction file exchanged between generation and scoring.

This format is the contract between the two stages. It lives with the other data
formats rather than with the model code, so that scoring can read it without
pulling in a deep learning stack.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction alongside the inputs it was produced from.

    Attributes:
        index: Position of the record in the corpus.
        prompt: The raw prompt submitted to the model.
        target: The expected target tree, serialised.
        prediction: The raw model output.
    """

    index: int
    prompt: str
    target: str
    prediction: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the record."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "target": self.target,
            "prediction": self.prediction,
        }


def write_predictions(predictions: Iterable[PredictionRecord], destination: str | Path) -> Path:
    """Write predictions as one JSON object per line.

    Args:
        predictions: The predictions to write, in the order they should appear.
        destination: File to write. Parent directories are created.

    Returns:
        The path that was written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.as_dict(), ensure_ascii=False))
            handle.write("\n")
    return destination


def load_predictions(source: str | Path) -> list[PredictionRecord]:
    """Read predictions previously written by :func:`write_predictions`.

    Args:
        source: File holding one JSON object per line.

    Returns:
        The predictions, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"predictions file not found: {source}")

    predictions = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        predictions.append(
            PredictionRecord(
                index=int(payload["index"]),
                prompt=str(payload["prompt"]),
                target=str(payload["target"]),
                prediction=str(payload["prediction"]),
            )
        )
    return predictions
