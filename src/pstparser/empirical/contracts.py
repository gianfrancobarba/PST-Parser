"""Immutable input and output contracts for downstream prompt rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

#: Registered prompt conditions. T is the original text,
#: H-gold and H-auto the PST from reviewed and from parsed spans, E the text in
#: a neutral JSON envelope, F the flat PST, H-explained H-gold with a schema
#: sentence in the system message.
CONDITIONS: Final = ("T", "H-gold", "E", "H-auto", "F", "H-explained")


@dataclass(frozen=True)
class SpanAnnotation:
    """A reviewed exact span, with half-open Python Unicode character offsets.

    Path names use the nine taxonomy leaves without the ``prompt.`` prefix.
    Phrase is stored explicitly to detect edits to the annotated source.
    """

    path: str
    start: int
    end: int
    phrase: str


@dataclass(frozen=True)
class CanonicalPrompt:
    """Original benchmark text and reviewed spans, never tests or solutions."""

    task_id: str
    text: str
    annotations: tuple[SpanAnnotation, ...]
    annotation_version: str


@dataclass(frozen=True)
class PromptVariant:
    """One rendered user message and its reproducibility metadata.

    Hashes cover UTF-8 text with no normalisation. System messages and chat
    templates belong to the generation manifest and are not included here.
    """

    task_id: str
    condition: str
    text: str
    prompt_hash: str
    source_hash: str
    annotation_hash: str
    annotation_version: str
    renderer_version: str

    def __post_init__(self) -> None:
        """Refuse a condition outside the registry."""
        if self.condition not in CONDITIONS:
            raise ValueError(f"unregistered condition: {self.condition!r}")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record without mutable internal state."""
        return asdict(self)
