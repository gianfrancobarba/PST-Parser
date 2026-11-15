"""Import benchmark prompts as canonical sources, byte for byte.

The text of a source is the text of the benchmark with the declared transforms
only. Tests, reference solutions and weakness labels never enter a source
record: they belong to the oracle side of the study.

Dataset files are identified by content. A file whose SHA-256 is not the one
pinned for the requested revision is refused, so a silent upstream change
cannot alter the material of the experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

BIGCODEBENCH: Final = "bigcodebench"
LLMSECEVAL: Final = "llmseceval"

BIGCODEBENCH_TEXT_FIELD: Final = "instruct_prompt"
#: Oracle columns: read only to prove that they do not leak into the prompt.
BIGCODEBENCH_ORACLE_FIELDS: Final = ("test", "canonical_solution")
LLMSECEVAL_TEXT_FIELD: Final = "Manually-fixed NL Prompt"
LLMSECEVAL_PLACEHOLDER: Final = "<language>"

#: Weaknesses of the C scenarios that have no counterpart in Python (memory safety).
MEMORY_CWES: Final = frozenset({119, 125, 190, 416, 476, 787})
SELECTIONS: Final = ("S2", "python-only", "c-memory")

_PROMPT_ID: Final = re.compile(r"CWE-(\d+)_([A-Z]+)-(\d+)([a-z])")
_WEAKNESS_LABEL: Final = re.compile(r"CWE[-\s]?\d+", re.IGNORECASE)
_SCENARIO_MARKER: Final = "/gen_scenario/"


@dataclass(frozen=True)
class DatasetPin:
    """Expected content of one dataset file at one revision."""

    file_sha256: str
    records: int


#: (Hugging Face revision of ``bigcode/bigcodebench``, split) -> parquet file of the split.
BIGCODEBENCH_PINS: dict[tuple[str, str], DatasetPin] = {
    ("b74c0d0bf70d2c0bc459be537895cca163007f1a", "v0.1.4"): DatasetPin(
        "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267", 1140
    ),
}

#: (Hugging Face revision of ``bigcode/bigcodebench-hard``, split) -> parquet file.
BIGCODEBENCH_HARD_PINS: dict[tuple[str, str], DatasetPin] = {
    ("298d2cc7b96612e15e47313c3603ee124cee0c1f", "v0.1.4"): DatasetPin(
        "73a4270b43feb81abefad7bb1b592937c768ea45c3263bedc1cabd30a9a6ce79", 148
    ),
}

#: Commit of ``tuhh-softsec/LLMSecEval`` -> ``Dataset/LLMSecEval-Prompts_dataset.json``.
LLMSECEVAL_PINS: dict[str, DatasetPin] = {
    "39e97f046dbb2c4c604d31b86f2a4d3d27827388": DatasetPin(
        "a1b7307678e061faefb8d666eb6cec506d6a40751175716af07703b3f12433cc", 150
    ),
}


@dataclass(frozen=True)
class Transform:
    """One declared change applied to the benchmark text."""

    kind: str
    old: str
    new: str
    count: int


@dataclass(frozen=True)
class SourceRecord:
    """One benchmark prompt as it enters the study.

    ``source_sha256`` covers ``text`` (UTF-8, no normalisation), which is what
    condition T sends. ``original_sha256`` covers the benchmark field before the
    transforms; the two coincide when there is none.
    """

    task_id: str
    text: str
    source_sha256: str
    original_sha256: str
    benchmark: str
    dataset_revision: str
    dataset_file_sha256: str
    original_id: str
    family_id: str | None
    text_field: str
    transforms: tuple[Transform, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pinned_bytes(path: Path, pin: DatasetPin | None, label: str) -> bytes:
    """Read a dataset file once and refuse it unless it is the pinned one."""
    if pin is None:
        raise ValueError(f"no pinned dataset file for {label}")
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin.file_sha256:
        raise ValueError(
            f"{path}: SHA-256 {digest} differs from the pinned {pin.file_sha256} for {label}"
        )
    return data


def _check_count(found: int, pin: DatasetPin, path: Path) -> None:
    if found != pin.records:
        raise ValueError(f"{path}: {found} records, expected {pin.records}")


def load_bigcodebench(parquet: Path, dataset_revision: str, split: str) -> list[SourceRecord]:
    """Load BigCodeBench-Instruct prompts from the parquet file of one split.

    The text is ``instruct_prompt``, unchanged. The parquet is decoded from the
    very bytes that were hashed, never through a text reader, so no line ending
    is translated.

    Raises:
        ValueError: If the file is not the pinned one, an identifier repeats, a
            prompt is empty, or a prompt contains its own tests or solution.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pin = BIGCODEBENCH_PINS.get((dataset_revision, split))
    data = _pinned_bytes(parquet, pin, f"BigCodeBench {dataset_revision} split {split}")
    assert pin is not None
    columns = ["task_id", BIGCODEBENCH_TEXT_FIELD, *BIGCODEBENCH_ORACLE_FIELDS]
    rows: list[dict[str, Any]] = pq.read_table(pa.BufferReader(data), columns=columns).to_pylist()
    _check_count(len(rows), pin, parquet)
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for row in rows:
        task_id, text = row["task_id"], row[BIGCODEBENCH_TEXT_FIELD]
        if not isinstance(task_id, str) or not task_id or not isinstance(text, str):
            raise ValueError(f"{parquet}: malformed row {task_id!r}")
        if task_id in seen:
            raise ValueError(f"{parquet}: duplicate task {task_id}")
        seen.add(task_id)
        if not text.strip():
            raise ValueError(f"{task_id}: empty prompt")
        for oracle in BIGCODEBENCH_ORACLE_FIELDS:
            if not row[oracle] or row[oracle] in text:
                raise ValueError(
                    f"{task_id}: the prompt contains its {oracle} field, or it is empty"
                )
        digest = _sha256_text(text)
        records.append(
            SourceRecord(
                task_id=task_id,
                text=text,
                source_sha256=digest,
                original_sha256=digest,
                benchmark=BIGCODEBENCH,
                dataset_revision=dataset_revision,
                dataset_file_sha256=pin.file_sha256,
                original_id=task_id,
                family_id=None,
                text_field=BIGCODEBENCH_TEXT_FIELD,
                transforms=(),
            )
        )
    return records


def load_bigcodebench_hard_ids(parquet: Path, dataset_revision: str, split: str) -> frozenset[str]:
    """Return the task identifiers of the pinned BigCodeBench-Hard file.

    Only the identifiers are read: Hard is a label on tasks of the full set.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pin = BIGCODEBENCH_HARD_PINS.get((dataset_revision, split))
    data = _pinned_bytes(parquet, pin, f"BigCodeBench-Hard {dataset_revision} split {split}")
    assert pin is not None
    ids: list[str] = pq.read_table(pa.BufferReader(data), columns=["task_id"])[
        "task_id"
    ].to_pylist()
    _check_count(len(ids), pin, parquet)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{parquet}: duplicate task")
    return frozenset(ids)


@dataclass(frozen=True)
class _SecurityRow:
    """Identity of one LLMSecEval record; the prompt text is kept apart."""

    task_id: str
    index: int
    prompt_id: str
    filename: str
    family_id: str
    cwe: int
    language: str
    original: str
    weakness_name: str


def _security_rows(
    json_path: Path, revision: str, text_field: str
) -> tuple[list[_SecurityRow], str]:
    """Parse every record of the pinned file and check identifiers and families."""
    pin = LLMSECEVAL_PINS.get(revision)
    data = _pinned_bytes(json_path, pin, f"LLMSecEval {revision}")
    assert pin is not None
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{json_path}: expected a JSON array of records")
    _check_count(len(payload), pin, json_path)
    rows: list[_SecurityRow] = []
    folder_of: dict[str, str] = {}
    family_of: dict[str, str] = {}
    for index, record in enumerate(payload):
        # The internal identifier is the position in the array at the pinned
        # revision: it is stable and separates the records sharing a prompt ID.
        task_id = f"LSE-{index:03d}"
        prompt_id, filename = record["Prompt ID"], record["Filename"]
        original, language = record[text_field], record["Language"]
        match = _PROMPT_ID.fullmatch(prompt_id)
        if match is None:
            raise ValueError(f"{task_id}: unexpected prompt ID {prompt_id!r}")
        if language not in {"Python", "C"} or not isinstance(original, str):
            raise ValueError(f"{task_id}: unexpected language or text")
        if _SCENARIO_MARKER not in filename:
            raise ValueError(f"{task_id}: file name without a scenario folder")
        family = f"{match[1]}-{match[2]}-{match[3]}"
        folder = filename.split(_SCENARIO_MARKER)[0]
        # A family is one scenario of Pearce et al.: one folder, and vice versa.
        if folder_of.setdefault(family, folder) != folder:
            raise ValueError(f"{task_id}: family {family} spans more than one scenario folder")
        if family_of.setdefault(folder, family) != family:
            raise ValueError(f"{task_id}: scenario folder {folder} holds more than one family")
        rows.append(
            _SecurityRow(
                task_id=task_id,
                index=index,
                prompt_id=prompt_id,
                filename=filename,
                family_id=family,
                cwe=int(match[1]),
                language=language,
                original=original,
                weakness_name=record["CWE Name"],
            )
        )
    return rows, pin.file_sha256


def _selected(rows: Sequence[_SecurityRow], selection: str, placeholder: str) -> list[_SecurityRow]:
    """Apply one named selection.

    ``S2`` is every Python-origin prompt plus the C-origin prompts whose
    weakness also exists in Python. ``python-only`` is its Python part.
    ``c-memory`` is the complement used as pilot material: C-origin memory
    weaknesses, whole families only, leaving out a family in which a record
    lacks the placeholder (it could not be rendered like the others).
    """
    if selection == "python-only":
        return [row for row in rows if row.language == "Python"]
    if selection == "S2":
        return [
            row
            for row in rows
            if row.language == "Python" or (row.language == "C" and row.cwe not in MEMORY_CWES)
        ]
    if selection == "c-memory":
        memory = [row for row in rows if row.language == "C" and row.cwe in MEMORY_CWES]
        unusable = {row.family_id for row in memory if row.original.count(placeholder) != 1}
        return [row for row in memory if row.family_id not in unusable]
    raise ValueError(f"unknown selection {selection!r}; choose one of {SELECTIONS}")


def load_llmseceval(
    json_path: Path,
    revision: str,
    text_field: str = LLMSECEVAL_TEXT_FIELD,
    language: str = "Python",
    substitute: Mapping[str, str] | None = None,
    selection: str = "S2",
) -> list[SourceRecord]:
    """Load one selection of LLMSecEval prompts, replacing the language placeholder.

    Every placeholder of ``substitute`` (by default ``<language>`` -> ``language``)
    must occur exactly once in a selected prompt. Any other angle-bracket string,
    such as ``<url>`` or ``<username>``, is prompt content and stays untouched.

    Raises:
        ValueError: If the file is not the pinned one, a placeholder does not
            occur exactly once, or a prompt names its own weakness.
    """
    replacements = (
        dict(substitute) if substitute is not None else {LLMSECEVAL_PLACEHOLDER: language}
    )
    rows, file_sha256 = _security_rows(Path(json_path), revision, text_field)
    records: list[SourceRecord] = []
    for row in _selected(rows, selection, LLMSECEVAL_PLACEHOLDER):
        text = row.original
        transforms: list[Transform] = []
        for old, new in replacements.items():
            count = text.count(old)
            if count != 1:
                raise ValueError(
                    f"{row.task_id}: {old!r} occurs {count} time(s) in the prompt, expected 1"
                )
            text = text.replace(old, new)
            transforms.append(Transform("placeholder", old, new, count))
        if _WEAKNESS_LABEL.search(text) or row.weakness_name.lower() in text.lower():
            raise ValueError(f"{row.task_id}: the prompt names its own weakness")
        records.append(
            SourceRecord(
                task_id=row.task_id,
                text=text,
                source_sha256=_sha256_text(text),
                original_sha256=_sha256_text(row.original),
                benchmark=LLMSECEVAL,
                dataset_revision=revision,
                dataset_file_sha256=file_sha256,
                original_id=row.prompt_id,
                family_id=row.family_id,
                text_field=text_field,
                transforms=tuple(transforms),
            )
        )
    return records


def llmseceval_id_map(
    json_path: Path, revision: str, text_field: str = LLMSECEVAL_TEXT_FIELD
) -> list[dict[str, Any]]:
    """Map every internal identifier to its record, without any prompt text.

    The dataset is not redistributed with the study: identifiers, file names
    and hashes are enough to rebuild each source from the pinned file.
    """
    rows, _ = _security_rows(Path(json_path), revision, text_field)
    member = {
        name: {row.task_id for row in _selected(rows, name, LLMSECEVAL_PLACEHOLDER)}
        for name in SELECTIONS
    }
    return [
        {
            "task_id": row.task_id,
            "record_index": row.index,
            "prompt_id": row.prompt_id,
            "filename": row.filename,
            "family_id": row.family_id,
            "cwe": f"CWE-{row.cwe}",
            "language": row.language,
            "selections": [name for name in SELECTIONS if row.task_id in member[name]],
            "original_sha256": _sha256_text(row.original),
        }
        for row in rows
    ]


def write_sources(records: Sequence[SourceRecord], path: Path) -> None:
    """Write sources as JSON Lines to a new file; an existing file is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False, allow_nan=False) + "\n")


def read_sources(path: Path) -> list[SourceRecord]:
    """Read sources back, checking every text against its recorded hash.

    Lines are split on the line feed only: a prompt may hold other Unicode line
    separators, which JSON leaves unescaped.
    """
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for line in Path(path).read_bytes().decode("utf-8").split("\n"):
        if not line:
            continue
        row = json.loads(line)
        row["transforms"] = tuple(Transform(**item) for item in row["transforms"])
        record = SourceRecord(**row)
        if _sha256_text(record.text) != record.source_sha256:
            raise ValueError(f"{record.task_id}: the text does not match its recorded hash")
        if record.task_id in seen:
            raise ValueError(f"duplicate source {record.task_id}")
        seen.add(record.task_id)
        records.append(record)
    return records
