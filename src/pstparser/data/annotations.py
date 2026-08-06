"""Reading and writing a corpus annotated as text.

A worksheet is how the delivered corpus arrived, not how one has to be written.
Annotations authored here are kept in a text file instead, for two reasons that
a spreadsheet cannot give: a batch of them is reviewed in a diff rather than by
opening a binary, and a leaf is named by its path in the taxonomy rather than by
a column heading, so the file cannot inherit a heading that was misspelled once
and kept for compatibility.

The separator token has no counterpart here. It exists because a cell holds one
string and a leaf may hold several segments; a sequence needs no token to say
where it divides, and one written inside a segment is a mistake this module
refuses rather than interprets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from pstparser.data.cleaning import SEGMENT_SEPARATOR
from pstparser.data.errors import CorpusError
from pstparser.data.targets import RawRecord
from pstparser.pst.taxonomy import LEAF_PATHS

#: Key holding the records at the top level of the file.
RECORDS_KEY: Final = "records"

#: Keys a record may declare.
RECORD_KEYS: Final = ("prompt", "paradigm", "leaves")

#: Reasoning paradigms a prompt may be attributed to. They are the paradigms the
#: seed file groups its prompts by, and a source is free to attribute a prompt to
#: none of them: not every annotated prompt was written to exercise one.
PARADIGMS: Final[tuple[str, ...]] = ("zero_shot_cot", "few_shot_cot", "tree_of_thoughts")


class _StrictLoader(yaml.SafeLoader):
    """A loader that refuses a mapping declaring the same key twice.

    The default keeps the last of them without a word. On a file of annotations
    written by hand that is a leaf silently discarded, and what surfaces later is
    a coverage failure pointing at the record rather than at the mistake.
    """


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    """Build a mapping, refusing a key that occurs more than once."""
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"key {key!r} is declared twice", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def read_annotations(source_path: str | Path) -> list[RawRecord]:
    """Read a corpus annotated as text.

    Args:
        source_path: File holding the records.

    Returns:
        The records, in file order.

    Raises:
        CorpusError: If the file is missing, is not valid YAML, does not have the
            expected layout, or holds a value the annotation contract forbids.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise CorpusError(f"corpus not found: {source_path}")

    try:
        content = yaml.load(source_path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise CorpusError(f"{source_path} is not valid YAML: {exc}") from exc

    payload = content or {}
    if not isinstance(payload, dict):
        raise CorpusError(f"{source_path} must be a mapping holding {RECORDS_KEY!r}")

    entries = payload.get(RECORDS_KEY)
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise CorpusError(f"{RECORDS_KEY!r} must be a list in {source_path}")

    return [
        _read_record(entry, f"{source_path} record {position}")
        for position, entry in enumerate(entries)
    ]


def write_annotation_skeleton(prompts: Iterable[tuple[str, str]], destination: str | Path) -> Path:
    """Write prompts as a file awaiting annotation.

    The layout is the one the preparation reads, so what is filled in here is
    fed back without being transcribed. Transcription between two formats is
    where an extraction stops being exact.

    Args:
        prompts: Pairs of paradigm and prompt text, in the order to write them.
        destination: File to write. Parent directories are created.

    Returns:
        The path that was written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "paradigm": paradigm,
            "prompt": text,
            "leaves": dict.fromkeys(LEAF_PATHS, ""),
        }
        for paradigm, text in prompts
    ]
    body = yaml.dump(
        {RECORDS_KEY: records},
        Dumper=_BlockDumper,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    destination.write_text(_HEADER + body, encoding="utf-8", newline="\n")
    return destination


class _BlockDumper(yaml.SafeDumper):
    """A dumper writing multi-line text as a literal block."""


def _represent_str(dumper: _BlockDumper, data: str) -> yaml.ScalarNode:
    """Write a string, keeping a multi-line one readable."""
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _represent_str)

_HEADER = """\
# Prompts awaiting annotation.
#
# A leaf is named by its path in the taxonomy. A leaf holding several parts of
# the prompt that are not next to each other writes each as its own list entry;
# there is no separator token in this format. A leaf left empty holds nothing,
# and may equally be deleted.
#
# Every segment must be a passage of its own prompt, copied exactly. The
# preparation checks it and refuses a record that breaks it.

"""


def _read_record(entry: object, where: str) -> RawRecord:
    """Read one record.

    Args:
        entry: The record as parsed.
        where: File and position, for the error messages.

    Returns:
        The record.

    Raises:
        CorpusError: If the record does not have the expected layout.
    """
    if not isinstance(entry, dict):
        raise CorpusError(f"{where} is not a mapping")

    unknown = sorted(str(key) for key in entry if key not in RECORD_KEYS)
    if unknown:
        raise CorpusError(
            f"{where}: unknown keys: {', '.join(unknown)} "
            f"(expected: {', '.join(sorted(RECORD_KEYS))})"
        )

    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CorpusError(f"{where}: 'prompt' must be a non-empty string")

    paradigm = entry.get("paradigm")
    if paradigm is not None and paradigm not in PARADIGMS:
        raise CorpusError(
            f"{where}: unknown paradigm {paradigm!r}; expected one of: {', '.join(PARADIGMS)}"
        )

    return RawRecord(
        prompt=prompt,
        leaves=_read_leaves(entry.get("leaves"), where),
        paradigm=paradigm,
    )


def _read_leaves(entry: object, where: str) -> dict[str, list[str]]:
    """Read the annotation of one record.

    Args:
        entry: The ``leaves`` mapping as parsed, or nothing.
        where: File and position, for the error messages.

    Returns:
        The segments of every declared leaf, keyed by dotted path.

    Raises:
        CorpusError: If a leaf is not declared by the taxonomy, or holds
            something other than text.
    """
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise CorpusError(f"{where}: 'leaves' must be a mapping of leaf path to segments")

    for key in entry:
        if key not in LEAF_PATHS:
            raise CorpusError(
                f"{where}: unknown leaf {str(key)!r}; expected one of: {', '.join(LEAF_PATHS)}"
            )

    return {path: _read_segments(entry.get(path), where, path) for path in LEAF_PATHS}


def _read_segments(value: object, where: str, path: str) -> list[str]:
    """Read the segments held at one leaf.

    Nothing, empty text and an empty list all mean a leaf with no segments. A
    string is one segment, a list is one segment per entry. Anything else is
    refused rather than rendered as text: unlike a spreadsheet cell, whose type
    is not ours to choose, a value written here is written deliberately, and a
    number where text belongs is a mistake worth naming at once.

    Args:
        value: The leaf value as parsed.
        where: File and position, for the error messages.
        path: Dotted path of the leaf, for the error messages.

    Returns:
        The segments, in the order they were written.

    Raises:
        CorpusError: If the value is not text or a list of text, or if a segment
            carries the separator token.
    """
    if value is None:
        return []
    if isinstance(value, str):
        entries: list[object] = [value]
    elif isinstance(value, list):
        entries = list(value)
    else:
        raise CorpusError(
            f"{where}: leaf {path!r} must be text or a list of text, not {type(value).__name__}"
        )

    segments = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, str):
            raise CorpusError(
                f"{where}: leaf {path!r} segment {position} is not text "
                f"({type(entry).__name__}); quote it if the prompt holds it literally"
            )
        if SEGMENT_SEPARATOR in entry:
            raise CorpusError(
                f"{where}: leaf {path!r} segment {position} carries {SEGMENT_SEPARATOR!r}; "
                f"this format writes each segment as its own entry"
            )
        stripped = entry.strip()
        if stripped:
            segments.append(stripped)
    return segments


def describe_paradigms(records: Iterable[RawRecord]) -> Mapping[str, int]:
    """Count the records attributed to each paradigm.

    Args:
        records: The records to count.

    Returns:
        The counts, keyed by paradigm, leaving out those with no record.
    """
    counts: dict[str, int] = {}
    for record in records:
        if record.paradigm is not None:
            counts[record.paradigm] = counts.get(record.paradigm, 0) + 1
    return counts
