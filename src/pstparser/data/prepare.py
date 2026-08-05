"""Turning the annotated spreadsheet into the records consumed by training."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pstparser.config import DataConfig
from pstparser.data.excel import CorpusError, iter_rows, read_corpus
from pstparser.data.fixes import AnnotationFix, apply_fixes, load_fixes
from pstparser.data.quality import QualityReport, check_corpus
from pstparser.data.splits import Split, group_key, make_split, save_split
from pstparser.data.targets import build_target
from pstparser.pst.alignment import DEFAULT_LEVELS, MatchLevel
from pstparser.pst.target import serialise_target
from pstparser.pst.taxonomy import LEAF_PATHS

RECORDS_FILE = "records.jsonl"
QUALITY_REPORT_FILE = "quality_report.json"


@dataclass(frozen=True)
class PreparedRecord:
    """One prompt paired with its serialised target.

    Attributes:
        index: Position of the record in the source corpus. Splits refer to
            records by this value.
        prompt: The raw, unsegmented prompt.
        target: The serialised target tree.
    """

    index: int
    prompt: str
    target: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the record."""
        return {"index": self.index, "prompt": self.prompt, "target": self.target}


@dataclass(frozen=True)
class PreparationResult:
    """Artefacts produced by a corpus preparation run.

    Attributes:
        records: The prepared records, in corpus order.
        quality: Outcome of the integrity check.
        split: The training and evaluation partition.
        fixes: Corrections applied to the annotations before conversion.
        records_path: File the records were written to.
        report_path: File the integrity report was written to.
        split_dir: Directory the partition was written to.
    """

    records: list[PreparedRecord]
    quality: QualityReport
    split: Split
    fixes: list[AnnotationFix]
    records_path: Path
    report_path: Path
    split_dir: Path


def prepare_corpus(
    config: DataConfig,
    levels: Sequence[MatchLevel] = DEFAULT_LEVELS,
) -> PreparationResult:
    """Read, convert, check and partition the annotated corpus.

    Args:
        config: Corpus location and preparation options.
        levels: Normalisations the integrity check may use to locate a segment.
            They come from the experiment, so the corpus is checked with the
            same tolerance the metrics are later computed with.

    Returns:
        The prepared records together with the paths they were written to.

    Raises:
        CorpusError: If the corpus is unreadable, if a declared leaf has no
            source column, or if the integrity check fails while
            ``config.quality.fail_on_issues`` is set.
    """
    missing_leaves = [path for path in LEAF_PATHS if path not in config.column_mapping]
    if missing_leaves:
        raise CorpusError(f"column_mapping does not cover: {', '.join(missing_leaves)}")

    frame = read_corpus(
        config.source_path,
        config.sheet_name,
        required_columns=[config.prompt_column, *config.column_mapping.values()],
    )

    fixes = load_fixes(config.fixes_path) if config.fixes_path is not None else []
    if fixes:
        frame = apply_fixes(frame, fixes)

    records = [
        PreparedRecord(
            index=index,
            prompt=str(row[config.prompt_column]),
            target=serialise_target(build_target(row, config.column_mapping)),
        )
        for index, row in enumerate(iter_rows(frame))
    ]

    report = check_corpus(
        prompts=[record.prompt for record in records],
        targets=[record.target for record in records],
        min_coverage_ratio=config.quality.min_coverage_ratio,
        levels=levels,
    )
    if config.quality.fail_on_issues and not report.passed:
        raise CorpusError(
            f"integrity check reported {len(report.issues)} issues "
            f"over {report.affected} of {report.total} records"
        )

    kept = deduplicate(records) if config.split.deduplicate else records
    split = make_split(
        keys=[group_key(record.prompt) for record in kept],
        val_fraction=config.split.val_fraction,
        test_fraction=config.split.test_fraction,
        random_state=config.split.random_state,
    )
    split = Split(
        train=[kept[position].index for position in split.train],
        val=[kept[position].index for position in split.val],
        test=[kept[position].index for position in split.test],
    )

    records_path = write_records(records, Path(config.processed_dir) / RECORDS_FILE)
    report_path = Path(config.processed_dir) / QUALITY_REPORT_FILE
    report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    save_split(split, config.split.output_dir)

    return PreparationResult(
        records=records,
        quality=report,
        split=split,
        fixes=fixes,
        records_path=records_path,
        report_path=report_path,
        split_dir=Path(config.split.output_dir),
    )


def deduplicate(records: Sequence[PreparedRecord]) -> list[PreparedRecord]:
    """Drop the records that repeat both a prompt and its annotation.

    A record identical to one already seen carries no further signal: it only
    lets the same example be counted twice, once on each side of a partition.
    Records that share a prompt but were annotated differently are all kept, and
    the partition puts them on the same side instead.

    Args:
        records: The prepared records, in corpus order.

    Returns:
        The records to keep, in corpus order.
    """
    seen: set[tuple[str, str]] = set()
    kept = []
    for record in records:
        signature = (record.prompt, record.target)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(record)
    return kept


def write_records(records: Iterable[PreparedRecord], destination: str | Path) -> Path:
    """Write records as one JSON object per line.

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


def load_records(source: str | Path) -> list[PreparedRecord]:
    """Read records previously written by :func:`write_records`.

    Args:
        source: File holding one JSON object per line.

    Returns:
        The records, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"records file not found: {source}")

    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            PreparedRecord(
                index=int(payload["index"]),
                prompt=str(payload["prompt"]),
                target=str(payload["target"]),
            )
        )
    return records


def select(records: Iterable[PreparedRecord], indices: Iterable[int]) -> list[PreparedRecord]:
    """Pick the records belonging to one side of a partition.

    Args:
        records: The full set of records.
        indices: Indices to keep, in the order they should be returned.

    Returns:
        The selected records.

    Raises:
        KeyError: If an index has no corresponding record.
    """
    by_index = {record.index: record for record in records}
    return [by_index[index] for index in indices]
