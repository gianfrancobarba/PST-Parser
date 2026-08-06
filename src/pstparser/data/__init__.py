"""Corpus reading, conversion, integrity checking and partitioning."""

from pstparser.data.alignments import (
    ALIGNMENTS_FILE,
    AlignmentRecord,
    load_alignments,
    write_alignments,
)
from pstparser.data.annotations import (
    PARADIGMS,
    describe_paradigms,
    read_annotations,
    write_annotation_skeleton,
)
from pstparser.data.cleaning import SEGMENT_SEPARATOR, clean_field
from pstparser.data.errors import CorpusError
from pstparser.data.excel import iter_rows, read_corpus, sheet_names
from pstparser.data.fixes import AnnotationFix, apply_fixes, describe, load_fixes
from pstparser.data.predictions import PredictionRecord, load_predictions, write_predictions
from pstparser.data.prepare import (
    PreparationResult,
    PreparedRecord,
    deduplicate,
    load_records,
    prepare_corpus,
    read_source,
    select,
    write_records,
)
from pstparser.data.quality import QualityIssue, QualityReport, check_corpus
from pstparser.data.splits import Split, group_key, load_split, make_split, save_split
from pstparser.data.targets import RawRecord, build_leaves, build_target

__all__ = [
    "ALIGNMENTS_FILE",
    "PARADIGMS",
    "SEGMENT_SEPARATOR",
    "AlignmentRecord",
    "AnnotationFix",
    "CorpusError",
    "PredictionRecord",
    "PreparationResult",
    "PreparedRecord",
    "QualityIssue",
    "QualityReport",
    "RawRecord",
    "Split",
    "apply_fixes",
    "build_leaves",
    "build_target",
    "check_corpus",
    "clean_field",
    "deduplicate",
    "describe",
    "describe_paradigms",
    "group_key",
    "iter_rows",
    "load_alignments",
    "load_fixes",
    "load_predictions",
    "load_records",
    "load_split",
    "make_split",
    "prepare_corpus",
    "read_annotations",
    "read_corpus",
    "read_source",
    "save_split",
    "select",
    "sheet_names",
    "write_alignments",
    "write_annotation_skeleton",
    "write_predictions",
    "write_records",
]
