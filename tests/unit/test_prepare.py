"""End-to-end preparation of a corpus, and regression locks on the shipped one."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pstparser.config import load_experiment
from pstparser.data import (
    CorpusError,
    PreparationResult,
    apply_fixes,
    build_target,
    iter_rows,
    load_fixes,
    load_records,
    load_split,
    prepare_corpus,
    read_corpus,
    select,
    sheet_names,
)
from pstparser.pst import serialise_target

#: Digest of the serialised targets produced from the shipped corpus, newline
#: separated. Any change to the taxonomy, to cell normalisation, to the
#: corrections or to the corpus itself moves this value.
SHIPPED_TARGETS_DIGEST = "513e1e40745325afae6130156d5649d2e2cab371ca24034486f76fd64dad5154"

#: Digest of the system message the model is conditioned on.
SYSTEM_PROMPT_DIGEST = "bc2282e04b35c1055a36edc30050286e3ead4bfec159385c09f0462763fd79d7"

SHIPPED_RECORD_COUNT = 975


@pytest.fixture
def prepared(config_dir: Path, tmp_path: Path) -> PreparationResult:
    """Prepare the tiny corpus into a temporary directory."""
    config = load_experiment(
        config_dir / "valid.yaml",
        overrides=[
            f"data.processed_dir={(tmp_path / 'processed').as_posix()}",
            f"data.split.output_dir={(tmp_path / 'splits').as_posix()}",
        ],
        root=config_dir,
    )
    return prepare_corpus(config.data)


def test_every_row_becomes_a_record(prepared: PreparationResult) -> None:
    assert len(prepared.records) == 7
    assert [record.index for record in prepared.records] == list(range(7))


def test_records_carry_the_raw_prompt(prepared: PreparationResult) -> None:
    assert prepared.records[1].prompt == "Fix this bug."


def test_targets_are_parseable(prepared: PreparationResult) -> None:
    for record in prepared.records:
        assert "prompt" in json.loads(record.target)


def test_a_cell_marked_as_disjoint_becomes_two_segments(prepared: PreparationResult) -> None:
    tree = json.loads(prepared.records[2].target)["prompt"]

    assert tree["context"]["data"] == ["class A: pass", "class B: pass"]


def test_a_list_shaped_cell_is_kept_as_the_prompt_writes_it(
    prepared: PreparationResult,
) -> None:
    # The prompt itself contains the brackets and quotes, so the annotation is
    # a faithful span of it. Reading the cell as a literal would replace that
    # span with two words the prompt never held on their own.
    tree = json.loads(prepared.records[6].target)["prompt"]

    assert tree["context"]["data"] == ["['alpha', 'beta']"]


def test_missing_annotation_becomes_an_empty_leaf(prepared: PreparationResult) -> None:
    tree = json.loads(prepared.records[3].target)["prompt"]

    assert tree["main_instruction"] == []
    assert tree["context"]["data"] == ["error: cannot open file"]


def test_under_annotated_row_is_reported(prepared: PreparationResult) -> None:
    # Row 4 is annotated only in part, and is the only row that is.
    assert [issue.index for issue in prepared.quality.issues] == [4]


def test_artefacts_are_written(prepared: PreparationResult) -> None:
    assert prepared.records_path.is_file()
    assert prepared.report_path.is_file()
    assert json.loads(prepared.report_path.read_text(encoding="utf-8"))["total"] == 7


def test_records_round_trip_through_disk(prepared: PreparationResult) -> None:
    assert load_records(prepared.records_path) == prepared.records


def test_partition_is_written_and_reloadable(prepared: PreparationResult) -> None:
    assert load_split(prepared.split_dir) == prepared.split


def test_selection_follows_the_requested_order(prepared: PreparationResult) -> None:
    selected = select(prepared.records, [3, 0])

    assert [record.index for record in selected] == [3, 0]


def test_incomplete_column_mapping_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    config = load_experiment(config_dir / "valid.yaml", root=config_dir)
    mapping = dict(config.data.column_mapping)
    del mapping["context.role"]
    broken = config.data.model_copy(update={"column_mapping": mapping})

    with pytest.raises(CorpusError, match="does not cover"):
        prepare_corpus(broken)


def test_failing_integrity_check_can_abort(config_dir: Path, tmp_path: Path) -> None:
    config = load_experiment(
        config_dir / "valid.yaml",
        overrides=[
            "data.quality.fail_on_issues=true",
            f"data.processed_dir={(tmp_path / 'processed').as_posix()}",
        ],
        root=config_dir,
    )

    with pytest.raises(CorpusError, match="integrity check reported"):
        prepare_corpus(config.data)


def test_unknown_worksheet_is_rejected(tiny_corpus: Path) -> None:
    with pytest.raises(CorpusError, match="worksheet 'absent' not found"):
        read_corpus(tiny_corpus, "absent", required_columns=[])


def test_missing_column_is_rejected(tiny_corpus: Path) -> None:
    with pytest.raises(CorpusError, match="missing columns"):
        read_corpus(tiny_corpus, "corpus", required_columns=["NO SUCH COLUMN"])


def test_worksheets_are_listed(tiny_corpus: Path) -> None:
    assert sheet_names(tiny_corpus) == ["corpus"]


def test_system_prompt_is_unchanged() -> None:
    digest = hashlib.sha256(Path("prompts/system_pst.md").read_bytes()).hexdigest()

    assert digest == SYSTEM_PROMPT_DIGEST


def test_shipped_corpus_produces_stable_targets() -> None:
    config = load_experiment("configs/experiments/baseline.yaml")
    frame = read_corpus(
        config.data.source_path,
        config.data.sheet_name,
        required_columns=[config.data.prompt_column, *config.data.column_mapping.values()],
    )
    assert config.data.fixes_path is not None
    frame = apply_fixes(frame, load_fixes(config.data.fixes_path))

    targets = [
        serialise_target(build_target(row, config.data.column_mapping)) for row in iter_rows(frame)
    ]
    digest = hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest()

    assert len(targets) == SHIPPED_RECORD_COUNT
    assert digest == SHIPPED_TARGETS_DIGEST
