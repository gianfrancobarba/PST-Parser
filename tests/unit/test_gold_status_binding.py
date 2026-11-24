"""Variants carry the state of their gold, and final runs need frozen gold.

``render`` writes next to the variants a record (``<variants>.render.json``)
with the state the canonical file was joined in (from the ``canonicalize``
manifest) and the hash of the annotation file. A final run is accepted only if
the freeze lists that record, the record says ``frozen`` and the freeze also
lists the annotation file.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.freeze import build_freeze, require_final_freeze, write_freeze
from pstparser.repro.manifest import digest_file

TEXT = "Sum the list."


def write_canonical(directory: Path, status: str | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    prompt = CanonicalPrompt(
        "BigCodeBench/1", TEXT, (SpanAnnotation("main_instruction", 0, len(TEXT), TEXT),), "v"
    )
    canonical = directory / "canonical.jsonl"
    row = {
        "task_id": prompt.task_id,
        "text": prompt.text,
        "annotation_version": prompt.annotation_version,
        "annotations": [vars(span) for span in prompt.annotations],
    }
    canonical.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    annotations = directory / "annotations.jsonl"
    annotations.write_text('{"status": "x"}\n', encoding="utf-8", newline="\n")
    if status is not None:
        manifest = {
            "stage": "empirical-canonicalize",
            "require_status": status,
            "annotations_sha256": digest_file(annotations),
            "canonical_sha256": digest_file(canonical),
        }
        (directory / "empirical-canonicalize_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8", newline="\n"
        )
    return canonical


def render(canonical: Path, out: Path) -> dict:
    result = CliRunner().invoke(app, ["render", str(canonical), str(out)])
    assert result.exit_code == 0, result.exception
    return json.loads(Path(f"{out}.render.json").read_text(encoding="utf-8"))


def test_render_records_the_state_of_the_gold(tmp_path: Path) -> None:
    draft = render(write_canonical(tmp_path / "d", "draft"), tmp_path / "d" / "v.jsonl")
    assert draft["gold_status"] == "draft"
    assert draft["variants_sha256"] == digest_file(tmp_path / "d" / "v.jsonl")
    assert draft["annotations_sha256"] == digest_file(tmp_path / "d" / "annotations.jsonl")
    unknown = render(write_canonical(tmp_path / "u", None), tmp_path / "u" / "v.jsonl")
    assert unknown["gold_status"] is None


def frozen_repository(root: Path, status: str, list_gold: bool = True) -> tuple[Path, Path, Path]:
    canonical = write_canonical(root / "protocol" / "variants", status)
    variants = root / "protocol" / "variants" / "v.jsonl"
    render(canonical, variants)
    (root / "src" / "pstparser").mkdir(parents=True)
    (root / "src" / "pstparser" / "__init__.py").write_text("", encoding="utf-8")
    (root / "uv.lock").write_text("lock\n", encoding="utf-8", newline="\n")
    configuration = root / "protocol" / "generate.json"
    configuration.write_text("{}\n", encoding="utf-8", newline="\n")
    items = [
        "protocol/generate.json",
        "protocol/variants/v.jsonl",
        "protocol/variants/v.jsonl.render.json",
        "protocol/variants/canonical.jsonl",
        "protocol/variants/empirical-canonicalize_manifest.json",
    ]
    if list_gold:
        items.append("protocol/variants/annotations.jsonl")
    meta = {"protocol_version": "v", "frozen_by": "x"}
    freeze = write_freeze(
        build_freeze(root, items, meta), root / "protocol" / "protocol-freeze.json"
    )
    return freeze, variants, configuration


def test_a_final_run_on_frozen_gold_is_accepted(tmp_path: Path) -> None:
    freeze, variants, configuration = frozen_repository(tmp_path, "frozen")
    assert require_final_freeze(freeze, variants, configuration)["format"]


def test_a_final_run_on_draft_gold_is_refused(tmp_path: Path) -> None:
    freeze, variants, configuration = frozen_repository(tmp_path, "draft")
    with pytest.raises(ValueError, match="frozen gold"):
        require_final_freeze(freeze, variants, configuration)


def test_a_final_run_whose_gold_is_not_in_the_freeze_is_refused(tmp_path: Path) -> None:
    freeze, variants, configuration = frozen_repository(tmp_path, "frozen", list_gold=False)
    with pytest.raises(ValueError, match="annotation file"):
        require_final_freeze(freeze, variants, configuration)
