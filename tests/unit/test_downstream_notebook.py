"""The downstream Colab notebook (M17): valid, a launcher only, without the parser stack."""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "downstream.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_bytes().decode("utf-8"))


def text(cell: dict) -> str:
    return "".join(cell["source"])


def code_cells(notebook: dict) -> list[str]:
    return [text(c) for c in notebook["cells"] if c["cell_type"] == "code"]


def generator_section(notebook: dict) -> list[str]:
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and text(cell).startswith("## Parser"):
            break
        cells.append(text(cell))
    return cells


def test_the_notebook_is_valid_json_with_clean_code_cells(notebook) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(code_cells(notebook)) == 7
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            ast.parse(text(cell))
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_the_file_has_lf_line_endings_only() -> None:
    raw = NOTEBOOK.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")


def test_the_generator_section_has_no_unsloth(notebook) -> None:
    section = generator_section(notebook)
    assert len(section) < len(notebook["cells"])  # the parser section exists and is apart
    assert not any("unsloth" in cell.lower() for cell in section)
    install = next(cell for cell in section if "uv" in cell and "sync" in cell)
    assert '"--extra", "cu128", "--extra", "train"' in install
    assert "structured" not in install


def test_the_source_archive_is_verified_against_the_expected_tree(notebook) -> None:
    cell = next(c for c in code_cells(notebook) if "verify_archive.py" in c)
    assert '"--expected-tree"' in cell and "check=True" in cell
    assert "hashlib.sha256(archive_bytes).hexdigest()" in cell
    assert cell.index("verify_archive.py") < cell.index("os.chdir(PROJECT)")


def test_the_run_identity_binds_archive_freeze_and_revision(notebook) -> None:
    cell = next(c for c in code_cells(notebook) if "run-identity.json" in c)
    for key in ("source_archive_sha256", "freeze_sha256", "model_revision", "inputs_sha256"):
        assert f'"{key}"' in cell
    assert "RUN_ID gia associata a un'altra identita" in cell
    default = re.search(r'^RUN_ID = "([^"]+)"', cell, re.MULTILINE)
    pattern = re.search(r'r"(ds-[^"]+)", RUN_ID', cell)
    assert default and pattern and re.fullmatch(pattern.group(1), default.group(1))
    assert 'MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"' in cell


def test_final_runs_verify_the_freeze_and_every_journal_uses_the_project_command(notebook) -> None:
    cells = code_cells(notebook)
    freeze = next(c for c in cells if '"freeze-verify"' in c)
    assert 'if PURPOSE == "final":' in freeze
    generate = next(c for c in cells if '"generate"' in c)
    assert "for name in JOURNALS:" in generate and "check=True" in generate
    assert 'EMPIRICAL = [*UV_RUN, "python", "-m", "pstparser.empirical"]' in "\n".join(cells)


def test_no_link_to_unversioned_documents_and_no_secrets(notebook) -> None:
    raw = NOTEBOOK.read_text(encoding="utf-8")
    assert "docs/" not in raw
    assert "HF_TOKEN" not in raw and "load_dotenv" not in raw
