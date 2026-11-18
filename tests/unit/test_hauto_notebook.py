"""The H-auto parser Colab notebook (C-24, D-01): a launcher of the pinned parse-prompts."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "hauto-parser.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_bytes().decode("utf-8"))


def code_cells(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_the_notebook_is_valid_with_clean_code_cells_and_lf_endings(notebook) -> None:
    assert notebook["nbformat"] == 4 and notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    raw = NOTEBOOK.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")


def test_the_parser_runs_on_the_hf_stack_without_unsloth(notebook) -> None:
    cells = code_cells(notebook)
    assert not any("unsloth" in cell.lower() and "--extra" in cell for cell in cells)
    install = next(cell for cell in cells if "uv" in cell and "sync" in cell)
    assert '"--extra", "cu128", "--extra", "train"' in install


def test_the_source_archive_is_verified_and_the_gpu_is_the_declared_one(notebook) -> None:
    cells = code_cells(notebook)
    source = next(c for c in cells if "verify_archive.py" in c)
    assert '"--expected-tree"' in source and "check=True" in source
    gpu = next(c for c in cells if "nvidia-smi" in c)
    assert 'DECLARED_GPU = "NVIDIA L4"' in gpu


def test_parse_prompts_is_pinned_resumable_and_frozen_for_final(notebook) -> None:
    cells = "\n".join(code_cells(notebook))
    for option in ('"--purpose"', '"--identity"', '"--parser-model"', '"--adapter"', '"--resume"'):
        assert option in cells
    assert 'arguments += ["--freeze", str(FREEZE)]' in cells
    assert '"model-digests"' in cells and '"freeze-verify"' in cells
    assert 'CONFIG = PROJECT / "configs" / "experiments" / "hauto-parser.yaml"' in cells
    assert (ROOT / "configs" / "experiments" / "hauto-parser.yaml").is_file()
    assert (ROOT / "protocol" / "hauto" / "parser.json").is_file()


def test_no_link_to_unversioned_documents_and_no_secrets() -> None:
    raw = NOTEBOOK.read_text(encoding="utf-8")
    assert "docs/" not in raw
    assert "HF_TOKEN" not in raw and "load_dotenv" not in raw
