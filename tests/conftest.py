"""Shared fixtures.

Configuration files reference their inputs relative to the repository root, so
the whole suite runs with that directory as the working directory regardless of
where pytest was invoked from.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "tests" / "assets"


@pytest.fixture(autouse=True)
def _run_from_repo_root() -> Iterator[None]:
    """Execute every test with the repository root as working directory."""
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


@pytest.fixture
def assets_dir() -> Path:
    """Directory holding the committed test fixtures."""
    return ASSETS


@pytest.fixture
def config_dir(assets_dir: Path) -> Path:
    """Directory holding the fixture configurations."""
    return assets_dir / "configs"


@pytest.fixture
def tiny_corpus(assets_dir: Path) -> Path:
    """Spreadsheet used by the data preparation tests."""
    return assets_dir / "tiny_corpus.xlsx"
