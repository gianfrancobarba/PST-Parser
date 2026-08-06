"""Reading credentials from the environment file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pstparser.config import load_env_file, parse_env


def test_a_plain_assignment_is_read() -> None:
    assert parse_env("TOKEN=abc123") == {"TOKEN": "abc123"}


@pytest.mark.parametrize(
    "line",
    [
        "TOKEN=abc123",
        "TOKEN= abc123",
        "TOKEN = abc123 ",
        'TOKEN="abc123"',
        "TOKEN='abc123'",
        "export TOKEN=abc123",
    ],
)
def test_spacing_and_quoting_do_not_change_the_value(line: str) -> None:
    # Whichever way it was written, the credential is the same string.
    assert parse_env(line) == {"TOKEN": "abc123"}


def test_a_quote_inside_the_value_is_kept() -> None:
    assert parse_env('TOKEN=ab"cd') == {"TOKEN": 'ab"cd'}


def test_an_unbalanced_quote_is_kept() -> None:
    assert parse_env('TOKEN="abc') == {"TOKEN": '"abc'}


def test_comments_and_blank_lines_are_ignored() -> None:
    text = "# a comment\n\nTOKEN=abc\n   \n# TOKEN=wrong\n"

    assert parse_env(text) == {"TOKEN": "abc"}


def test_a_line_without_an_assignment_is_ignored() -> None:
    assert parse_env("not an assignment\nTOKEN=abc") == {"TOKEN": "abc"}


def test_the_last_assignment_wins() -> None:
    assert parse_env("TOKEN=first\nTOKEN=second") == {"TOKEN": "second"}


def test_an_empty_value_is_read_but_not_applied(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("EMPTY=\n", encoding="utf-8")

    assert parse_env("EMPTY=") == {"EMPTY": ""}
    assert load_env_file(path) == {}


def test_the_file_fills_a_variable_that_is_not_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PSTPARSER_TEST_TOKEN", raising=False)
    path = tmp_path / ".env"
    path.write_text("PSTPARSER_TEST_TOKEN=from-file\n", encoding="utf-8")

    applied = load_env_file(path)

    assert applied == {"PSTPARSER_TEST_TOKEN": "from-file"}
    assert os.environ["PSTPARSER_TEST_TOKEN"] == "from-file"


def test_the_environment_wins_over_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exporting a variable for one command is how an override is expressed, so
    # the file must not quietly take precedence over it.
    monkeypatch.setenv("PSTPARSER_TEST_TOKEN", "from-environment")
    path = tmp_path / ".env"
    path.write_text("PSTPARSER_TEST_TOKEN=from-file\n", encoding="utf-8")

    assert load_env_file(path) == {}
    assert os.environ["PSTPARSER_TEST_TOKEN"] == "from-environment"


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent") == {}
