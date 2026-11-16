"""Journals and variant files are JSON Lines split on the line feed only (QUESTIONI Q-04).

Files are written with ensure_ascii=False, so U+2028, U+2029 and U+0085 stay
unescaped inside a record, and str.splitlines would cut the record there. The
other separators of str.splitlines (VT, FF, U+001C-U+001E) are control
characters, which json.dumps always escapes.
"""

import json

import pytest

from pstparser.empirical.cli import _rows
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.rendering import render_prompt_variants

SEPARATORS = ["\u2028", "\u2029", "\x85"]


@pytest.mark.parametrize("separator", SEPARATORS)
def test_a_journal_with_a_unicode_separator_in_an_output_resumes(tmp_path, separator) -> None:
    text = f"Write{separator}code."
    plan = request_plan(
        render_prompt_variants(
            CanonicalPrompt(
                "t", text, (SpanAnnotation("main_instruction", 0, len(text), text),), "v"
            )
        ),
        [0],
        1,
    )

    def output(request):
        return {
            "raw_output": f"x = 1{separator}y = 2",
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "eos",
        }

    written = run_journal(plan, {}, tmp_path, output)
    assert separator in (tmp_path / "generations.jsonl").read_text(encoding="utf-8")

    def refuse(request):
        raise AssertionError("a completed request was generated again")

    assert run_journal(plan, {}, tmp_path, refuse) == written


@pytest.mark.parametrize("separator", SEPARATORS)
def test_variant_rows_keep_unicode_separators(tmp_path, separator) -> None:
    rows = [{"text": f"a{separator}b"}, {"text": "c"}]
    path = tmp_path / "rows.jsonl"
    path.write_bytes(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    )
    assert _rows(path) == rows


def test_blank_lines_are_still_skipped_by_variant_rows(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a": 1}\n\n{"a": 2}\n')
    assert _rows(path) == [{"a": 1}, {"a": 2}]
