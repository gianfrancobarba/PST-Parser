"""Versioned extraction policies: the same rule for every condition, no repair."""

import pytest

from pstparser.empirical.extraction import (
    BIGCODEBENCH_SANITIZE,
    EXTRACTION_VERSION,
    FENCED_BLOCKS,
    LONGEST_BLOCK,
    POLICIES,
    ExtractedCode,
    extract,
    extract_fenced_blocks,
    extract_longest_block,
    extract_python,
    extraction_statuses,
)

SOLUTION = "import os\n\ndef task_func(path):\n    return os.path.basename(path)"
USAGE = "print(task_func('/tmp/a.txt'))"
BODY_ONLY = "    result = a + b\n    return result"


def fenced(code: str, label: str = "python") -> str:
    return f"```{label}\n{code}\n```"


def test_prose_before_and_after_a_block_is_dropped() -> None:
    raw = f"Here is the solution:\n\n{fenced(SOLUTION)}\n\nIt uses `os.path` only."

    result = extract_fenced_blocks(raw)

    assert result == ExtractedCode(SOLUTION, "ok", FENCED_BLOCKS, 1)


def test_two_blocks_are_joined_in_order_with_one_empty_line() -> None:
    raw = f"Solution:\n{fenced(SOLUTION)}\nUsage:\n{fenced(USAGE, '')}\n"

    result = extract_fenced_blocks(raw)

    assert result == ExtractedCode(SOLUTION + "\n\n" + USAGE, "ok", FENCED_BLOCKS, 2)


def test_plain_code_without_any_fence_is_accepted() -> None:
    assert extract_fenced_blocks(SOLUTION) == ExtractedCode(SOLUTION, "ok", FENCED_BLOCKS, 0)


def test_unclosed_block_is_an_invalid_fence() -> None:
    result = extract_fenced_blocks(f"Sure:\n```python\n{SOLUTION}")

    assert result == ExtractedCode(None, "invalid_fence", FENCED_BLOCKS, None)
    # Also when an earlier block was complete: nothing is guessed.
    assert extract_fenced_blocks(f"{fenced(SOLUTION)}\n```python\nx = 1").status == "invalid_fence"


@pytest.mark.parametrize(
    "raw,status,code,blocks",
    [
        ("", "empty", None, 0),
        ("  \n\t", "empty", None, 0),
        (fenced(" \n"), "empty", None, 1),
        ("I cannot help with that request.", "syntax_error", "I cannot help with that request.", 0),
        (fenced("def broken(:\n    pass"), "syntax_error", "def broken(:\n    pass", 1),
        (fenced("# TODO: implement"), "no_code", "# TODO: implement", 1),
        (fenced('{"code": "x"}', "json"), "no_code", None, 0),
        (fenced("ls -la", "bash") + "\n" + fenced("x = 1", "PY"), "ok", "x = 1", 1),
        # A response that is a JSON object is a valid Python expression: it is
        # extracted as it is, and fails later, where the program is judged.
        ('{"code": "def f(): pass"}', "ok", '{"code": "def f(): pass"}', 0),
        # Only the indented body of a function: not repaired.
        (BODY_ONLY, "syntax_error", BODY_ONLY, 0),
        (fenced(BODY_ONLY), "syntax_error", BODY_ONLY, 1),
    ],
)
def test_fenced_blocks_statuses(raw: str, status: str, code: str | None, blocks: int) -> None:
    assert extract_fenced_blocks(raw) == ExtractedCode(code, status, FENCED_BLOCKS, blocks)


def test_extracted_code_is_never_altered() -> None:
    code = "def f():\r\n\treturn  1   \n\n\n# trailing"

    assert extract_fenced_blocks(fenced(code)).code == code
    assert extract_fenced_blocks(code).code == code


def test_backticks_inside_a_line_are_not_a_fence() -> None:
    code = 'DOC = "use ```python fences"\nx = 1'

    assert extract_fenced_blocks(code) == ExtractedCode(code, "ok", FENCED_BLOCKS, 0)
    assert extract_fenced_blocks(fenced(code)).code == code


def test_longest_block_leaves_the_usage_example_out() -> None:
    raw = f"{fenced(USAGE)}\nand\n{fenced(SOLUTION)}\n"

    assert extract_longest_block(raw) == ExtractedCode(SOLUTION, "ok", LONGEST_BLOCK, 1)
    assert extract_fenced_blocks(raw).code == USAGE + "\n\n" + SOLUTION
    # A tie goes to the first block.
    assert extract_longest_block(f"{fenced('a = 1')}\n{fenced('b = 2')}").code == "a = 1"
    assert extract_longest_block(SOLUTION) == ExtractedCode(SOLUTION, "ok", LONGEST_BLOCK, 0)
    assert extract_longest_block("```\nx = 1").status == "invalid_fence"


def test_registry_holds_the_frozen_policies() -> None:
    assert POLICIES == (
        "python-single-block-v1",
        "python-fenced-blocks-v1",
        "python-longest-block-v1",
        "bigcodebench-sanitize",
    )
    raw = f"Text\n{fenced(SOLUTION)}"
    assert extract(raw, FENCED_BLOCKS) == extract_fenced_blocks(raw)
    assert extract(raw, LONGEST_BLOCK) == extract_longest_block(raw)
    assert extract(raw, EXTRACTION_VERSION) == extract_python(raw)
    assert extract(raw, EXTRACTION_VERSION).policy == "python-single-block-v1"
    with pytest.raises(ValueError, match="unknown extraction policy"):
        extract(raw, "first-block")


def test_sanitize_policy_is_never_run_outside_the_harness_image() -> None:
    with pytest.raises(ValueError, match="inside the harness image"):
        extract(SOLUTION, BIGCODEBENCH_SANITIZE)


def test_every_local_policy_is_recorded_for_one_output() -> None:
    raw = f"Here you go:\n{fenced(SOLUTION)}\n{fenced(USAGE)}"

    assert extraction_statuses(raw) == {
        "python-single-block-v1": "invalid_fence",
        "python-fenced-blocks-v1": "ok",
        "python-longest-block-v1": "ok",
    }
    assert extraction_statuses(SOLUTION) == dict.fromkeys(extraction_statuses(raw), "ok")


@pytest.mark.parametrize(
    "raw,status",
    [
        ("", "empty"),
        ("# comment", "empty"),
        ("def bad(:", "syntax_error"),
        ("```python\ndef f():\n    return 1\n```", "ok"),
        ("def f():\n    return 1", "ok"),
        ("Here is code:\n```python\nx=1\n```", "invalid_fence"),
        ("```python\nx=1", "invalid_fence"),
        ("```\nx=1\n```\n```\ny=2\n```", "invalid_fence"),
    ],
)
def test_single_block_policy_is_unchanged(raw: str, status: str) -> None:
    result = extract_python(raw)

    assert result.status == status
    assert result.policy == "python-single-block-v1" and result.blocks is None
