"""Condition-independent taxonomy rules and a layout-robust fenced extraction.

Counterexamples of the independent review,
inverted: each one documented a defect of the v1 rules.
"""

from typing import Any

from pstparser.empirical import bigcodebench as bcb
from pstparser.empirical.extraction import (
    FENCED_BLOCKS,
    LONGEST_BLOCK,
    extract_fenced_blocks,
    extract_longest_block,
)

CODE_OK = "import json\ndef task_func(x):\n    return x + 1\n"
ASSERT_FAIL = {"test_a": "Traceback (most recent call last):\nAssertionError: 1 != 2"}


def gen(raw: str, finish: str = "stop") -> dict[str, Any]:
    return {"finish_reason": finish, "raw_output": raw}


def test_rule_versions() -> None:
    assert bcb.JSON_RESPONSE_RULE == "json-response-v2"
    assert bcb.PROMPT_AS_DATA_RULE == "prompt-as-data-v2"
    assert FENCED_BLOCKS == "python-fenced-blocks-v2"
    assert LONGEST_BLOCK == "python-longest-block-v2"


# ------------------------------------------------------------------- json-response-v2


def test_prose_quoting_pst_keys_keeps_its_semantic_category() -> None:
    raw = (
        'From the `"main_instruction":` and the `"context":` fields I understand the task.\n'
        "```python\n" + CODE_OK + "```\n"
    )
    assert not bcb.is_json_response(raw)
    assert bcb.categorize(gen(raw), CODE_OK, "task_func", "fail", ASSERT_FAIL) == (
        "assertion_failure"
    )


def test_the_envelope_key_counts_like_the_pst_keys() -> None:
    echo_e = 'The input is {"prompt": "Sum two numbers."}\n```python\n' + CODE_OK + "```"
    echo_h = (
        'The input is {"prompt": {"main_instruction": [{"position": 0, "phrase": "Sum."}]}}\n'
        "```python\n" + CODE_OK + "```"
    )
    assert bcb.is_json_response(echo_e)
    assert bcb.is_json_response(echo_h)


def test_markdown_answer_starting_with_a_bracket_is_not_json() -> None:
    raw = "[Solution]\n```python\n" + CODE_OK + "```"
    assert not bcb.is_json_response(raw)
    assert bcb.categorize(gen(raw), CODE_OK, "task_func", "fail", ASSERT_FAIL) == (
        "assertion_failure"
    )


def test_a_json_answer_or_a_json_block_echoing_the_prompt_is_json_response() -> None:
    assert bcb.is_json_response('{"code": "def task_func(): pass"}')
    assert bcb.is_json_response("[1, 2, 3]")
    block = '```json\n{"prompt": {"main_instruction": []}}\n```'
    assert bcb.is_json_response(block)
    # A JSON example the task asks for, without any key of the prompt, is not.
    assert not bcb.is_json_response(
        'Example output:\n```json\n{"total": 3}\n```\n```python\n' + CODE_OK + "```"
    )


# ------------------------------------------------------------------- prompt-as-data-v2


def test_legitimate_prompt_and_position_keys_are_not_prompt_as_data() -> None:
    legit = (
        "def task_func(df, cfg):\n"
        "    body = {'prompt': cfg['text'], 'max_tokens': 5}\n"
        "    return df['position'].mean()\n"
    )
    assert not bcb.is_prompt_as_data(legit)


def test_get_on_a_loaded_prompt_is_prompt_as_data() -> None:
    get_style = (
        "import json\nSPEC = open('spec.json').read()\n"
        "def task_func():\n    d = json.loads(SPEC)\n    return d.get('main_instruction')\n"
    )
    assert bcb.is_prompt_as_data(get_style)
    loaded = "import json\ndef task_func(s):\n    return json.loads(s)['prompt']\n"
    assert bcb.is_prompt_as_data(loaded)


# ----------------------------------------------------------------- fenced extraction v2


def test_complete_blocks_survive_a_truncated_last_block() -> None:
    raw = "```python\nimport os\nos.system('ls')\n```\nUsage:\n```python\nprint(1"
    result = extract_fenced_blocks(raw)
    assert (result.status, result.code, result.blocks) == ("ok", "import os\nos.system('ls')", 1)
    assert result.open_fence is True
    assert extract_longest_block(raw).status == "ok"


def test_python3_label_is_python() -> None:
    raw = "```python3\nimport os\nos.system('ls')\n```"
    result = extract_fenced_blocks(raw)
    assert (result.status, result.open_fence) == ("ok", False)


def test_a_lone_open_fence_is_still_invalid() -> None:
    assert extract_fenced_blocks("```python\nprint(1").status == "invalid_fence"
