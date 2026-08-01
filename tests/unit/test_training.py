"""Dataset construction and conversation rendering, without loading a model."""

from __future__ import annotations

from typing import Any

import pytest

from pstparser.data import PreparedRecord
from pstparser.models import resolve_precision
from pstparser.training import build_dataset, make_formatting_func

SYSTEM_PROMPT = "Segment the prompt."


class StubTokenizer:
    """Minimal stand-in rendering a conversation as one line per turn."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render a conversation, recording how it was called."""
        self.last_call: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)


@pytest.fixture
def records() -> list[PreparedRecord]:
    return [
        PreparedRecord(index=0, prompt="Fix the bug.", target='{"prompt": {}}'),
        PreparedRecord(index=1, prompt="Summarise this.", target='{"prompt": {"a": []}}'),
    ]


def test_dataset_has_one_conversation_per_record(records: list[PreparedRecord]) -> None:
    dataset = build_dataset(records, SYSTEM_PROMPT)

    assert len(dataset) == 2
    assert dataset.column_names == ["messages"]


def test_conversation_carries_three_turns(records: list[PreparedRecord]) -> None:
    dataset = build_dataset(records, SYSTEM_PROMPT)

    messages = dataset[0]["messages"]
    assert [turn["role"] for turn in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "Fix the bug."
    assert messages[2]["content"] == '{"prompt": {}}'


def test_record_order_is_preserved(records: list[PreparedRecord]) -> None:
    dataset = build_dataset(list(reversed(records)), SYSTEM_PROMPT)

    assert dataset[0]["messages"][1]["content"] == "Summarise this."


def test_empty_corpus_yields_an_empty_dataset() -> None:
    assert len(build_dataset([], SYSTEM_PROMPT)) == 0


def test_formatting_accepts_a_single_example() -> None:
    tokenizer = StubTokenizer()
    formatter = make_formatting_func(tokenizer)

    rendered = formatter({"messages": [{"role": "user", "content": "hello"}]})

    assert rendered == ["user: hello"]


def test_formatting_accepts_a_batch() -> None:
    tokenizer = StubTokenizer()
    formatter = make_formatting_func(tokenizer)

    rendered = formatter(
        {
            "messages": [
                [{"role": "user", "content": "one"}],
                [{"role": "user", "content": "two"}],
            ]
        }
    )

    assert rendered == ["user: one", "user: two"]


def test_formatting_keeps_the_assistant_turn() -> None:
    tokenizer = StubTokenizer()
    formatter = make_formatting_func(tokenizer)
    dataset = build_dataset(
        [PreparedRecord(index=0, prompt="p", target="t")],
        SYSTEM_PROMPT,
    )

    rendered = formatter(dataset[0])

    assert rendered[0].endswith("assistant: t")
    assert tokenizer.last_call == {"tokenize": False, "add_generation_prompt": False}


@pytest.mark.parametrize("precision", ["auto", "fp16", "bf16"])
def test_reduced_precision_is_disabled_without_a_device(precision: str) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present")

    assert resolve_precision(precision) == (False, False)  # type: ignore[arg-type]
