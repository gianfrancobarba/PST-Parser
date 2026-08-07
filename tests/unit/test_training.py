"""Dataset construction, without loading a model."""

from __future__ import annotations

import pytest

from pstparser.conversation import build_prompt
from pstparser.data import PreparedRecord
from pstparser.models import resolve_precision, trainable_parameters
from pstparser.training import build_dataset

SYSTEM_PROMPT = "Segment the prompt."


@pytest.fixture
def records() -> list[PreparedRecord]:
    return [
        PreparedRecord(index=0, prompt="Fix the bug.", target='{"prompt": {}}'),
        PreparedRecord(index=1, prompt="Summarise this.", target='{"prompt": {"a": []}}'),
    ]


def test_dataset_has_one_record_per_prompt(records: list[PreparedRecord]) -> None:
    dataset = build_dataset(records, SYSTEM_PROMPT)

    assert len(dataset) == 2
    # The question and the answer are separate columns: it is that separation
    # the trainer needs to compute the loss on the answer alone.
    assert sorted(dataset.column_names) == ["completion", "prompt"]


def test_the_prompt_carries_the_conditioning_turns(records: list[PreparedRecord]) -> None:
    dataset = build_dataset(records, SYSTEM_PROMPT)

    turns = dataset[0]["prompt"]
    assert [turn["role"] for turn in turns] == ["system", "user"]
    assert turns[0]["content"] == SYSTEM_PROMPT


def test_the_completion_is_the_target_alone(records: list[PreparedRecord]) -> None:
    dataset = build_dataset(records, SYSTEM_PROMPT)

    assert dataset[0]["completion"] == [{"role": "assistant", "content": '{"prompt": {}}'}]


def test_the_prompt_arrives_inside_the_delimiters(records: list[PreparedRecord]) -> None:
    # The system message tells the model to read only what is between them, so
    # without them the instruction points at markers that never arrive.
    dataset = build_dataset(records, SYSTEM_PROMPT)

    assert dataset[0]["prompt"][1]["content"] == "<input_prompt>Fix the bug.</input_prompt>"


def test_training_and_generation_ask_the_same_thing(records: list[PreparedRecord]) -> None:
    # The two used to build the turns separately and happened to agree. Nothing
    # made them agree, and a gap between them is invisible in every artefact.
    dataset = build_dataset(records, SYSTEM_PROMPT)

    assert dataset[0]["prompt"] == build_prompt(SYSTEM_PROMPT, records[0].prompt)


def test_record_order_is_preserved(records: list[PreparedRecord]) -> None:
    dataset = build_dataset(list(reversed(records)), SYSTEM_PROMPT)

    assert dataset[0]["prompt"][1]["content"] == "<input_prompt>Summarise this.</input_prompt>"


def test_empty_corpus_yields_an_empty_dataset() -> None:
    assert len(build_dataset([], SYSTEM_PROMPT)) == 0


@pytest.mark.parametrize("precision", ["auto", "fp16", "bf16"])
def test_reduced_precision_is_disabled_without_a_device(precision: str) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present")

    assert resolve_precision(precision) == (False, False)  # type: ignore[arg-type]


class Params4bit:
    """Stands in for a weight stored several values to a byte.

    The name matters: quantised weights are recognised by their class name.
    """

    def __init__(self, elements: int, itemsize: int) -> None:
        """Store the packed element count and the width of one storage unit."""
        self._elements = elements
        self.quant_storage = type("Storage", (), {"itemsize": itemsize})()
        self.requires_grad = False

    def numel(self) -> int:
        return self._elements

    def element_size(self) -> int:
        return 1


class PlainParameter:
    """Stands in for an ordinary trainable weight."""

    def __init__(self, elements: int, requires_grad: bool) -> None:
        """Store the element count and whether the weight is trained."""
        self._elements = elements
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._elements

    def element_size(self) -> int:
        return 2


class FakeModel:
    """Exposes a fixed list of parameters."""

    def __init__(self, parameters: list[object]) -> None:
        """Store the parameters the model reports."""
        self._parameters = parameters

    def parameters(self) -> list[object]:
        return self._parameters


def test_plain_parameters_are_counted_as_they_are() -> None:
    model = FakeModel([PlainParameter(100, True), PlainParameter(900, False)])

    assert trainable_parameters(model) == (100, 1000)


def test_packed_weights_count_the_values_they_represent() -> None:
    # One byte of storage holds two quantised values.
    model = FakeModel([Params4bit(elements=500, itemsize=1)])

    assert trainable_parameters(model) == (0, 1000)


def test_wider_storage_packs_proportionally_more() -> None:
    model = FakeModel([Params4bit(elements=500, itemsize=4)])

    assert trainable_parameters(model) == (0, 4000)
