from types import SimpleNamespace

import pytest
import torch

from pstparser.config import InferenceConfig
from pstparser.inference.generate import generate_messages


class Tokenizer:
    model_max_length = 20
    eos_token_id = 2

    def apply_chat_template(self, *args, **kwargs):
        return torch.tensor([[5, 6, 7]])

    def decode(self, values, **kwargs):
        return "answer"


class Model:
    config = SimpleNamespace(max_position_embeddings=20)
    generation_config = SimpleNamespace(eos_token_id=[2, 3])

    def __init__(self, output):
        """Set a deterministic completion for budget and termination tests."""
        self.output = output
        self.called = False

    def parameters(self):
        return iter(())

    def generate(self, input_ids, **kwargs):
        self.called = True
        return torch.tensor([[5, 6, 7, *self.output]])


def test_budget_failure_precedes_generation() -> None:
    model = Model([1])
    with pytest.raises(ValueError, match="no truncation"):
        generate_messages(model, Tokenizer(), [], InferenceConfig(max_new_tokens=18))
    assert not model.called


@pytest.mark.parametrize("tokens,reason", [([1, 1], "length"), ([1, 2], "eos"), ([1], "stopped")])
def test_observed_termination(tokens, reason) -> None:
    result = generate_messages(Model(tokens), Tokenizer(), [], InferenceConfig(max_new_tokens=2))
    assert result.finish_reason == reason
    assert result.input_tokens == 3 and result.output_tokens == len(tokens)


def test_experiment_limit_is_respected() -> None:
    with pytest.raises(ValueError, match="context 4"):
        generate_messages(
            Model([2]), Tokenizer(), [], InferenceConfig(max_new_tokens=2), max_seq_length=4
        )
