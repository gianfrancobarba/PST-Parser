"""Decoding is explicit: nothing is inherited from a model's generation_config."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel
from typer.testing import CliRunner

from pstparser.config import InferenceConfig
from pstparser.inference.generate import decoding_parameters, generate_messages

# Sampling values of the security arm, and of Qwen's published generation_config.
SAMPLED = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


class Tokenizer:
    model_max_length = 64
    eos_token_id = 2

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor([[5, 6, 7]])

    def decode(self, values: Any, **kwargs: Any) -> str:
        return "answer"


class RecordingModel:
    """Fake model whose own generation_config carries Qwen's published defaults."""

    config = SimpleNamespace(max_position_embeddings=64)

    def __init__(self) -> None:
        """Publish a repetition penalty of 1.1 and sampling defaults, as Qwen does."""
        self.generation_config = SimpleNamespace(
            eos_token_id=[2], repetition_penalty=1.1, do_sample=True, temperature=0.7, top_k=20
        )
        self.received: dict[str, Any] = {}

    def parameters(self) -> Any:
        return iter(())

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.received = kwargs
        return torch.tensor([[5, 6, 7, 1, 2]])


def tiny_model(**generation_defaults: Any) -> GPT2LMHeadModel:
    """A randomly initialised model: no download, real ``GenerationMixin``."""
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_positions=64, n_embd=16, n_layer=1, n_head=2)
    )
    model.eval()
    for name, value in generation_defaults.items():
        setattr(model.generation_config, name, value)
    return model


def effective_configs(model: GPT2LMHeadModel) -> list[GenerationConfig]:
    """Record the resolved configuration transformers builds its processors from."""
    seen: list[GenerationConfig] = []
    original = model._get_logits_processor

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["generation_config"] if "generation_config" in kwargs else args[0])
        return original(*args, **kwargs)

    model._get_logits_processor = spy  # type: ignore[method-assign]
    return seen


QWEN_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.1,
    # A version at or above 4.50 makes transformers fall back to the model's
    # values for every field left at the library default.
    "transformers_version": "4.57.3",
}


def test_defaults_are_greedy_with_neutral_repetition_penalty() -> None:
    config = InferenceConfig()
    assert config.do_sample is False
    assert config.repetition_penalty == 1.0
    assert config.top_k is None


def test_fake_model_with_penalty_1_1_receives_1_0() -> None:
    model = RecordingModel()
    result = generate_messages(model, Tokenizer(), [], InferenceConfig(max_new_tokens=8))
    passed = model.received["generation_config"]
    assert isinstance(passed, GenerationConfig)
    assert passed.repetition_penalty == 1.0
    assert passed.do_sample is False
    assert passed.num_beams == 1
    assert passed.max_new_tokens == 8
    assert model.received["use_model_defaults"] is False
    # Every decoding value travels inside the explicit configuration.
    assert set(model.received) == {"generation_config", "use_model_defaults"}
    assert result.generation_config == {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8,
        "use_cache": True,
    }


def test_sampling_parameters_are_all_passed_explicitly() -> None:
    model = RecordingModel()
    config = InferenceConfig(max_new_tokens=8, **SAMPLED)
    result = generate_messages(model, Tokenizer(), [], config)
    passed = model.received["generation_config"]
    assert (passed.do_sample, passed.temperature, passed.top_p, passed.top_k) == (
        True,
        0.7,
        0.8,
        20,
    )
    assert passed.repetition_penalty == 1.0
    assert result.generation_config == {
        "do_sample": True,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8,
        "use_cache": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }


def test_sampling_without_top_k_disables_the_library_default_of_50() -> None:
    values = decoding_parameters(InferenceConfig(do_sample=True, temperature=0.7, top_p=0.8))
    assert values["top_k"] == 0
    assert GenerationConfig().top_k == 50


def test_real_generation_resolves_penalty_1_0_despite_model_defaults() -> None:
    model = tiny_model(**QWEN_DEFAULTS)
    seen = effective_configs(model)
    generate_messages(model, Tokenizer(), [], InferenceConfig(max_new_tokens=6))
    assert len(seen) == 1
    assert seen[0].repetition_penalty == 1.0
    assert seen[0].do_sample is False
    assert seen[0].num_beams == 1


def test_the_inheritance_hazard_is_real_without_the_explicit_switch() -> None:
    """Passing a GenerationConfig alone is not enough in this transformers version."""
    model = tiny_model(**QWEN_DEFAULTS)
    seen = effective_configs(model)
    explicit = GenerationConfig(do_sample=False, repetition_penalty=1.0, max_new_tokens=6)
    model.generate(input_ids=torch.tensor([[5, 6, 7]]), generation_config=explicit)
    assert seen[0].repetition_penalty == 1.1


def test_greedy_output_is_independent_of_the_model_generation_defaults() -> None:
    config = InferenceConfig(max_new_tokens=12)
    captured: list[list[int]] = []

    class Capture(Tokenizer):
        def decode(self, values: Any, **kwargs: Any) -> str:
            captured.append([int(v) for v in values])
            return "answer"

    generate_messages(tiny_model(), Capture(), [], config)
    generate_messages(tiny_model(**QWEN_DEFAULTS), Capture(), [], config)
    generate_messages(
        tiny_model(repetition_penalty=5.0, no_repeat_ngram_size=1), Capture(), [], config
    )
    assert len(captured[0]) == 12
    assert captured[0] == captured[1] == captured[2]


def test_real_sampling_resolves_the_configured_values() -> None:
    model = tiny_model(temperature=0.2, top_k=3, top_p=0.5, repetition_penalty=1.3)
    seen = effective_configs(model)
    generate_messages(model, Tokenizer(), [], InferenceConfig(max_new_tokens=4, **SAMPLED))
    resolved = seen[0]
    assert (resolved.temperature, resolved.top_p, resolved.top_k) == (0.7, 0.8, 20)
    assert resolved.repetition_penalty == 1.0
    assert resolved.do_sample is True


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"do_sample": True, "temperature": 0.0}, "temperature greater than 0"),
        ({"do_sample": True}, "temperature greater than 0"),
        ({"do_sample": False, "temperature": 0.7}, "greedy decoding ignores"),
        ({"do_sample": False, "top_p": 0.8}, "greedy decoding ignores"),
        ({"do_sample": False, "top_k": 20}, "greedy decoding ignores"),
        ({"repetition_penalty": 0.0}, "greater than 0"),
        ({"do_sample": True, "temperature": 0.7, "top_k": 0}, "greater than 0"),
    ],
)
def test_inconsistent_decoding_is_rejected_at_validation(values, message) -> None:
    with pytest.raises(ValidationError, match=message):
        InferenceConfig(**values)


def test_generate_command_rejects_sampling_at_temperature_zero_before_loading(
    tmp_path, monkeypatch
) -> None:
    import pstparser.models.loader as loader
    from pstparser.empirical.cli import app

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(loader, "load_untuned", forbidden)
    (tmp_path / "review.md").write_text("reviewed\n", encoding="utf-8")
    configuration = tmp_path / "generate.json"
    configuration.write_text(
        json.dumps(
            {
                "model": {"backend": "hf", "name": "fake/model", "revision": "a" * 40},
                "inference": {"max_new_tokens": 8, "do_sample": True, "temperature": 0.0},
                "seed": 1,
                "replicate_ids": [1, 2],
                "system_prompt": "system",
                "protocol_version": "test",
                "dataset_revision": "test",
                "purpose": "pilot",
                "review_record": "review.md",
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app, ["generate", str(tmp_path / "missing.jsonl"), str(configuration), str(tmp_path / "o")]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValidationError)
    assert "temperature greater than 0" in str(result.exception)
    assert not (tmp_path / "o").exists()
