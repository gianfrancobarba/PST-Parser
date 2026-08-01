"""Decoding constrained to the target schema.

The model used here is untrained, so it emits noise. What matters is that the
noise is still shaped like a valid target: the constraint is what guarantees it,
not the weights.
"""

from __future__ import annotations

import json

import pytest

from pstparser.config import InferenceConfig
from pstparser.inference.generate import build_schema_constraint, generate_one

pytestmark = pytest.mark.slow

pytest.importorskip("outlines", reason="the 'structured' extra is not installed")

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load the tiny model once for the module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture(scope="module")
def constraint(model_and_tokenizer):
    """Compile the schema constraint once, as the pipeline does."""
    model, tokenizer = model_and_tokenizer
    return build_schema_constraint(model, tokenizer)


def test_constraint_is_a_processor_list(constraint) -> None:
    assert len(constraint) == 1


def test_constrained_output_follows_the_schema(model_and_tokenizer, constraint) -> None:
    model, tokenizer = model_and_tokenizer
    config = InferenceConfig(max_new_tokens=64, do_sample=False, structured_output=True)

    output = generate_one(
        model=model,
        tokenizer=tokenizer,
        system_prompt="Segment the prompt.",
        prompt="Summarise the report.",
        config=config,
        logits_processors=constraint,
    )

    # The budget is too small for the model to close the object, so only the
    # prefix can be checked. It must already follow the declared key order.
    assert output.startswith('{"prompt"')
    assert '"main_instruction"' in output


def test_unconstrained_output_is_unrestricted(model_and_tokenizer) -> None:
    model, tokenizer = model_and_tokenizer
    config = InferenceConfig(max_new_tokens=16, do_sample=False)

    output = generate_one(
        model=model,
        tokenizer=tokenizer,
        system_prompt="Segment the prompt.",
        prompt="Summarise the report.",
        config=config,
    )

    # An untrained model has no reason to emit JSON when nothing forces it to.
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)
