"""Decoding constrained to the target schema.

The model used here is untrained, so it emits noise. What matters is that the
noise is still shaped like a valid target: the constraint is what guarantees it,
not the weights.
"""

from __future__ import annotations

import json

import pytest

from pstparser.config import InferenceConfig
from pstparser.data import PreparedRecord
from pstparser.inference.generate import (
    build_schema_constraint,
    generate_one,
    generate_predictions,
)

pytestmark = pytest.mark.slow

pytest.importorskip("outlines", reason="the 'structured' extra is not installed")

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

#: How a target begins, before the whitespace the grammar also permits.
OPENING = '{"prompt":{"main_instruction":['


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
    # Where exactly the text is cut depends on how many tokens the budget buys,
    # so the check stops short of the closing quotation mark.
    assert output.startswith('{"prompt"')
    assert "main_instruction" in output


def test_every_prediction_is_constrained(model_and_tokenizer) -> None:
    """The constraint must restart for each record, not only for the first.

    The processor walks the schema as the model writes, so it ends a prediction
    in the state where the object has been closed. Left there, it permits
    nothing but the end of the text, and every record after the first comes back
    empty. One prediction cannot show this, which is why the check needs two and
    why it goes through the loop rather than through a single call.
    """
    model, tokenizer = model_and_tokenizer
    records = [
        PreparedRecord(index=0, prompt="Summarise the report.", target="{}"),
        PreparedRecord(index=1, prompt="Rename the variable.", target="{}"),
    ]
    config = InferenceConfig(max_new_tokens=64, do_sample=False, structured_output=True)

    predictions = list(
        generate_predictions(
            model=model,
            tokenizer=tokenizer,
            records=records,
            system_prompt="Segment the prompt.",
            config=config,
        )
    )

    assert len(predictions) == 2
    for prediction in predictions:
        written = "".join(prediction.prediction.split())
        # Each one wrote something, and what it wrote is how a target begins.
        # The budget decides how far it got and the grammar leaves whitespace
        # free, so the comparison ignores spacing and accepts any prefix.
        assert written
        assert OPENING.startswith(written) or written.startswith(OPENING)


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
