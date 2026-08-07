"""The conversation the model is trained on and asked at inference.

The turns were built twice, once for training and once for generation, and the
two had to be kept in agreement by hand. They happened to agree, but nothing
made them: a change to one of them would have opened a gap between what the
model was taught and what it is asked, which is invisible in every artefact the
pipeline produces. They are built here instead, once.

The user turn is delivered inside delimiters because the system message tells
the model to read only what is inside them. Without them the instruction points
at markers that never arrive, and the model has to learn to disregard a sentence
it was given.
"""

from __future__ import annotations

from typing import Final

#: Delimiters the system message names. Changing them without changing the
#: system message would leave the instruction pointing at nothing again.
PROMPT_OPEN: Final = "<input_prompt>"
PROMPT_CLOSE: Final = "</input_prompt>"

Message = dict[str, str]


def wrap_prompt(prompt: str) -> str:
    """Enclose a prompt in the delimiters the system message refers to.

    Args:
        prompt: The raw, unsegmented prompt.

    Returns:
        The prompt between its delimiters.
    """
    return f"{PROMPT_OPEN}{prompt}{PROMPT_CLOSE}"


def build_prompt(system_prompt: str, prompt: str) -> list[Message]:
    """Build the turns the model is conditioned on.

    These are the turns present both while training and while generating: the
    shared instruction, and the prompt to segment.

    Args:
        system_prompt: Instruction prepended to every conversation.
        prompt: The raw, unsegmented prompt.

    Returns:
        The system and user turns, in order.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": wrap_prompt(prompt)},
    ]


def build_completion(target: str) -> list[Message]:
    """Build the turn the model is asked to produce.

    Args:
        target: The serialised target tree.

    Returns:
        The assistant turn, alone.
    """
    return [{"role": "assistant", "content": target}]
