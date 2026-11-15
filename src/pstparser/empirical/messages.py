"""Messages sent to the generator: one system text and one wrapped user text.

The wrapper is the same for every condition and lives outside the prompt
variant, whose hash covers the body alone. It is filled by concatenation, never
with ``str.format``: a hierarchical body is JSON and is full of braces.
"""

from __future__ import annotations

PLACEHOLDER = "{prompt}"


def validate_user_template(template: str) -> None:
    """Require exactly one placeholder, so that the body is inserted once.

    Raises:
        ValueError: If the placeholder is missing or repeated.
    """
    count = template.count(PLACEHOLDER)
    if count != 1:
        raise ValueError(f"user_template must contain {PLACEHOLDER} exactly once, found {count}")


def fill_user_template(template: str | None, body: str) -> str:
    """Return the user message: the body alone, or the body inside the wrapper.

    The body is never interpreted: braces, placeholders and format fields in it
    reach the model unchanged.
    """
    if template is None:
        return body
    validate_user_template(template)
    prefix, suffix = template.split(PLACEHOLDER)
    return prefix + body + suffix


def chat_messages(system: str, template: str | None, body: str) -> list[dict[str, str]]:
    """Build the two-message conversation of one request."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": fill_user_template(template, body)},
    ]
