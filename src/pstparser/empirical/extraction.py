"""One deterministic Python extraction policy shared by all prompt conditions."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

EXTRACTION_VERSION = "python-single-block-v1"


#: Primary policy for LLMSecEval: every complete Python fenced block, in order.
FENCED_BLOCKS: Final = "python-fenced-blocks-v1"
#: Sensitivity for LLMSecEval: the longest complete Python fenced block alone.
LONGEST_BLOCK: Final = "python-longest-block-v1"
#: Primary policy for BigCodeBench: the official ``sanitize`` of the harness.
#: It runs inside the pinned harness image and is never reimplemented here.
BIGCODEBENCH_SANITIZE: Final = "bigcodebench-sanitize"

_PYTHON_LABELS: Final = frozenset({"", "python", "py"})
_BLOCK_SEPARATOR: Final = "\n\n"


@dataclass(frozen=True)
class ExtractedCode:
    """Extracted source and a failure category; no automatic repair is attempted.

    ``policy`` names the frozen rule that produced the result. ``blocks`` is the
    number of fenced blocks the code was taken from, when the policy counts them.
    """

    code: str | None
    status: str
    policy: str = EXTRACTION_VERSION
    blocks: int | None = None


def extract_python(raw: str) -> ExtractedCode:
    """Accept plain Python or one complete Python/unlabelled fenced block.

    Text outside a fenced block is rejected, as are multiple blocks, incomplete
    fences and syntax errors. Blank responses never count as executable code.
    The original raw output is the caller's responsibility to retain.
    """
    text = raw.strip()
    if not text:
        return ExtractedCode(None, "empty")
    if "```" in text:
        match = re.fullmatch(r"```(?:python)?[ \t]*\r?\n(.*?)\r?\n```", text, re.DOTALL)
        if match is None or "```" in match[1]:
            return ExtractedCode(None, "invalid_fence")
        text = match[1]
    if not text.strip():
        return ExtractedCode(None, "empty")
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return ExtractedCode(text, "syntax_error")
    if not tree.body:
        return ExtractedCode(text, "empty")
    return ExtractedCode(text, "ok")


def _fenced_blocks(raw: str) -> list[tuple[str, str]] | None:
    """Return (label, content) of every fenced block, or ``None`` if a fence stays open.

    A fence is a line that starts with three backticks; the opening line may
    carry a language label. Backticks inside a line are ordinary text.
    """
    blocks: list[tuple[str, str]] = []
    label: str | None = None
    content: list[str] = []
    for line in raw.split("\n"):
        marker = line.strip()
        if label is None:
            if marker.startswith("```"):
                label = marker[3:].strip().lower()
        elif marker == "```":
            blocks.append((label, "\n".join(content)))
            label, content = None, []
        else:
            content.append(line)
    return None if label is not None else blocks


def _parsed(code: str, policy: str, blocks: int) -> ExtractedCode:
    """Classify extracted text: it is kept as it is, never repaired."""
    if not code.strip():
        return ExtractedCode(None, "empty", policy, blocks)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ExtractedCode(code, "syntax_error", policy, blocks)
    if not tree.body:
        return ExtractedCode(code, "no_code", policy, blocks)
    return ExtractedCode(code, "ok", policy, blocks)


def _from_blocks(raw: str, policy: str, *, longest_only: bool) -> ExtractedCode:
    if not raw.strip():
        return ExtractedCode(None, "empty", policy, 0)
    fenced = _fenced_blocks(raw)
    if fenced is None:
        return ExtractedCode(None, "invalid_fence", policy)
    if not fenced:
        # No fence at all: the answer may be plain code.
        return _parsed(raw, policy, 0)
    python = [content for label, content in fenced if label in _PYTHON_LABELS]
    if not python:
        return ExtractedCode(None, "no_code", policy, 0)
    if longest_only:
        # The first of the longest blocks, so that a tie has one answer.
        return _parsed(max(python, key=len), policy, 1)
    return _parsed(_BLOCK_SEPARATOR.join(python), policy, len(python))


def extract_fenced_blocks(raw: str) -> ExtractedCode:
    """Join every complete Python or unlabelled fenced block, in order.

    Policy ``python-fenced-blocks-v1``. Prose outside the blocks is allowed and
    dropped; blocks are separated by one empty line; blocks labelled with
    another language are ignored. Without any fence the whole answer is taken
    as code. Statuses: ``ok``, ``empty``, ``invalid_fence`` (a fence is never
    closed), ``syntax_error``, ``no_code`` (only blocks of other languages, or
    nothing but comments).
    """
    return _from_blocks(raw, FENCED_BLOCKS, longest_only=False)


def extract_longest_block(raw: str) -> ExtractedCode:
    """Take only the longest complete Python or unlabelled fenced block.

    Policy ``python-longest-block-v1``, the sensitivity analysis of the joined
    blocks: a usage example in a second block cannot add findings.
    """
    return _from_blocks(raw, LONGEST_BLOCK, longest_only=True)


#: Policies that run here. The sanitize policy runs in the harness image.
LOCAL_POLICIES: Final[dict[str, Callable[[str], ExtractedCode]]] = {
    EXTRACTION_VERSION: extract_python,
    FENCED_BLOCKS: extract_fenced_blocks,
    LONGEST_BLOCK: extract_longest_block,
}
POLICIES: Final = (*LOCAL_POLICIES, BIGCODEBENCH_SANITIZE)


def extract(raw: str, policy: str) -> ExtractedCode:
    """Apply one frozen extraction policy, chosen by name.

    Raises:
        ValueError: For an unknown policy, and for the sanitize policy, which
            only the BigCodeBench adapter can apply, inside the harness image.
    """
    if policy == BIGCODEBENCH_SANITIZE:
        raise ValueError(
            f"{policy} runs inside the harness image, through the BigCodeBench adapter"
        )
    if policy not in LOCAL_POLICIES:
        raise ValueError(f"unknown extraction policy {policy!r}; choose one of {POLICIES}")
    return LOCAL_POLICIES[policy](raw)


def extraction_statuses(raw: str) -> dict[str, str]:
    """Status of one raw output under every local policy, for the diagnostic tables."""
    return {policy: function(raw).status for policy, function in LOCAL_POLICIES.items()}
