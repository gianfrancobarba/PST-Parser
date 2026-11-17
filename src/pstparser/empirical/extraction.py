"""One deterministic Python extraction policy shared by all prompt conditions."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

EXTRACTION_VERSION = "python-single-block-v1"


#: Primary policy for LLMSecEval: every complete Python fenced block, in order.
#: v2 (R-ORA-08): ``python3`` is a Python label, and a last fence left open no
#: longer discards the complete blocks before it.
FENCED_BLOCKS: Final = "python-fenced-blocks-v2"
#: Sensitivity for LLMSecEval: the longest complete Python fenced block alone.
LONGEST_BLOCK: Final = "python-longest-block-v2"
#: Primary policy for BigCodeBench: the official ``sanitize`` of the harness.
#: It runs inside the pinned harness image and is never reimplemented here.
BIGCODEBENCH_SANITIZE: Final = "bigcodebench-sanitize"

_PYTHON_LABELS: Final = frozenset({"", "python", "py", "python3"})
_BLOCK_SEPARATOR: Final = "\n\n"


@dataclass(frozen=True)
class ExtractedCode:
    """Extracted source and a failure category; no automatic repair is attempted.

    ``policy`` names the frozen rule that produced the result. ``blocks`` is the
    number of fenced blocks the code was taken from, when the policy counts them.
    ``open_fence`` flags a last fence that was never closed and was left out.
    """

    code: str | None
    status: str
    policy: str = EXTRACTION_VERSION
    blocks: int | None = None
    open_fence: bool = False


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


def _fenced_blocks(raw: str) -> tuple[list[tuple[str, str]], bool]:
    """Return (label, content) of every complete fenced block, and whether a fence stays open.

    A fence is a line that starts with three backticks; the opening line may
    carry a language label. Backticks inside a line are ordinary text. A fence
    left open at the end (a cut answer) is not a block.
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
    return blocks, label is not None


def _parsed(code: str, policy: str, blocks: int, open_fence: bool = False) -> ExtractedCode:
    """Classify extracted text: it is kept as it is, never repaired."""
    if not code.strip():
        return ExtractedCode(None, "empty", policy, blocks, open_fence)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ExtractedCode(code, "syntax_error", policy, blocks, open_fence)
    if not tree.body:
        return ExtractedCode(code, "no_code", policy, blocks, open_fence)
    return ExtractedCode(code, "ok", policy, blocks, open_fence)


def _from_blocks(raw: str, policy: str, *, longest_only: bool) -> ExtractedCode:
    if not raw.strip():
        return ExtractedCode(None, "empty", policy, 0)
    fenced, open_fence = _fenced_blocks(raw)
    if not fenced:
        if open_fence:
            return ExtractedCode(None, "invalid_fence", policy, None, True)
        # No fence at all: the answer may be plain code.
        return _parsed(raw, policy, 0)
    python = [content for label, content in fenced if label in _PYTHON_LABELS]
    if not python:
        return ExtractedCode(None, "no_code", policy, 0, open_fence)
    if longest_only:
        # The first of the longest blocks, so that a tie has one answer.
        return _parsed(max(python, key=len), policy, 1, open_fence)
    return _parsed(_BLOCK_SEPARATOR.join(python), policy, len(python), open_fence)


def extract_fenced_blocks(raw: str) -> ExtractedCode:
    """Join every complete Python or unlabelled fenced block, in order.

    Policy ``python-fenced-blocks-v2``. Prose outside the blocks is allowed and
    dropped; blocks are separated by one empty line; blocks labelled with
    another language are ignored (``python``, ``py``, ``python3`` and no label
    are Python). Without any fence the whole answer is taken as code. A last
    fence left open is dropped and flagged (``open_fence``); the complete
    blocks before it are kept. Statuses: ``ok``, ``empty``, ``invalid_fence``
    (the only fence is never closed), ``syntax_error``, ``no_code`` (only
    blocks of other languages, or nothing but comments).
    """
    return _from_blocks(raw, FENCED_BLOCKS, longest_only=False)


def extract_longest_block(raw: str) -> ExtractedCode:
    """Take only the longest complete Python or unlabelled fenced block.

    Policy ``python-longest-block-v2``, the sensitivity analysis of the joined
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
