"""One deterministic Python extraction policy shared by all prompt conditions."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

EXTRACTION_VERSION = "python-single-block-v1"


@dataclass(frozen=True)
class ExtractedCode:
    """Extracted source and a failure category; no automatic repair is attempted."""

    code: str | None
    status: str


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
