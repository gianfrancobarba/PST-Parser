"""The error raised when a corpus cannot be read.

It lives on its own because a corpus is no longer necessarily a spreadsheet:
every source format raises this, and none of them should have to import from
another format's module to do so.
"""

from __future__ import annotations


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be read or does not have the expected layout."""
