"""Reading credentials from the environment file.

Credentials never live in a configuration file: the configuration is versioned
and a credential must not be. They are read from the environment instead, and
``.env`` is where a developer keeps them locally. Containers get the same file
through their own mechanism, so this reader exists for the path that has none.

A variable already present in the environment always wins. Exporting one for a
single command is the ordinary way to override a stored value, and a file that
quietly took precedence would break that expectation.

Parsing is deliberately small: the file holds credentials, not a shell script.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Default name of the file holding local credentials.
ENV_FILE = ".env"

#: Quote characters stripped from around a value, when they match.
_QUOTES = ("'", '"')


def parse_env(text: str) -> dict[str, str]:
    """Read the assignments of an environment file.

    Blank lines and comments are ignored, as is any line without an assignment.
    Whitespace around the name and the value is dropped, and a value wrapped in
    matching quotes is unwrapped, so that all of ``KEY=value``, ``KEY= value``
    and ``KEY="value"`` mean the same thing.

    Args:
        text: Contents of the file.

    Returns:
        The assignments, in file order. A name assigned twice keeps the last.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        name, _, value = stripped.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if not name:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        values[name] = value

    return values


def load_env_file(path: str | Path = ENV_FILE) -> dict[str, str]:
    """Apply the assignments of an environment file to the process.

    Args:
        path: File to read. A missing file is not an error: the variables may
            well be set some other way.

    Returns:
        The names and values that were applied, leaving out those the
        environment already carried and those assigned an empty value.
    """
    source = Path(path)
    if not source.is_file():
        return {}

    applied = {}
    for name, value in parse_env(source.read_text(encoding="utf-8")).items():
        if not value or os.environ.get(name):
            continue
        os.environ[name] = value
        applied[name] = value
    return applied
