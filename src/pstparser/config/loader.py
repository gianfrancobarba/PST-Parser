"""Loading and composition of YAML experiment configurations.

A configuration file may inherit from one or more base files through an
``extends`` key. Bases are merged in declaration order and the inheriting file
is applied last, so the most specific value always wins. Command line overrides
are applied after composition and before validation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from pstparser.config.schema import ExperimentConfig

EXTENDS_KEY = "extends"


class ConfigError(RuntimeError):
    """Raised when a configuration cannot be composed or parsed."""


def load_experiment(
    path: str | Path,
    overrides: list[str] | None = None,
    root: str | Path | None = None,
) -> ExperimentConfig:
    """Compose, override and validate an experiment configuration.

    Args:
        path: Path to the experiment YAML file.
        overrides: Assignments in ``dotted.key=value`` form applied on top of
            the composed mapping. Values are parsed as YAML scalars.
        root: Directory against which ``extends`` entries are resolved. Defaults
            to the top-level ``configs`` directory inferred from ``path``.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If a file is missing, malformed, or if an override does not
            follow the expected syntax.
    """
    path = Path(path)
    root = Path(root) if root is not None else _infer_root(path)

    merged = _compose(path, root, seen=set())
    for assignment in overrides or []:
        _apply_override(merged, assignment)

    return ExperimentConfig.model_validate(merged)


def dump_resolved(config: ExperimentConfig, destination: str | Path) -> Path:
    """Write the fully resolved configuration next to a run's artefacts.

    Args:
        config: The validated configuration.
        destination: File to write. Parent directories are created.

    Returns:
        The path that was written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def _infer_root(path: Path) -> Path:
    """Locate the configuration root by walking up to a directory named ``configs``."""
    for parent in path.resolve().parents:
        if parent.name == "configs":
            return parent
    return path.resolve().parent


def _compose(path: Path, root: Path, seen: set[Path]) -> dict[str, Any]:
    """Recursively merge a configuration file with the bases it extends."""
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in [*seen, resolved])
        raise ConfigError(f"circular extends chain: {chain}")

    document = _read_yaml(resolved)
    bases = document.pop(EXTENDS_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigError(f"'{EXTENDS_KEY}' must be a string or a list in {resolved}")

    merged: dict[str, Any] = {}
    for base in bases:
        base_path = _resolve_base(str(base), root, relative_to=resolved.parent)
        merged = _deep_merge(merged, _compose(base_path, root, seen | {resolved}))

    return _deep_merge(merged, document)


def _resolve_base(reference: str, root: Path, relative_to: Path) -> Path:
    """Resolve an ``extends`` entry against the config root, then the current directory."""
    candidates = [root / reference, relative_to / reference]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise ConfigError(f"base configuration not found: {reference} (tried {tried})")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a mapping."""
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"configuration root must be a mapping in {path}")
    return content


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge two mappings recursively, with ``update`` taking precedence."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_override(target: dict[str, Any], assignment: str) -> None:
    """Apply a single ``dotted.key=value`` assignment in place."""
    key, separator, raw_value = assignment.partition("=")
    if not separator or not key.strip():
        raise ConfigError(f"override must follow 'dotted.key=value', got: {assignment!r}")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid value in override {assignment!r}: {exc}") from exc

    cursor: dict[str, Any] = target
    parts = [part.strip() for part in key.strip().split(".")]
    for part in parts[:-1]:
        node = cursor.get(part)
        if not isinstance(node, dict):
            node = {}
            cursor[part] = node
        cursor = node
    cursor[parts[-1]] = value
