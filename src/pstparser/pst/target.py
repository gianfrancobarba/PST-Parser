"""Serialisation of the target object and the schema it conforms to."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pstparser.pst.taxonomy import LEAF_PATHS, ROOT_KEY, split_path


def serialise_target(target: Mapping[str, Any]) -> str:
    """Serialise a target tree to the string the model is trained to produce.

    Args:
        target: The nested target tree.

    Returns:
        Its JSON representation.
    """
    return json.dumps(target)


def target_json_schema() -> dict[str, Any]:
    """Derive a JSON Schema describing a well-formed target.

    Every leaf holds a list of text segments. The schema is generated from the
    taxonomy, so it cannot drift from the objects the pipeline produces.

    Returns:
        A JSON Schema document.
    """
    leaf_schema = {"type": "array", "items": {"type": "string"}}
    tree: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for path in LEAF_PATHS:
        *branches, leaf = split_path(path)
        node = tree
        for branch in branches:
            properties = node["properties"]
            if branch not in properties:
                properties[branch] = {"type": "object", "properties": {}, "required": []}
                node["required"].append(branch)
            node = properties[branch]
        node["properties"][leaf] = dict(leaf_schema)
        node["required"].append(leaf)

    _seal(tree)
    return {
        "type": "object",
        "properties": {ROOT_KEY: tree},
        "required": [ROOT_KEY],
        "additionalProperties": False,
    }


def conforms(payload: Any, schema: Mapping[str, Any] | None = None) -> bool:
    """Report whether a parsed object satisfies the target schema.

    Only the constructs :func:`target_json_schema` emits are interpreted, which
    is what keeps the check free of a validation dependency. The schema defaults
    to the one derived from the taxonomy, so the two cannot drift apart.

    Args:
        payload: The object to check, already parsed.
        schema: Schema to check it against. Defaults to the target schema.

    Returns:
        ``True`` when the object satisfies the schema.
    """
    return _conforms(payload, target_json_schema() if schema is None else schema)


def _conforms(payload: Any, schema: Mapping[str, Any]) -> bool:
    """Check one node against one schema fragment, depth first."""
    declared = schema.get("type")

    if declared == "object":
        if not isinstance(payload, dict):
            return False
        properties: Mapping[str, Any] = schema.get("properties", {})
        if any(key not in payload for key in schema.get("required", ())):
            return False
        sealed = schema.get("additionalProperties") is False
        if sealed and any(key not in properties for key in payload):
            return False
        return all(
            _conforms(value, properties[key]) for key, value in payload.items() if key in properties
        )

    if declared == "array":
        if not isinstance(payload, list):
            return False
        items = schema.get("items")
        return items is None or all(_conforms(item, items) for item in payload)

    if declared == "string":
        return isinstance(payload, str)

    return True


def _seal(node: dict[str, Any]) -> None:
    """Forbid undeclared keys on every object of a schema, depth first."""
    node["additionalProperties"] = False
    for child in node["properties"].values():
        if child.get("type") == "object":
            _seal(child)
