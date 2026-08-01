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


def _seal(node: dict[str, Any]) -> None:
    """Forbid undeclared keys on every object of a schema, depth first."""
    node["additionalProperties"] = False
    for child in node["properties"].values():
        if child.get("type") == "object":
            _seal(child)
