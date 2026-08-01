"""The Prompt Syntax Tree taxonomy.

This module is the single source of truth for the shape of the target object.
The order in which leaves are declared here determines the key order of every
serialised target, so it must not be changed without regenerating the corpus.
"""

from __future__ import annotations

from typing import Any, Final

#: Key wrapping the tree in the serialised target.
ROOT_KEY: Final = "prompt"

#: Separator between levels in a dotted leaf path.
PATH_SEPARATOR: Final = "."

#: Leaf nodes, in serialisation order. Interior nodes are implied by the paths.
LEAF_PATHS: Final[tuple[str, ...]] = (
    "main_instruction",
    "context.format",
    "context.constrains",
    "context.data",
    "context.role",
    "examples",
    "reasoning.influence",
    "reasoning.reasoning_examples",
    "reasoning.paths",
)


def split_path(path: str) -> tuple[str, ...]:
    """Split a dotted leaf path into its components.

    Args:
        path: Dotted path such as ``context.format``.

    Returns:
        The path components, outermost first.
    """
    return tuple(path.split(PATH_SEPARATOR))


def assemble(values: dict[str, Any]) -> dict[str, Any]:
    """Build the nested tree from a flat mapping of leaf values.

    Leaves are inserted following :data:`LEAF_PATHS`, which fixes the key order
    of the resulting mapping and therefore of its JSON serialisation.

    Args:
        values: Mapping from dotted leaf path to the value stored at that leaf.

    Returns:
        The nested tree, wrapped under :data:`ROOT_KEY`.

    Raises:
        KeyError: If a declared leaf is absent from ``values``.
    """
    tree: dict[str, Any] = {}
    for path in LEAF_PATHS:
        *branches, leaf = split_path(path)
        node = tree
        for branch in branches:
            node = node.setdefault(branch, {})
        node[leaf] = values[path]
    return {ROOT_KEY: tree}


def collect_text(node: Any) -> str:
    """Concatenate every string reachable from a node, in traversal order.

    Args:
        node: A tree, a subtree, a list of segments or a single segment.

    Returns:
        The concatenation of all strings found, without any separator.
    """
    if isinstance(node, dict):
        return "".join(collect_text(value) for value in node.values())
    if isinstance(node, list):
        return "".join(collect_text(item) for item in node)
    if isinstance(node, str):
        return node
    return ""
