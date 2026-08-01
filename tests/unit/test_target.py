"""Assembly and serialisation of the target tree."""

from __future__ import annotations

import json

import pytest

from pstparser.data import build_target
from pstparser.pst import (
    LEAF_PATHS,
    ROOT_KEY,
    assemble,
    collect_text,
    serialise_target,
    split_path,
    target_json_schema,
)

COLUMN_MAPPING = {
    "main_instruction": "MAIN INSTRUCTION",
    "context.format": "FORMAT",
    "context.constrains": "CONSTRAINS",
    "context.data": "DATA",
    "context.role": "ROLE",
    "examples": "EXAMPLES",
    "reasoning.influence": "INFLUENCE",
    "reasoning.reasoning_examples": "REASONING EXAMPLES",
    "reasoning.paths": "PATHS",
}


def test_taxonomy_declares_nine_leaves() -> None:
    assert len(LEAF_PATHS) == 9
    assert len(set(LEAF_PATHS)) == 9


def test_split_path_separates_levels() -> None:
    assert split_path("context.format") == ("context", "format")
    assert split_path("examples") == ("examples",)


def test_assembled_tree_has_the_declared_key_order() -> None:
    tree = assemble({path: [] for path in LEAF_PATHS})

    root = tree[ROOT_KEY]
    assert list(root) == ["main_instruction", "context", "examples", "reasoning"]
    assert list(root["context"]) == ["format", "constrains", "data", "role"]
    assert list(root["reasoning"]) == ["influence", "reasoning_examples", "paths"]


def test_assemble_rejects_an_incomplete_mapping() -> None:
    values = {path: [] for path in LEAF_PATHS}
    del values["context.role"]

    with pytest.raises(KeyError):
        assemble(values)


def test_build_target_places_each_column_at_its_leaf() -> None:
    row = {
        "MAIN INSTRUCTION": "Summarise the code.",
        "FORMAT": "Answer as JSON.",
        "CONSTRAINS": None,
        "DATA": "def add(a, b): return a + b",
        "ROLE": "You are a senior engineer.",
        "EXAMPLES": None,
        "INFLUENCE": None,
        "REASONING EXAMPLES": None,
        "PATHS": None,
    }

    tree = build_target(row, COLUMN_MAPPING)[ROOT_KEY]

    assert tree["main_instruction"] == ["Summarise the code."]
    assert tree["context"]["format"] == ["Answer as JSON."]
    assert tree["context"]["constrains"] == []
    assert tree["context"]["role"] == ["You are a senior engineer."]
    assert tree["reasoning"]["paths"] == []


def test_serialised_target_is_stable() -> None:
    row = dict.fromkeys(COLUMN_MAPPING.values())
    row["MAIN INSTRUCTION"] = "Do the thing."

    serialised = serialise_target(build_target(row, COLUMN_MAPPING))

    assert serialised == (
        '{"prompt": {"main_instruction": ["Do the thing."], '
        '"context": {"format": [], "constrains": [], "data": [], "role": []}, '
        '"examples": [], '
        '"reasoning": {"influence": [], "reasoning_examples": [], "paths": []}}}'
    )


def test_serialised_target_round_trips() -> None:
    row = dict.fromkeys(COLUMN_MAPPING.values())
    row["DATA"] = "payload with unicode: caffè ☕"

    restored = json.loads(serialise_target(build_target(row, COLUMN_MAPPING)))

    assert restored[ROOT_KEY]["context"]["data"] == ["payload with unicode: caffè ☕"]


def test_collect_text_concatenates_in_traversal_order() -> None:
    tree = assemble(
        {
            **{path: [] for path in LEAF_PATHS},
            "main_instruction": ["one "],
            "context.data": ["two "],
            "reasoning.influence": ["three"],
        }
    )

    assert collect_text(tree[ROOT_KEY]) == "one two three"


def test_collect_text_ignores_non_text_nodes() -> None:
    assert collect_text({"a": None, "b": [1, "kept", 2.5]}) == "kept"


def test_schema_mirrors_the_taxonomy() -> None:
    schema = target_json_schema()

    root = schema["properties"][ROOT_KEY]
    assert schema["required"] == [ROOT_KEY]
    assert root["additionalProperties"] is False
    assert root["properties"]["context"]["properties"]["constrains"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert sorted(root["properties"]["reasoning"]["required"]) == [
        "influence",
        "paths",
        "reasoning_examples",
    ]
