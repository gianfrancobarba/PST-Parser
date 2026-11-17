"""Frozen prompt covariates (M13) and judgement units of the gold (M25)."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical import cli
from pstparser.empirical.annotations import propose
from pstparser.empirical.benchmarks import SourceRecord
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.covariates import (
    covariate_table,
    has_code,
    judgement_mask,
    judgement_units,
    prompt_covariates,
    structure,
)
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants

# The prompts of test_gold_annotations.py, whose draft units that file pins down.
CODE = "```\nimport os\nLIMIT = 3\ndef task_func(x):\n```"
BCB_TEXT = (
    "Compute the sum of a list. It must not modify the input, e.g. by sorting it. "
    "The result is scaled by 2.5 afterwards.\n"
    "Note that: Notes: The title is 'Sum'. Empty lists never fail.\n"
    "The function should raise the exception for: ValueError: If the list is empty.\n"
    "The function should output with:\n    float: The sum. None if nothing was added.\n"
    "You should write self-contained code starting with:\n" + CODE
)
BCB_EXAMPLE_TEXT = (
    "Add two numbers. Args: - a (int): the first one.\n"
    "Example 2: >>> task_func(1)\n2\n"
    "The function should output with:\n    int: The total.\n"
    "You should write self-contained code starting with:\n" + CODE
)
LSE_TEXT = (
    "Generate Python code for the following: \n The code reads a file.The file must exist. "
    "It returns the content.  Then it sends everything to <url>."
)


def source(task: str, text: str, benchmark: str = "bigcodebench") -> SourceRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SourceRecord(
        task, text, digest, digest, benchmark, "1" * 40, "d" * 64, task, None, "prompt", ()
    )


BCB = source("BigCodeBench/1", BCB_TEXT)
BCB_EXAMPLE = source("BigCodeBench/2", BCB_EXAMPLE_TEXT)
LSE = source("LSE-001", LSE_TEXT, "llmseceval")

BCB_RULE = "bcb-markers-v1"
LSE_RULE = "lse-prefix-v1"


def canonical(record, rule: str) -> CanonicalPrompt:
    row = propose(record, rule)
    return CanonicalPrompt(record.task_id, record.text, row.spans, row.annotation_version)


def variants(prompt: CanonicalPrompt, envelope: bool = False) -> dict:
    t, h = render_prompt_variants(prompt)
    found = {"T": t, "H-gold": h}
    if envelope:
        found["E"] = render_envelope_variant(prompt)
    return found


def characters(text: str) -> int:
    """Stand-in tokenizer: one token per character plus a fixed message overhead."""
    return len(text) + 10


def test_structure_counts_leaves_segments_and_reordered_segments() -> None:
    # Source ranks 0..8 on main, constraints, main, format, constraints, constraints,
    # format, constraints, data. H emits main [0, 2], format [3, 6], constraints
    # [1, 4, 5, 7], data [8]: ranks 0 2 3 6 1 4 5 7 8, six out of their source place.
    assert structure(canonical(BCB, BCB_RULE)) == {"n_leaves": 4, "n_segments": 9, "reorder": 6}


def test_reorder_is_zero_when_h_keeps_the_source_order() -> None:
    text = "Do this. Use that data."
    ordered = CanonicalPrompt(
        "x",
        text,
        (
            SpanAnnotation("main_instruction", 0, 8, "Do this."),
            SpanAnnotation("context.data", 9, 23, "Use that data."),
        ),
        "v",
    )
    assert structure(ordered) == {"n_leaves": 2, "n_segments": 2, "reorder": 0}
    swapped = CanonicalPrompt(
        "x",
        text,
        (
            SpanAnnotation("context.data", 0, 8, "Do this."),
            SpanAnnotation("main_instruction", 9, 23, "Use that data."),
        ),
        "v",
    )
    assert structure(swapped)["reorder"] == 2


def test_has_code_rule() -> None:
    assert has_code(BCB_TEXT)  # fenced starting code
    assert has_code("Fix this:\ndef f(x):\n    return x")
    assert has_code("  def g ( ):")
    assert not has_code(LSE_TEXT)
    assert not has_code("Define a function that returns the default value.")
    assert not has_code("the undef x() call")


def test_judgement_units_are_the_prose_sentences_of_the_description() -> None:
    # judgement-units-v2: the two G3 sentences and the two notes sentences (B6); the
    # first sentence is fixed by rule G3.1, the fixed sections and the code are formulas.
    assert judgement_units(canonical(BCB, BCB_RULE), BCB_RULE) == 4
    # "Add two numbers." only, a first sentence: the parameters and the example are
    # marker blocks, so no unit is left to judge.
    assert judgement_units(canonical(BCB_EXAMPLE, BCB_RULE), BCB_RULE) == 0
    # LLMSecEval: the three sentences after the first one ("file.The" is split).
    assert judgement_units(canonical(LSE, LSE_RULE), LSE_RULE) == 3


def test_judgement_units_follow_the_gold_when_sentences_are_joined() -> None:
    draft = canonical(BCB, BCB_RULE)
    spans = sorted(draft.annotations, key=lambda span: span.start)
    joined = SpanAnnotation(
        "main_instruction", spans[1].start, spans[2].end, BCB.text[spans[1].start : spans[2].end]
    )
    gold = CanonicalPrompt(
        BCB.task_id, BCB.text, (spans[0], joined, *spans[3:]), draft.annotation_version
    )
    # The joined G3 sentences and the two notes.
    assert judgement_units(gold, BCB_RULE) == 3


def test_judgement_mask_covers_the_prose_judgements_and_the_notes() -> None:
    mask = judgement_mask(BCB_TEXT, BCB_RULE)
    covered = "".join(c for c, flag in zip(BCB_TEXT, mask, strict=True) if flag)
    assert covered == (
        "It must not modify the input, e.g. by sorting it."
        "The result is scaled by 2.5 afterwards."
        "Note that: Notes: The title is 'Sum'."
        "Empty lists never fail."
    )


def test_prompt_covariates_by_hand() -> None:
    prompt = canonical(BCB, BCB_RULE)
    found = variants(prompt, envelope=True)
    row = prompt_covariates(prompt, found, characters, BCB_RULE)
    t, h, e = (len(found[name].text) for name in ("T", "H-gold", "E"))
    assert row["characters"] == {"T": t, "H-gold": h, "E": e}
    assert row["tokens"] == {"T": t + 10, "H-gold": h + 10, "E": e + 10}
    assert row["ratio_HT"] == (h + 10) / (t + 10)
    assert row["log_ratio_HT"] == pytest.approx(math.log((h + 10) / (t + 10)))
    assert row["log_tokens_T"] == pytest.approx(math.log(t + 10))
    assert (row["n_leaves"], row["n_segments"], row["reorder"]) == (4, 9, 6)
    assert row["has_code"] is True
    assert row["n_judgement_units"] == 4
    assert row["judgement_units_rule"] == "judgement-units-v2"
    assert row["covariates_version"] == "prompt-covariates-v1"


def test_covariates_are_deterministic_and_depend_on_the_prompt_alone() -> None:
    prompts = [canonical(BCB, BCB_RULE), canonical(BCB_EXAMPLE, BCB_RULE)]
    rendered = [v for p in prompts for v in variants(p).values()]
    first = covariate_table(prompts, rendered, characters, BCB_RULE)
    second = covariate_table(prompts, list(reversed(rendered)), characters, BCB_RULE)
    assert first == second
    assert [row["task_id"] for row in first] == ["BigCodeBench/1", "BigCodeBench/2"]


def test_variants_from_another_gold_are_refused() -> None:
    prompt = canonical(BCB, BCB_RULE)
    other = variants(canonical(BCB_EXAMPLE, BCB_RULE))
    with pytest.raises(ValueError, match="not rendered from this gold"):
        prompt_covariates(prompt, other, characters, BCB_RULE)
    with pytest.raises(ValueError, match="T and H-gold are required"):
        prompt_covariates(prompt, {"T": variants(prompt)["T"]}, characters, BCB_RULE)
    with pytest.raises(ValueError, match="cover the same tasks"):
        covariate_table([prompt], list(other.values()), characters, BCB_RULE)


def test_unknown_rule_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown rule"):
        judgement_units(canonical(BCB, BCB_RULE), "rules-r-v9")


class Tokenizer:
    """Counts characters of the rendered conversation, like ``characters``."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, truncation):
        assert tokenize and add_generation_prompt and not truncation
        return list("".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>")


def test_covariates_command_counts_the_whole_model_input(tmp_path: Path, monkeypatch) -> None:
    prompt = canonical(BCB, BCB_RULE)
    canonical_file = tmp_path / "canonical.jsonl"
    canonical_file.write_text(
        json.dumps(
            {
                "task_id": prompt.task_id,
                "text": prompt.text,
                "annotation_version": prompt.annotation_version,
                "annotations": [vars(span) for span in prompt.annotations],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rendered = variants(prompt, envelope=True)
    variants_file = tmp_path / "variants.jsonl"
    variants_file.write_text(
        "".join(json.dumps(v.as_dict()) + "\n" for v in rendered.values()), encoding="utf-8"
    )
    config = tmp_path / "generate.json"
    config.write_text(
        json.dumps({"system_prompt": "SYS", "user_template": "W:\n{prompt}\n"}), encoding="utf-8"
    )
    monkeypatch.setattr(cli, "_load_tokenizer", lambda snapshot: Tokenizer())
    out = tmp_path / "cov" / "covariates.jsonl"
    result = CliRunner().invoke(
        cli.app,
        [
            "covariates",
            str(canonical_file),
            str(variants_file),
            "--rule",
            BCB_RULE,
            "--generate-config",
            str(config),
            "--tokenizer-snapshot",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.exception
    row = json.loads(out.read_text(encoding="utf-8"))
    overhead = len("<system>SYS<user>W:\n\n<assistant>")
    assert row["tokens"] == {
        name: len(variant.text) + overhead for name, variant in rendered.items()
    }
    manifest = json.loads((out.parent / "empirical-covariates_manifest.json").read_text("utf-8"))
    assert manifest["rows"] == 1 and manifest["rule"] == BCB_RULE
    # No overwrite.
    again = CliRunner().invoke(
        cli.app,
        [
            "covariates",
            str(canonical_file),
            str(variants_file),
            "--rule",
            BCB_RULE,
            "--generate-config",
            str(config),
            "--tokenizer-snapshot",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert again.exit_code != 0
