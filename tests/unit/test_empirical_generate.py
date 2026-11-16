"""The generate command, end to end, with a fake model and a fake snapshot."""

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from typer.testing import CliRunner

from pstparser.empirical.cli import app
from pstparser.empirical.contracts import CanonicalPrompt, SpanAnnotation
from pstparser.empirical.messages import fill_user_template
from pstparser.empirical.rendering import render_envelope_variant, render_prompt_variants
from pstparser.models import loader
from pstparser.models.loader import SnapshotIdentity
from pstparser.repro import manifest as manifest_module

SHA = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
OTHER = "0db785ab" + "0" * 32
TEMPLATE = "official-template"
SYSTEM = "You are a test system."
# A body that would break str.format and that repeats the wrapper placeholder.
BRACES = "Write {x}, {prompt} and }{ braces {0}."
# The instruction prefix of the official BigCodeBench harness, as a wrapper.
WRAPPER = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:\n{prompt}\n"
)
V2: dict[str, Any] = {
    "format_version": 2,
    "user_template": WRAPPER,
    "conditions": ["T", "H-gold"],
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wrapped(body: str) -> str:
    return WRAPPER[: WRAPPER.index("{prompt}")] + body + "\n"


def rendering(user: str) -> str:
    return f"[{TEMPLATE}]<system>{SYSTEM}<user>{user}<assistant>"


class FakeTokenizer:
    """Byte-level tokenizer whose rendering depends on the template in use."""

    model_max_length = 4096
    eos_token_id = 2

    def __init__(self, path: Path, template: str = TEMPLATE) -> None:
        """Remember where it was loaded from and which template it applies."""
        self.name_or_path = str(path)
        self.chat_template = template
        self.init_kwargs: dict[str, Any] = {}
        self.rendered: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        chat_template: str | None = None,
        return_tensors: str | None = None,
        truncation: bool = False,
    ) -> Any:
        text = f"[{chat_template or self.chat_template}]" + "".join(
            f"<{m['role']}>{m['content']}" for m in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        self.rendered.append(text)
        ids = list(text.encode("utf-8"))
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, values: Any, **kwargs: Any) -> str:
        return "generated code"


class FakeModel:
    """Model that records each call and answers with two tokens ending in EOS."""

    def __init__(self, path: Path) -> None:
        """Publish Qwen-like generation defaults that must never be applied."""
        self.name_or_path = str(path)
        self.config = SimpleNamespace(
            max_position_embeddings=4096, _commit_hash=None, _attn_implementation="sdpa"
        )
        self.generation_config = SimpleNamespace(
            eos_token_id=[2], to_dict=lambda: {"repetition_penalty": 1.1, "do_sample": True}
        )
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.calls: list[dict[str, Any]] = []

    def parameters(self) -> Any:
        return iter(())

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.cat([input_ids, torch.tensor([[1, 2]])], dim=-1)


class Harness:
    """Files and fakes for one invocation of the command."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Create a coherent snapshot, a variants file and a review record."""
        self.root = tmp_path
        self.snapshot_sha = SHA
        self.tokenizer_template = TEMPLATE
        self.loaded_from: Path | None = None
        self.model: FakeModel | None = None
        self.tokenizer: FakeTokenizer | None = None
        self.gpu = "NVIDIA A100-SXM4-80GB"
        self.variants = tmp_path / "variants.jsonl"
        rows = []
        self.rows = rows
        for task, text in (("task-a", BRACES), ("task-b", "Second task.")):
            canonical = CanonicalPrompt(
                task, text, (SpanAnnotation("main_instruction", 0, len(text), text),), "v1"
            )
            rows.extend(v.as_dict() for v in render_prompt_variants(canonical))
        self.variants.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n"
        )
        (tmp_path / "review.md").write_text("reviewed\n", encoding="utf-8")
        monkeypatch.setattr(loader, "resolve_snapshot", self.resolve)
        monkeypatch.setattr(loader, "load_untuned", self.load)
        monkeypatch.setattr(
            manifest_module,
            "accelerator",
            lambda: {"available": True, "devices": [self.gpu], "driver": "550.54"},
        )

    def snapshot_path(self) -> Path:
        path = self.root / "hub" / "snapshots" / self.snapshot_sha
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": TEMPLATE}), encoding="utf-8"
        )
        return path

    def resolve(self, name: str, revision: str, *, hash_weights: bool = False) -> SnapshotIdentity:
        files = {"tokenizer_config.json": "t" * 64}
        if hash_weights:
            files["model.safetensors"] = "w" * 64
        return SnapshotIdentity(self.snapshot_path(), self.snapshot_sha, files, hash_weights)

    def load(self, config: Any) -> tuple[FakeModel, FakeTokenizer]:
        path = self.loaded_from or self.snapshot_path()
        self.model = FakeModel(path)
        self.tokenizer = FakeTokenizer(path, self.tokenizer_template)
        return self.model, self.tokenizer

    def configuration(self, **changes: Any) -> Path:
        document: dict[str, Any] = {
            "model": {
                "backend": "hf",
                "name": "fake/model",
                "revision": SHA,
                "max_seq_length": 4096,
                "load_in_4bit": False,
                "dtype": "bfloat16",
                "chat_template": "native",
            },
            "inference": {"max_new_tokens": 8},
            "seed": 11,
            "replicate_ids": [0],
            "system_prompt": SYSTEM,
            "protocol_version": "test-protocol",
            "dataset_revision": "test-dataset",
            "purpose": "pilot",
            "review_record": "review.md",
        }
        document.update(changes)
        path = self.root / "generate.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def run(self, directory: str = "out", **changes: Any) -> Any:
        return CliRunner().invoke(
            app,
            [
                "generate",
                str(self.variants),
                str(self.configuration(**changes)),
                str(self.root / directory),
            ],
        )

    def manifest(self, directory: str = "out") -> dict[str, Any]:
        path = self.root / directory / "empirical-generate_manifest.json"
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return document

    def journal(self, directory: str = "out") -> list[dict[str, Any]]:
        path = self.root / directory / "generations.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def test_pilot_run_records_a_verified_revision_and_explicit_decoding(harness) -> None:
    result = harness.run()
    assert result.exit_code == 0, result.exception
    configuration = harness.manifest()["configuration"]
    runtime = configuration["runtime"]
    assert runtime["revision_verified"] is True
    assert runtime["snapshot_sha"] == SHA
    assert runtime["chat_template_matches_official"] is True
    assert runtime["files_sha256"] == {"tokenizer_config.json": "t" * 64}
    assert runtime["weights_hashed"] is False
    assert runtime["model_generation_config"] == {"repetition_penalty": 1.1, "do_sample": True}
    assert configuration["decoding"] == {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8,
        "use_cache": True,
    }
    records = harness.journal()
    assert len(records) == 4
    assert {(r["task_id"], r["condition"]) for r in records} == {
        ("task-a", "T"),
        ("task-a", "H-gold"),
        ("task-b", "T"),
        ("task-b", "H-gold"),
    }
    assert all(r["raw_output"] == "generated code" for r in records)
    assert len(harness.model.calls) == 4
    for call in harness.model.calls:
        assert call["generation_config"].repetition_penalty == 1.0
        assert call["use_model_defaults"] is False


def test_journal_plan_records_the_environment(harness) -> None:
    assert harness.run().exit_code == 0
    environment = harness.manifest()["configuration"]["environment"]
    from importlib import metadata

    for package in ("transformers", "torch", "tokenizers", "huggingface_hub"):
        assert environment["packages"][package] == metadata.version(package)
    assert environment["accelerator"]["devices"] == ["NVIDIA A100-SXM4-80GB"]
    assert environment["accelerator"]["driver"] == "550.54"
    assert environment["python"] == ".".join(str(part) for part in sys.version_info[:3])


def test_pilot_run_with_a_diverging_revision_is_recorded_as_unverified(harness) -> None:
    harness.snapshot_sha = OTHER
    result = harness.run()
    assert result.exit_code == 0, result.exception
    runtime = harness.manifest()["configuration"]["runtime"]
    assert runtime["revision_verified"] is False
    assert runtime["snapshot_sha"] == OTHER
    assert runtime["requested_revision"] == SHA


def test_final_run_with_a_diverging_revision_is_refused_before_loading(harness) -> None:
    harness.snapshot_sha = OTHER
    result = harness.run(purpose="final")
    assert isinstance(result.exception, ValueError)
    assert "snapshot of the requested revision" in str(result.exception)
    assert harness.model is None
    assert not (harness.root / "out").exists()


def test_final_run_is_refused_when_loading_bypassed_the_snapshot(harness) -> None:
    harness.loaded_from = harness.root / "elsewhere"
    result = harness.run(purpose="final")
    assert isinstance(result.exception, ValueError)
    assert "verified model revision" in str(result.exception)
    assert harness.tokenizer.rendered == []
    assert not (harness.root / "out").exists()


def test_final_run_is_refused_with_the_unsloth_backend(harness) -> None:
    model = {"backend": "unsloth", "name": "fake/model", "revision": SHA}
    result = harness.run(purpose="final", model=model)
    assert isinstance(result.exception, ValueError)
    assert "snapshot of the requested revision" in str(result.exception)
    assert harness.model is None


def test_final_run_with_a_verified_revision_hashes_the_weights(harness) -> None:
    result = harness.run(purpose="final")
    assert result.exit_code == 0, result.exception
    runtime = harness.manifest()["configuration"]["runtime"]
    assert runtime["revision_verified"] is True
    assert runtime["weights_hashed"] is True
    assert runtime["files_sha256"]["model.safetensors"] == "w" * 64


@pytest.mark.parametrize("purpose", ["pilot", "final"])
def test_a_replaced_conversation_template_stops_the_run(harness, purpose) -> None:
    harness.tokenizer_template = "llama-3.1"
    result = harness.run(purpose=purpose)
    assert isinstance(result.exception, ValueError)
    assert "conversation template" in str(result.exception)
    assert harness.model.calls == []
    assert not (harness.root / "out").exists()


def test_resume_is_idempotent_on_the_same_environment(harness) -> None:
    assert harness.run().exit_code == 0
    first = harness.journal()
    assert harness.run().exit_code == 0
    assert harness.journal() == first
    assert harness.model.calls == []


def test_resume_on_a_different_gpu_is_refused(harness) -> None:
    assert harness.run().exit_code == 0
    before = harness.journal()
    harness.gpu = "NVIDIA L4"
    result = harness.run()
    assert isinstance(result.exception, ValueError)
    assert "resume refused" in str(result.exception)
    assert harness.journal() == before
    assert harness.model.calls == []


def tokenised(harness: Harness) -> Counter[str]:
    assert harness.tokenizer is not None
    return Counter(harness.tokenizer.rendered)


def expected_tokenisations(harness: Harness, user: Callable[[str], str]) -> Counter[str]:
    """Every request is tokenised in the budget preflight and when generating.

    The first variant of each condition, here those of task-a, is tokenised
    once more by the conversation template check.
    """
    return Counter(
        {
            rendering(user(row["text"])): 3 if row["task_id"] == "task-a" else 2
            for row in harness.rows
        }
    )


def test_fill_user_template_concatenates_without_interpreting_the_body() -> None:
    assert fill_user_template(None, BRACES) == BRACES
    assert fill_user_template("A:\n{prompt}\n", BRACES) == "A:\n" + BRACES + "\n"
    assert fill_user_template("{prompt}", "{prompt}{prompt}") == "{prompt}{prompt}"
    assert fill_user_template("{{x}} {prompt} {y}", "b") == "{{x}} b {y}"


@pytest.mark.parametrize(("template", "found"), [("no placeholder", 0), ("{prompt}{prompt}", 2)])
def test_user_template_requires_exactly_one_placeholder(template, found) -> None:
    with pytest.raises(ValueError, match=f"exactly once, found {found}"):
        fill_user_template(template, "body")


def test_v2_wraps_every_condition_identically_and_records_the_hashes(harness) -> None:
    result = harness.run(**V2)
    assert result.exit_code == 0, result.exception
    configuration = harness.manifest()["configuration"]
    assert configuration["format_version"] == 2
    assert configuration["user_template"] == WRAPPER
    assert configuration["user_template_sha256"] == sha256(WRAPPER)
    assert configuration["system_prompt_sha256"] == sha256(SYSTEM)
    assert configuration["conditions"] == ["T", "H-gold"]
    records = harness.journal()
    assert len(records) == 4
    for record in records:
        assert record["user_message_sha256"] == sha256(wrapped(record["text"]))
    assert tokenised(harness) == expected_tokenisations(harness, wrapped)


def test_hierarchical_body_with_braces_reaches_the_model_intact(harness) -> None:
    assert harness.run(**V2).exit_code == 0
    hierarchical = next(
        row["text"]
        for row in harness.rows
        if row["task_id"] == "task-a" and row["condition"] == "H-gold"
    )
    assert json.loads(hierarchical)["prompt"]["main_instruction"] == [
        {"position": 0, "phrase": BRACES}
    ]
    assert hierarchical.count("{") > 3
    # Template check, budget preflight and generation: three tokenisations each.
    assert harness.tokenizer.rendered.count(rendering(wrapped(hierarchical))) == 3
    assert harness.tokenizer.rendered.count(rendering(wrapped(BRACES))) == 3


def test_wrapper_counts_in_the_token_budget_preflight(harness) -> None:
    longest = max(len(rendering(row["text"]).encode("utf-8")) for row in harness.rows)
    model = {
        "backend": "hf",
        "name": "fake/model",
        "revision": SHA,
        "max_seq_length": longest + 8,
        "load_in_4bit": False,
        "dtype": "bfloat16",
        "chat_template": "native",
    }
    bare = harness.run("bare", model=model)
    assert bare.exit_code == 0, bare.exception
    refused = harness.run("wrapped", model=model, **V2)
    assert isinstance(refused.exception, ValueError)
    assert "no truncation performed" in str(refused.exception)
    assert harness.model.calls == []
    assert not (harness.root / "wrapped").exists()


def test_version_1_configuration_is_accepted_and_sends_the_bare_body(harness) -> None:
    result = harness.run()
    assert result.exit_code == 0, result.exception
    configuration = harness.manifest()["configuration"]
    assert "format_version" not in configuration
    assert configuration["user_template_sha256"] is None
    for record in harness.journal():
        assert record["user_message_sha256"] == record["prompt_hash"]
    assert tokenised(harness) == expected_tokenisations(harness, lambda body: body)


def test_sampling_with_five_replicates_plans_five_requests_per_condition(harness) -> None:
    sampled = {
        "max_new_tokens": 8,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    result = harness.run(inference=sampled, replicate_ids=[1, 2, 3, 4, 5], **V2)
    assert result.exit_code == 0, result.exception
    records = harness.journal()
    assert len(records) == 20
    for task in ("task-a", "task-b"):
        for condition in ("T", "H-gold"):
            cell = [r for r in records if (r["task_id"], r["condition"]) == (task, condition)]
            assert sorted(r["replicate_id"] for r in cell) == [1, 2, 3, 4, 5]
            assert sorted(r["seed"] for r in cell) == [12, 13, 14, 15, 16]
    assert harness.manifest()["configuration"]["decoding"] == {
        "do_sample": True,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8,
        "use_cache": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    assert len(harness.model.calls) == 20
    for call in harness.model.calls:
        passed = call["generation_config"]
        assert (passed.do_sample, passed.temperature, passed.top_p, passed.top_k) == (
            True,
            0.7,
            0.8,
            20,
        )
        assert passed.repetition_penalty == 1.0


HF_MODEL = {"backend": "hf", "name": "fake/model", "revision": SHA, "chat_template": "native"}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"replicate_ids": [0, 1]}, "greedy repetitions are not independent samples"),
        ({"model": {**HF_MODEL, "revision": "c03e6d3"}}, "40-character model revision"),
        ({"model": {**HF_MODEL, "revision": None}}, "40-character model revision"),
        ({**V2, "user_template": "no placeholder"}, "exactly once, found 0"),
        ({**V2, "user_template": "{prompt} and {prompt}"}, "exactly once, found 2"),
        ({**V2, "user_template": 3}, "must be a string or null"),
        ({"user_template": WRAPPER}, "configuration requires exactly"),
        ({"conditions": ["T", "H-gold"]}, "configuration requires exactly"),
        ({**V2, "format_version": 3}, "format_version must be 2"),
        ({**V2, "format_version": 1}, "format_version must be 2"),
        ({"format_version": 2, "user_template": WRAPPER}, "configuration requires exactly"),
        ({**V2, "assistant_prefill": "x"}, "configuration requires exactly"),
        ({**V2, "conditions": ["T"]}, "at least two distinct"),
        ({**V2, "conditions": ["T", "T"]}, "at least two distinct"),
        ({**V2, "conditions": ["T", "E"]}, "exactly the configured conditions"),
        ({**V2, "conditions": ["T", "H-gold", "E"]}, "exactly the configured conditions"),
        ({**V2, "model": {"name": "fake/model", "revision": SHA}}, "must be explicit"),
        (
            {**V2, "model": {"backend": "hf", "name": "fake/model", "revision": SHA}},
            "must be explicit",
        ),
        ({**V2, "model": {**HF_MODEL, "chat_template": "llama-3.1"}}, "use 'native'"),
        ({"inference": {"max_new_tokens": 8, "structured_output": True}}, "unconstrained output"),
        ({"inference": {"max_new_tokens": 8, "batch_size": 2}}, "batch_size=1"),
        ({"purpose": "smoke"}, "purpose must be pilot or final"),
        ({"review_record": "absent.md"}, "review/freeze record is required"),
        ({"system_prompt": " "}, "missing system_prompt"),
    ],
)
def test_invalid_configuration_is_refused_before_loading(harness, changes, message) -> None:
    result = harness.run(**changes)
    assert isinstance(result.exception, ValueError), result.output
    assert message in str(result.exception)
    assert harness.model is None
    assert not (harness.root / "out").exists()


def test_the_core_journal_generates_t_h_gold_and_e_in_one_session(harness) -> None:
    rows = list(harness.rows)
    for task, text in (("task-a", BRACES), ("task-b", "Second task.")):
        canonical = CanonicalPrompt(task, text, (), "")
        rows.append(render_envelope_variant(canonical).as_dict())
    harness.variants.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n"
    )
    harness.rows = rows
    result = harness.run(**{**V2, "conditions": ["T", "H-gold", "E"]})
    assert result.exit_code == 0, result.exception
    assert harness.manifest()["configuration"]["conditions"] == ["T", "H-gold", "E"]
    records = harness.journal()
    assert len(records) == 6
    for start in (0, 3):
        block = records[start : start + 3]
        assert len({r["task_id"] for r in block}) == 1
        assert sorted(r["condition"] for r in block) == ["E", "H-gold", "T"]
    envelope = next(r for r in records if (r["task_id"], r["condition"]) == ("task-a", "E"))
    assert json.loads(envelope["text"]) == {"prompt": BRACES}
    assert envelope["user_message_sha256"] == sha256(wrapped(envelope["text"]))
    assert tokenised(harness) == expected_tokenisations(harness, wrapped)


def test_version_2_accepts_a_null_template(harness) -> None:
    result = harness.run(**{**V2, "user_template": None})
    assert result.exit_code == 0, result.exception
    assert harness.manifest()["configuration"]["user_template_sha256"] is None
