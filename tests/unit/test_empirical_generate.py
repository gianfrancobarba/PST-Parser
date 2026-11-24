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
from pstparser.empirical.freeze import build_freeze, write_freeze
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
        self.driver = "550.54"
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
            lambda: {"available": True, "devices": [self.gpu], "driver": self.driver},
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
            "gpu": "NVIDIA A100-SXM4-80GB",
        }
        document.update(changes)
        # A final run needs a verified protocol freeze listing its inputs:
        # unless a test names its own record, one is written for these files.
        frozen = document["purpose"] == "final" and "review_record" not in changes
        if frozen:
            document["review_record"] = "protocol/protocol-freeze.json"
        path = self.root / "generate.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        if frozen:
            self.freeze(["generate.json", "variants.jsonl"])
        return path

    def freeze(self, items: list[str]) -> Path:
        """Write a minimal repository (source, lock) and a fresh freeze of ``items``."""
        package = self.root / "src" / "pstparser"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
        (self.root / "uv.lock").write_text("lock\n", encoding="utf-8")
        (self.root / "protocol").mkdir(exist_ok=True)
        target = self.root / "protocol" / "protocol-freeze.json"
        target.unlink(missing_ok=True)
        # H-gold of a final run comes from a listed rendering of frozen gold.
        gold = self.root / "annotations.jsonl"
        gold.write_text('{"status": "frozen"}\n', encoding="utf-8", newline="\n")
        record = self.root / "variants.jsonl.render.json"
        record.unlink(missing_ok=True)
        hashes = sorted(row["prompt_hash"] for row in self.rows if row["condition"] == "H-gold")
        record.write_text(
            json.dumps(
                {
                    "gold_status": "frozen",
                    "annotations_sha256": hashlib.sha256(gold.read_bytes()).hexdigest(),
                    "h_gold_prompt_hashes": hashes,
                }
            ),
            encoding="utf-8",
        )
        # The expected model files, unless a test wrote its own.
        model = self.root / "protocol" / "model.json"
        if not model.exists():
            files = {"tokenizer_config.json": "t" * 64, "model.safetensors": "w" * 64}
            model.write_text(
                json.dumps({"name": "fake/model", "revision": SHA, "files": files}),
                encoding="utf-8",
            )
        items = [*items, "annotations.jsonl", "variants.jsonl.render.json", "protocol/model.json"]
        meta = {"protocol_version": "test-1.0.0", "frozen_by": "test"}
        return write_freeze(build_freeze(self.root, items, meta), target)

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
    # The driver is recorded per completion, outside the frozen plan.
    assert "driver" not in environment["accelerator"]
    assert {record["gpu_driver"] for record in harness.journal()} == {"550.54"}
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
    assert "hf backend" in str(result.exception)
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
    before, loaded = harness.journal(), harness.model
    harness.gpu = "NVIDIA L4"
    result = harness.run()
    assert isinstance(result.exception, ValueError)
    assert "declared GPU" in str(result.exception)
    assert harness.journal() == before
    assert harness.model is loaded  # refused before the model is loaded again


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
        # S-PREFILL: the prefill is a version 2 key, refused in version 1.
        ({"assistant_prefill": "x"}, "configuration requires exactly"),
        ({**V2, "assistant_prefill": ""}, "non-empty string or null"),
        ({**V2, "assistant_prefill": 3}, "non-empty string or null"),
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


def test_peak_vram_is_recorded_per_request_on_a_gpu(harness, monkeypatch) -> None:
    from pstparser.empirical import cli

    resets: list[bool] = []
    peaks = iter([111, 222, 333, 444])
    monkeypatch.setattr(cli, "_reset_peak_vram", lambda: resets.append(True) or True)
    monkeypatch.setattr(cli, "_peak_vram_bytes", lambda: next(peaks))
    result = harness.run(**V2)
    assert result.exit_code == 0, result.exception
    assert [r["peak_vram_bytes"] for r in harness.journal()] == [111, 222, 333, 444]
    assert len(resets) == 4
    # The optional field does not disturb a resume.
    again = harness.run(**V2)
    assert again.exit_code == 0, again.exception
    assert len(harness.journal()) == 4


def test_no_peak_vram_field_without_a_gpu(harness) -> None:
    assert harness.run(**V2).exit_code == 0
    assert all("peak_vram_bytes" not in r for r in harness.journal())


# --- S-PREFILL ---------------------------------------------------------------------

PREFILL = "```python\n"


def recording_generate(seen: list[bytes]) -> Callable[..., torch.Tensor]:
    def generate(self: FakeModel, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seen.append(bytes(int(value) for value in input_ids[0]))
        self.calls.append(kwargs)
        return torch.cat([input_ids, torch.tensor([[1, 2]])], dim=-1)

    return generate


def test_the_prefill_follows_the_generation_prompt_identically_in_every_condition(
    harness, monkeypatch
) -> None:
    seen: list[bytes] = []
    monkeypatch.setattr(FakeModel, "generate", recording_generate(seen))
    result = harness.run(**V2, assistant_prefill=PREFILL)
    assert result.exit_code == 0, result.exception
    records = harness.journal()
    assert len(seen) == len(records) == 4
    expected = {(rendering(wrapped(row["text"])) + PREFILL).encode("utf-8") for row in harness.rows}
    assert set(seen) == expected
    for record in records:
        # The journal keeps what the model generated, without the prefill.
        assert record["raw_output"] == "generated code"
        assert record["input_tokens"] == len(
            (rendering(wrapped(record["text"])) + PREFILL).encode("utf-8")
        )
    configuration = harness.manifest()["configuration"]
    assert configuration["assistant_prefill"] == PREFILL
    assert configuration["assistant_prefill_sha256"] == sha256(PREFILL)


def test_the_prefill_counts_in_the_token_budget_preflight(harness) -> None:
    longest = max(len(rendering(wrapped(row["text"])).encode("utf-8")) for row in harness.rows)
    model = {
        "backend": "hf",
        "name": "fake/model",
        "revision": SHA,
        "max_seq_length": longest + 8,
        "load_in_4bit": False,
        "dtype": "bfloat16",
        "chat_template": "native",
    }
    plain = harness.run("plain", model=model, **V2)
    assert plain.exit_code == 0, plain.exception
    refused = harness.run("prefilled", model=model, **V2, assistant_prefill=PREFILL)
    assert isinstance(refused.exception, ValueError)
    assert "no truncation performed" in str(refused.exception)
    assert not (harness.root / "prefilled").exists()


def test_without_a_prefill_the_frozen_configuration_is_unchanged(harness, monkeypatch) -> None:
    seen: list[bytes] = []
    monkeypatch.setattr(FakeModel, "generate", recording_generate(seen))
    assert harness.run(**V2).exit_code == 0
    configuration = harness.manifest()["configuration"]
    assert "assistant_prefill" not in configuration
    assert "assistant_prefill_sha256" not in configuration
    assert set(seen) == {rendering(wrapped(row["text"])).encode("utf-8") for row in harness.rows}
    # A null prefill is the primary analysis: same messages, recorded as null.
    seen.clear()
    assert harness.run("null", **V2, assistant_prefill=None).exit_code == 0
    assert harness.manifest("null")["configuration"]["assistant_prefill"] is None
    assert "assistant_prefill_sha256" not in harness.manifest("null")["configuration"]
    assert set(seen) == {rendering(wrapped(row["text"])).encode("utf-8") for row in harness.rows}


# --- Session identity ------------------------------------------------------------------


def test_every_completion_records_its_session_and_a_resume_adds_a_second_one(
    harness, monkeypatch
) -> None:
    calls = {"count": 0}
    original = FakeModel.generate

    def interrupted(self: FakeModel, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("session ended")
        return original(self, input_ids, **kwargs)

    monkeypatch.setattr(FakeModel, "generate", interrupted)
    first = harness.run(**V2)
    assert isinstance(first.exception, RuntimeError)
    assert len(harness.journal()) == 2
    resumed = harness.run(**V2)
    assert resumed.exit_code == 0, resumed.exception
    records = harness.journal()
    assert len(records) == 4
    sessions = [record["session_id"] for record in records]
    assert len(set(sessions)) == 2 and sessions[0] == sessions[1] != sessions[2] == sessions[3]
    assert all(len(session) == 32 for session in sessions)
    assert {record["gpu_name"] for record in records} == {"NVIDIA A100-SXM4-80GB"}
    assert len({record["environment_digest"] for record in records}) == 1
    # A resume on another GPU is still refused: the session cannot hide it.
    harness.gpu = "NVIDIA L4"
    other = harness.run(**V2)
    assert isinstance(other.exception, ValueError)


# --- Protocol freeze -------------------------------------------------------------------


def test_a_final_run_on_a_freeze_that_does_not_list_the_configuration_is_refused(harness) -> None:
    harness.freeze(["variants.jsonl"])
    result = harness.run(purpose="final", review_record="protocol/protocol-freeze.json")
    assert isinstance(result.exception, ValueError)
    assert "does not list the configuration" in str(result.exception)
    assert harness.model is None
    assert not (harness.root / "out").exists()


def test_a_final_run_on_a_plain_review_record_is_refused(harness) -> None:
    result = harness.run(purpose="final", review_record="review.md")
    assert isinstance(result.exception, ValueError)
    assert "require a protocol freeze" in str(result.exception)
    assert harness.model is None


def test_a_final_run_on_a_valid_freeze_records_its_hash(harness) -> None:
    result = harness.run(purpose="final")
    assert result.exit_code == 0, result.exception
    freeze = harness.root / "protocol" / "protocol-freeze.json"
    configuration = harness.manifest()["configuration"]
    assert configuration["review_record"] == "protocol/protocol-freeze.json"
    assert configuration["review_record_sha256"] == hashlib.sha256(freeze.read_bytes()).hexdigest()


def test_a_pilot_run_keeps_a_plain_review_record(harness) -> None:
    result = harness.run()
    assert result.exit_code == 0, result.exception
    assert harness.manifest()["configuration"]["review_record"] == "review.md"


# A pilot measures the same treatment as the final runs, so a
# v1 configuration without a backend may no longer fall back to the optimised
# loader and the Llama template.
@pytest.mark.parametrize("purpose", ["pilot", "final"])
def test_a_v1_configuration_without_backend_is_refused_before_loading(harness, purpose) -> None:
    result = harness.run(purpose=purpose, model={"name": "fake/model", "revision": SHA})
    assert isinstance(result.exception, ValueError)
    assert "hf backend" in str(result.exception)
    assert harness.model is None
    assert not (harness.root / "out").exists()


def test_a_pilot_with_the_unsloth_backend_is_refused_before_loading(harness) -> None:
    model = {"backend": "unsloth", "name": "fake/model", "revision": SHA, "chat_template": "chatml"}
    result = harness.run(purpose="pilot", model=model)
    assert isinstance(result.exception, ValueError)
    assert "hf backend" in str(result.exception)
    assert harness.model is None


# --- One declared GPU and resume rule -----------------------------


def test_a_configuration_must_declare_its_gpu(harness) -> None:
    for changes in ({}, V2):
        path = harness.configuration(**changes)
        document = json.loads(path.read_text(encoding="utf-8"))
        document.pop("gpu", None)
        path.write_text(json.dumps(document), encoding="utf-8")
        result = CliRunner().invoke(
            app, ["generate", str(harness.variants), str(path), str(harness.root / "out")]
        )
        assert isinstance(result.exception, ValueError)
        assert "gpu" in str(result.exception)
        assert harness.model is None


@pytest.mark.parametrize("purpose", ["pilot", "final"])
def test_a_run_on_a_gpu_other_than_the_declared_one_is_refused_before_loading(
    harness, purpose
) -> None:
    harness.gpu = "NVIDIA L4"
    result = harness.run(purpose=purpose, gpu="NVIDIA A100-SXM4-40GB")
    assert isinstance(result.exception, ValueError)
    assert "declared GPU" in str(result.exception)
    assert harness.model is None
    assert not (harness.root / "out").exists()


def test_a_run_on_the_declared_gpu_records_it(harness) -> None:
    harness.gpu = "NVIDIA L4"
    result = harness.run(gpu="NVIDIA L4")
    assert result.exit_code == 0, result.exception
    assert harness.manifest()["configuration"]["gpu"] == "NVIDIA L4"
    assert {record["gpu_name"] for record in harness.journal()} == {"NVIDIA L4"}


def test_a_resume_after_a_kernel_or_driver_update_is_accepted(harness, monkeypatch) -> None:
    calls = {"count": 0}
    original = FakeModel.generate

    def interrupted(self: FakeModel, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("session ended")
        return original(self, input_ids, **kwargs)

    monkeypatch.setattr(FakeModel, "generate", interrupted)
    assert isinstance(harness.run().exception, RuntimeError)
    harness.driver = "560.10"
    monkeypatch.setattr(manifest_module.platform, "platform", lambda: "Linux-6.9-colab")
    resumed = harness.run()
    assert resumed.exit_code == 0, resumed.exception
    records = harness.journal()
    assert [record["gpu_driver"] for record in records] == ["550.54"] * 2 + ["560.10"] * 2
    assert records[-1]["platform"] == "Linux-6.9-colab"
    assert len({record["environment_digest"] for record in records}) == 1
    environment = harness.manifest()["configuration"]["environment"]
    assert "platform" not in environment and "driver" not in environment["accelerator"]


# --- Expected model files ----------------------------------------------


def test_a_final_run_with_a_file_digest_other_than_the_frozen_one_is_refused(harness) -> None:
    (harness.root / "protocol").mkdir(exist_ok=True)
    files = {"tokenizer_config.json": "t" * 64, "model.safetensors": "0" * 64}
    (harness.root / "protocol" / "model.json").write_text(
        json.dumps({"name": "fake/model", "revision": SHA, "files": files}), encoding="utf-8"
    )
    result = harness.run(purpose="final")
    assert isinstance(result.exception, ValueError)
    assert "model.safetensors" in str(result.exception)
    assert harness.model is None


def test_a_final_run_needs_the_expected_model_files_in_the_freeze(harness) -> None:
    (harness.root / "protocol").mkdir(exist_ok=True)
    files = {"tokenizer_config.json": "t" * 64}  # the weight shard is not listed
    (harness.root / "protocol" / "model.json").write_text(
        json.dumps({"name": "fake/model", "revision": SHA, "files": files}), encoding="utf-8"
    )
    result = harness.run(purpose="final")
    assert isinstance(result.exception, ValueError)
    assert "model.safetensors" in str(result.exception)


def test_model_digests_writes_every_file_of_the_snapshot(harness) -> None:
    out = harness.root / "digests" / "model.json"
    result = CliRunner().invoke(
        app, ["model-digests", "--name", "fake/model", "--revision", SHA, "--out", str(out)]
    )
    assert result.exit_code == 0, result.exception
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["revision"] == SHA
    assert document["files"] == {"model.safetensors": "w" * 64, "tokenizer_config.json": "t" * 64}
