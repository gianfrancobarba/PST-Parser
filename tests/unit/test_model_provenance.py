"""A revision is verified only when model and tokenizer come from its snapshot."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from tokenizers import Tokenizer as RustTokenizer
from tokenizers import models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from pstparser.config import ModelConfig
from pstparser.models import loader
from pstparser.models.loader import (
    BackendError,
    SnapshotIdentity,
    official_chat_template,
    resolve_snapshot,
    runtime_provenance,
    verify_chat_template,
)

ROOT = Path(__file__).resolve().parents[2]
SHA = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
OTHER = "0db785ab" + "0" * 32
TEMPLATE = (
    "{% for m in messages %}<s> {{ m['role'] }} : {{ m['content'] }} </s> {% endfor %}"
    "{% if add_generation_prompt %}<s> assistant :{% endif %}"
)
MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "say hello"}]


def make_snapshot(root: Path, sha: str = SHA, template: str | None = TEMPLATE) -> Path:
    path = root / "models--fake--model" / "snapshots" / sha
    path.mkdir(parents=True)
    (path / "config.json").write_bytes(b'{"a": 1}\n')
    (path / "generation_config.json").write_bytes(b'{"repetition_penalty": 1.1}\n')
    tokenizer_config: dict[str, Any] = {"model_max_length": 64}
    if template is not None:
        tokenizer_config["chat_template"] = template
    (path / "tokenizer_config.json").write_bytes(json.dumps(tokenizer_config).encode())
    (path / "model.safetensors.index.json").write_bytes(b"{}")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (path / "README.md").write_bytes(b"not loaded, not hashed\n")
    return path


def patch_download(monkeypatch: pytest.MonkeyPatch, path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    return calls


def real_tokenizer(template: str = TEMPLATE) -> PreTrainedTokenizerFast:
    words = ["[UNK]", "<s>", "</s>", "system", "user", "assistant", ":", "be", "brief", "say"]
    rust = RustTokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    rust.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=rust, unk_token="[UNK]", eos_token="</s>")
    tokenizer.chat_template = template
    return tokenizer


def loaded(path: Path | str, template: str | None = TEMPLATE) -> tuple[Any, Any]:
    model = SimpleNamespace(
        name_or_path=str(path),
        config=SimpleNamespace(_commit_hash=None, _attn_implementation="sdpa"),
        generation_config=SimpleNamespace(to_dict=lambda: {"repetition_penalty": 1.1}),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    tokenizer = SimpleNamespace(name_or_path=str(path), chat_template=template, init_kwargs={})
    return model, tokenizer


def config(revision: str | None = SHA) -> ModelConfig:
    return ModelConfig(
        backend="hf",
        name="fake/model",
        revision=revision,
        load_in_4bit=False,
        dtype="bfloat16",
        chat_template="native",
    )


def test_snapshot_identity_reads_the_commit_from_the_directory(tmp_path, monkeypatch) -> None:
    path = make_snapshot(tmp_path)
    calls = patch_download(monkeypatch, path)
    snapshot = resolve_snapshot("fake/model", SHA)
    assert calls == [
        {
            "repo_id": "fake/model",
            "revision": SHA,
            "allow_patterns": ["*.json", "*.txt", "*.jinja", "*.safetensors"],
        }
    ]
    assert snapshot.path == path
    assert snapshot.sha == SHA
    assert snapshot.weights_hashed is False
    assert snapshot.files == {
        "config.json": hashlib.sha256(b'{"a": 1}\n').hexdigest(),
        "generation_config.json": hashlib.sha256(b'{"repetition_penalty": 1.1}\n').hexdigest(),
        "model.safetensors.index.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(
            (path / "tokenizer_config.json").read_bytes()
        ).hexdigest(),
    }


def test_weight_shards_are_hashed_on_request(tmp_path, monkeypatch) -> None:
    patch_download(monkeypatch, make_snapshot(tmp_path))
    snapshot = resolve_snapshot("fake/model", SHA, hash_weights=True)
    assert snapshot.weights_hashed is True
    assert (
        snapshot.files["model-00001-of-00001.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    )
    assert "README.md" not in snapshot.files


def test_a_branch_resolves_to_the_commit_on_disk_not_to_its_name(tmp_path, monkeypatch) -> None:
    patch_download(monkeypatch, make_snapshot(tmp_path, OTHER))
    assert resolve_snapshot("fake/model", "main").sha == OTHER


def test_a_directory_that_is_not_a_snapshot_is_refused(tmp_path, monkeypatch) -> None:
    plain = tmp_path / "local-model"
    plain.mkdir()
    patch_download(monkeypatch, plain)
    with pytest.raises(BackendError, match="not a commit snapshot"):
        resolve_snapshot("fake/model", SHA)


def test_revision_is_verified_with_a_coherent_snapshot(tmp_path, monkeypatch) -> None:
    path = make_snapshot(tmp_path)
    patch_download(monkeypatch, path)
    snapshot = resolve_snapshot("fake/model", SHA)
    record = runtime_provenance(*loaded(path), config(), snapshot)
    assert record["revision_verified"] is True
    assert record["snapshot_sha"] == SHA
    assert record["files_sha256"] == snapshot.files
    assert record["chat_template_matches_official"] is True
    assert record["official_chat_template_digest"] == record["chat_template_digest"]
    assert record["model_generation_config"] == {"repetition_penalty": 1.1}
    assert record["quantization"] is None
    assert record["attn_implementation"] == "sdpa"
    assert record["dtype"] == "torch.bfloat16"
    assert record["device"] == "cpu"


def test_revision_is_not_verified_when_the_snapshot_diverges(tmp_path, monkeypatch) -> None:
    path = make_snapshot(tmp_path, OTHER)
    patch_download(monkeypatch, path)
    snapshot = resolve_snapshot("fake/model", SHA)
    record = runtime_provenance(*loaded(path), config(), snapshot)
    assert record["snapshot_sha"] == OTHER
    assert record["requested_revision"] == SHA
    assert record["revision_verified"] is False


@pytest.mark.parametrize("which", ["model", "tokenizer"])
def test_revision_is_not_verified_when_loading_bypassed_the_snapshot(
    tmp_path, monkeypatch, which
) -> None:
    path = make_snapshot(tmp_path)
    patch_download(monkeypatch, path)
    snapshot = resolve_snapshot("fake/model", SHA)
    model, tokenizer = loaded(path)
    {"model": model, "tokenizer": tokenizer}[which].name_or_path = "fake/model"
    assert runtime_provenance(model, tokenizer, config(), snapshot)["revision_verified"] is False


def test_without_a_snapshot_nothing_is_claimed(tmp_path) -> None:
    record = runtime_provenance(*loaded("fake/model"), config(revision=None))
    assert record["revision_verified"] is False
    assert record["snapshot_sha"] is None
    assert record["files_sha256"] is None
    assert record["chat_template_matches_official"] is None


def test_a_replaced_template_is_reported(tmp_path, monkeypatch) -> None:
    path = make_snapshot(tmp_path)
    patch_download(monkeypatch, path)
    snapshot = resolve_snapshot("fake/model", SHA)
    record = runtime_provenance(*loaded(path, template="llama"), config(), snapshot)
    assert record["chat_template_matches_official"] is False
    assert record["official_chat_template_digest"] != record["chat_template_digest"]


def test_standalone_template_file_takes_precedence_and_keeps_its_bytes(tmp_path) -> None:
    path = make_snapshot(tmp_path)
    (path / "chat_template.jinja").write_bytes(b"line\r\nkept")
    snapshot = SnapshotIdentity(path=path, sha=SHA, files={}, weights_hashed=False)
    assert official_chat_template(snapshot) == "line\r\nkept"


def test_template_check_passes_with_the_published_template(tmp_path) -> None:
    snapshot = SnapshotIdentity(make_snapshot(tmp_path), SHA, {}, False)
    verify_chat_template(real_tokenizer(), snapshot, MESSAGES)


def test_template_check_fails_with_another_template(tmp_path) -> None:
    snapshot = SnapshotIdentity(make_snapshot(tmp_path), SHA, {}, False)
    other = TEMPLATE.replace("<s> assistant :", "<s> user :")
    with pytest.raises(ValueError, match="does not render the published"):
        verify_chat_template(real_tokenizer(other), snapshot, MESSAGES)


def test_template_check_fails_without_a_published_template(tmp_path) -> None:
    snapshot = SnapshotIdentity(make_snapshot(tmp_path, template=None), SHA, {}, False)
    with pytest.raises(ValueError, match="publishes no conversation template"):
        verify_chat_template(real_tokenizer(), snapshot, MESSAGES)


class Recorder:
    """Stand-in for an ``Auto*`` class that records how it was asked to load."""

    def __init__(self) -> None:
        """Start with no recorded call."""
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, source: str, **kwargs: Any) -> Any:
        self.calls.append((source, kwargs))
        return SimpleNamespace(
            chat_template=TEMPLATE,
            pad_token="</s>",
            eos_token="</s>",
            name_or_path=source,
            config=SimpleNamespace(_commit_hash=None),
        )


@pytest.mark.parametrize(("cuda", "device_map"), [(False, None), (True, {"": 0})])
def test_pinned_hf_loading_uses_the_snapshot_directory(
    tmp_path, monkeypatch, cuda, device_map
) -> None:
    import transformers

    path = make_snapshot(tmp_path)
    patch_download(monkeypatch, path)
    models_, tokenizers_ = Recorder(), Recorder()
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", models_)
    monkeypatch.setattr(transformers, "AutoTokenizer", tokenizers_)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    loader._load_hf(config())
    source, keywords = models_.calls[0]
    assert source == str(path)
    assert keywords.get("device_map") == device_map
    assert keywords["attn_implementation"] == "sdpa"
    assert keywords["dtype"] is torch.bfloat16
    assert "quantization_config" not in keywords
    assert "revision" not in keywords
    assert tokenizers_.calls == [(str(path), {})]


def test_unpinned_hf_loading_is_unchanged(monkeypatch) -> None:
    import transformers

    models_, tokenizers_ = Recorder(), Recorder()
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", models_)
    monkeypatch.setattr(transformers, "AutoTokenizer", tokenizers_)
    loader._load_hf(ModelConfig(backend="hf", name="fake/model", load_in_4bit=False))
    assert models_.calls == [("fake/model", {"revision": None})]
    assert tokenizers_.calls == [("fake/model", {"revision": None})]


def test_qwen_configuration_carries_no_unsloth_or_llama_default() -> None:
    document = yaml.safe_load(
        (ROOT / "configs" / "_base" / "model" / "qwen25_coder_7b.yaml").read_text(encoding="utf-8")
    )
    model = ModelConfig(**document["model"])
    assert model.backend == "hf"
    assert model.chat_template == "native"
    assert model.name == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert model.revision == SHA
    assert model.load_in_4bit is False
    assert model.dtype == "bfloat16"
    assert model.max_seq_length == 8192
    # Every field is written out: none is left to a schema default.
    assert set(document["model"]) == set(ModelConfig.model_fields)
