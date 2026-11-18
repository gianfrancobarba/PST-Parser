"""parse-prompts with a pinned and verified parser (R-HAU-03, C-24).

A pilot or final run loads the hf backend at the revision of the parser
identity, with the adapter and conversation template of the reference run, and
refuses any difference before generating. A final run also needs a verified
freeze listing the identity. An interrupted run resumes from its journal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from pstparser.config import ExperimentConfig, ModelConfig, load_experiment
from pstparser.data.predictions import PredictionRecord, load_predictions
from pstparser.inference import parse_prompts as module
from pstparser.models import loader
from pstparser.models.loader import SnapshotIdentity
from pstparser.repro.manifest import digest_directory, digest_text

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/experiments/hauto-parser.yaml"
TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
FILES = {"config.json": "a" * 64, "model.safetensors": "b" * 64}


def config(**model: Any) -> ExperimentConfig:
    loaded = load_experiment(ROOT / CONFIG)
    return loaded.model_copy(update={"model": loaded.model.model_copy(update=model)})


def write_sources(path: Path, count: int = 2) -> Path:
    rows = [{"task_id": f"BigCodeBench/{i}", "text": f"Prompt {i}."} for i in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def make_adapter(path: Path) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"lora")
    return path


def identity(experiment: ExperimentConfig, adapter: Path, **changes: Any) -> dict[str, Any]:
    prompt = (ROOT / experiment.system_prompt_path).read_bytes()
    document: dict[str, Any] = {
        "format": "parser-identity-v1",
        "run_id": "parser-2026-11-12-r3",
        "backend": "hf",
        "model": {"name": experiment.model.name, "revision": experiment.model.revision},
        "adapter": {
            "digest": digest_directory(adapter),
            "chat_template_digest": digest_text(TEMPLATE),
        },
        "system_prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "decoding": experiment.inference.model_dump(),
    }
    document.update(changes)
    return document


def write_json(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def model_digests(experiment: ExperimentConfig, path: Path, **files: str) -> Path:
    document = {
        "format": "model-digests-v1",
        "name": experiment.model.name,
        "revision": experiment.model.revision,
        "files": {**FILES, **files},
    }
    return write_json(path, document)


class Harness:
    """Fake hub, loader and generation; records what was loaded and generated."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str = TEMPLATE):
        """Install the fakes on the module under test."""
        self.loaded = 0
        self.generated: list[int] = []
        self.snapshot = tmp_path / "models--fake" / "snapshots" / str(config().model.revision)
        self.snapshot.mkdir(parents=True)

        def resolve(name: str, revision: str, *, hash_weights: bool = False) -> SnapshotIdentity:
            return SnapshotIdentity(self.snapshot, revision, dict(FILES), hash_weights)

        def load_adapter(model_config: ModelConfig, adapter_dir: Path) -> tuple[Any, Any]:
            self.loaded += 1
            base = SimpleNamespace(name_or_path=str(self.snapshot))
            model = SimpleNamespace(get_base_model=lambda: base, name_or_path=str(self.snapshot))
            tokenizer = SimpleNamespace(chat_template=template, name_or_path=str(adapter_dir))
            return model, tokenizer

        def generate(**kwargs: Any) -> Iterable[PredictionRecord]:
            for record in kwargs["records"]:
                self.generated.append(record.index)
                yield PredictionRecord(
                    record.index, record.prompt, record.target, "{}", 5, 2, "stop"
                )

        monkeypatch.setattr(module.loader, "resolve_snapshot", resolve)
        monkeypatch.setattr(module.loader, "load_adapter", load_adapter)
        monkeypatch.setattr(module.loader, "runtime_provenance", lambda *a: {"fake": True})
        monkeypatch.setattr(module.generation, "generate_predictions", generate)
        monkeypatch.chdir(ROOT)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    harness = Harness(monkeypatch, tmp_path)
    experiment = config()
    adapter = make_adapter(tmp_path / "adapter")
    return {
        "harness": harness,
        "config": experiment,
        "sources": write_sources(tmp_path / "sources.jsonl"),
        "adapter": adapter,
        "identity": write_json(tmp_path / "parser.json", identity(experiment, adapter)),
        "model": model_digests(experiment, tmp_path / "parser-model.json"),
        "run": tmp_path / "run",
    }


def run(setup: dict[str, Any], **changes: Any) -> module.ParseOutcome:
    arguments: dict[str, Any] = {
        "config": setup["config"],
        "sources": setup["sources"],
        "run_dir": setup["run"],
        "adapter_dir": setup["adapter"],
        "purpose": "pilot",
        "identity": setup["identity"],
        "parser_model": setup["model"],
    }
    arguments.update(changes)
    return module.parse_prompts(**arguments)


def test_the_h_auto_configuration_pins_the_reference_parser() -> None:
    document = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    model = ModelConfig(**document["model"])
    assert model.backend == "hf"
    assert model.name == "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
    assert model.revision == "0db785ab56c082e30ae7dea3645d45465fbb5797"
    assert model.load_in_4bit is True
    assert model.dtype == "bfloat16"
    assert set(document["model"]) == set(ModelConfig.model_fields)
    # M03: greedy, with every decoding value written out, repetition_penalty 1.0 included.
    assert document["inference"]["do_sample"] is False
    assert document["inference"]["repetition_penalty"] == 1.0
    assert document["inference"]["max_new_tokens"] == 3072
    assert {"temperature", "top_p", "top_k", "structured_output", "batch_size"} <= set(
        document["inference"]
    )


def test_the_frozen_parser_identity_matches_the_configuration() -> None:
    experiment = load_experiment(ROOT / CONFIG)
    document = json.loads((ROOT / "protocol/hauto/parser.json").read_text(encoding="utf-8"))
    assert document["format"] == "parser-identity-v1"
    assert document["backend"] == "hf"
    assert document["model"] == {
        "name": experiment.model.name,
        "revision": experiment.model.revision,
    }
    assert document["decoding"] == experiment.inference.model_dump()
    prompt = (ROOT / experiment.system_prompt_path).read_bytes()
    assert document["system_prompt_sha256"] == hashlib.sha256(prompt).hexdigest()


@pytest.mark.parametrize("purpose", ["pilot", "final"])
def test_a_pinned_run_without_a_revision_is_refused_before_loading(setup, purpose) -> None:
    with pytest.raises(ValueError, match="revision"):
        run(setup, config=config(revision=None), purpose=purpose)
    assert setup["harness"].loaded == 0


def test_a_pinned_run_with_the_unsloth_backend_is_refused_before_loading(setup) -> None:
    with pytest.raises(ValueError, match="hf"):
        run(setup, config=config(backend="unsloth"))
    assert setup["harness"].loaded == 0


def test_an_adapter_with_another_digest_is_refused_before_loading(setup) -> None:
    (setup["adapter"] / "adapter_model.safetensors").write_bytes(b"other lora")
    with pytest.raises(ValueError, match="adapter"):
        run(setup)
    assert setup["harness"].loaded == 0


def test_a_pinned_run_without_an_adapter_is_refused(setup) -> None:
    with pytest.raises(ValueError, match="adapter"):
        run(setup, adapter_dir=None)


def test_model_files_other_than_the_expected_digests_are_refused_before_loading(
    setup, tmp_path
) -> None:
    other = model_digests(setup["config"], tmp_path / "other.json", **{"config.json": "c" * 64})
    with pytest.raises(ValueError, match="model files"):
        run(setup, parser_model=other)
    assert setup["harness"].loaded == 0


def test_a_decoding_other_than_the_identity_is_refused(setup, tmp_path) -> None:
    changed = identity(
        setup["config"],
        setup["adapter"],
        decoding={**setup["config"].inference.model_dump(), "repetition_penalty": 1.1},
    )
    with pytest.raises(ValueError, match="decoding"):
        run(setup, identity=write_json(tmp_path / "changed.json", changed))


def test_a_conversation_template_other_than_the_reference_run_is_refused(
    tmp_path, monkeypatch
) -> None:
    Harness(monkeypatch, tmp_path, template="{{ messages }}")
    experiment = config()
    adapter = make_adapter(tmp_path / "adapter")
    setup = {
        "config": experiment,
        "sources": write_sources(tmp_path / "sources.jsonl"),
        "adapter": adapter,
        "identity": write_json(tmp_path / "parser.json", identity(experiment, adapter)),
        "model": model_digests(experiment, tmp_path / "parser-model.json"),
        "run": tmp_path / "run",
    }
    with pytest.raises(ValueError, match="template"):
        run(setup)


def test_a_final_run_needs_a_freeze(setup) -> None:
    with pytest.raises(ValueError, match="freeze"):
        run(setup, purpose="final")
    assert setup["harness"].loaded == 0


def test_a_pilot_run_records_the_verified_identity(setup) -> None:
    outcome = run(setup)
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["purpose"] == "pilot"
    assert manifest["freeze_sha256"] is None
    pinned = manifest["parser"]
    assert pinned["revision_verified"] is True
    assert pinned["adapter_verified"] is True
    assert pinned["chat_template_verified"] is True
    assert pinned["model_files_verified"] is True
    assert pinned["revision"] == setup["config"].model.revision
    assert pinned["identity_sha256"] == hashlib.sha256(setup["identity"].read_bytes()).hexdigest()
    assert manifest["decoding"]["repetition_penalty"] == 1.0
    assert manifest["resumed_records"] == 0
    assert len(load_predictions(outcome.predictions_path)) == 2


def test_an_interrupted_run_resumes_after_its_last_complete_prediction(setup) -> None:
    sources = write_sources(setup["sources"], count=3)
    harness = setup["harness"]
    original = module.generation.generate_predictions

    def interrupted(**kwargs: Any) -> Iterable[PredictionRecord]:
        for number, record in enumerate(original(**kwargs)):
            if number == 2:
                raise RuntimeError("runtime disconnected")
            yield record

    module.generation.generate_predictions = interrupted  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            run(setup, sources=sources)
    finally:
        module.generation.generate_predictions = original  # type: ignore[assignment]
    predictions = setup["run"] / "predictions.jsonl"
    with predictions.open("a", encoding="utf-8") as handle:
        handle.write('{"index": 2, "prom')  # a line cut by the disconnection
    with pytest.raises(FileExistsError):
        run(setup, sources=sources)
    harness.generated.clear()
    outcome = run(setup, sources=sources, resume=True)
    assert harness.generated == [2]
    assert [p.index for p in load_predictions(outcome.predictions_path)] == [0, 1, 2]
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["resumed_records"] == 2


def test_a_resume_with_another_plan_is_refused(setup, tmp_path) -> None:
    sources = write_sources(setup["sources"], count=3)
    original = module.generation.generate_predictions

    def interrupted(**kwargs: Any) -> Iterable[PredictionRecord]:
        for number, record in enumerate(original(**kwargs)):
            if number == 1:
                raise RuntimeError("runtime disconnected")
            yield record

    module.generation.generate_predictions = interrupted  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            run(setup, sources=sources)
    finally:
        module.generation.generate_predictions = original  # type: ignore[assignment]
    other = write_sources(tmp_path / "other.jsonl", count=4)
    with pytest.raises(ValueError, match="plan"):
        run(setup, sources=other, resume=True)


def test_a_prequantised_snapshot_keeps_its_own_quantisation(tmp_path, monkeypatch) -> None:
    import transformers

    sha = "0db785ab56c082e30ae7dea3645d45465fbb5797"
    path = tmp_path / "models--fake" / "snapshots" / sha
    path.mkdir(parents=True)
    quantised = {"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}}
    (path / "config.json").write_text(json.dumps(quantised), encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(path))
    calls: list[tuple[str, dict[str, Any]]] = []

    class Recorder:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any) -> Any:
            calls.append((source, kwargs))
            return SimpleNamespace(
                chat_template=TEMPLATE, pad_token="x", eos_token="x", name_or_path=source
            )

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", Recorder)
    monkeypatch.setattr(transformers, "AutoTokenizer", Recorder)
    loader._load_hf(
        ModelConfig(
            backend="hf", name="fake/model", revision=sha, load_in_4bit=True, dtype="bfloat16"
        )
    )
    source, keywords = calls[0]
    assert source == str(path)
    assert "quantization_config" not in keywords
    assert keywords["device_map"] == "auto"


def test_a_final_run_needs_a_freeze_listing_the_identity_and_the_model_digests(
    setup, tmp_path, monkeypatch
) -> None:
    from pstparser.empirical import freeze as freezing

    freeze = write_json(tmp_path / "protocol-freeze-hauto.json", {"format": "fake"})
    listed = {
        "protocol/hauto/parser.json": hashlib.sha256(setup["identity"].read_bytes()).hexdigest()
    }
    monkeypatch.setattr(freezing, "verify_freeze", lambda path, root=None: [])
    monkeypatch.setattr(freezing, "load_freeze", lambda path: {"items": dict(listed)})
    with pytest.raises(ValueError, match=r"parser-model.json"):
        run(setup, purpose="final", freeze=freeze)
    assert setup["harness"].loaded == 0
    listed["protocol/hauto/parser-model.json"] = hashlib.sha256(
        setup["model"].read_bytes()
    ).hexdigest()
    outcome = run(setup, purpose="final", freeze=freeze)
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["purpose"] == "final"
    assert manifest["freeze_sha256"] == hashlib.sha256(freeze.read_bytes()).hexdigest()
