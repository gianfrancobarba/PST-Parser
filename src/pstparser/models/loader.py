"""Loading of the base model, the tokenizer and the low-rank adapters.

Two backends are available. ``unsloth`` uses the optimised loader, which patches
transformers and TRL when it is imported; ``hf`` uses transformers together with
peft and imposes no such side effects. Every import of the optimised backend is
confined to this module so that the rest of the package, and the test suite, can
run without it.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pstparser.config import LoraConfig, ModelConfig
from pstparser.config.schema import Precision

Model = Any
Tokenizer = Any


# Files of a model repository that decide what is loaded: configuration,
# generation defaults, tokenizer vocabulary and template, weight index, weights.
SNAPSHOT_PATTERNS = ("*.json", "*.txt", "*.jinja", "*.safetensors")
_WEIGHT_PATTERN = "*.safetensors"
_COMMIT = re.compile(r"[0-9a-f]{40}")


class BackendError(RuntimeError):
    """Raised when a backend is unavailable or cannot honour the configuration."""


@dataclass(frozen=True)
class SnapshotIdentity:
    """A model repository materialised at one commit.

    Attributes:
        path: Local directory of the snapshot, ending in ``snapshots/<sha>``.
        sha: Commit the directory holds, read from its name.
        files: SHA-256 of every hashed file, keyed by its path in the repository.
        weights_hashed: Whether ``files`` covers the weight shards as well.
    """

    path: Path
    sha: str
    files: dict[str, str]
    weights_hashed: bool


def resolve_snapshot(name: str, revision: str, *, hash_weights: bool = False) -> SnapshotIdentity:
    """Download a repository at a revision and identify exactly what was fetched.

    The commit is read from the snapshot directory the hub client resolves, so
    a branch name, or a revision the hub maps elsewhere, is reported as the
    commit actually on disk rather than as the value that was asked for.

    Args:
        name: Repository identifier on the Hugging Face Hub.
        revision: Commit hash, branch or tag to resolve.
        hash_weights: Also hash the weight shards. It reads every byte of the
            model, which takes minutes, so it is left to the runs that need it.

    Returns:
        The snapshot directory, its commit and the digests of its files.

    Raises:
        BackendError: If the resolved directory is not a commit snapshot.
    """
    from huggingface_hub import snapshot_download

    from pstparser.repro.manifest import digest_file

    path = Path(
        snapshot_download(repo_id=name, revision=revision, allow_patterns=list(SNAPSHOT_PATTERNS))
    )
    if path.parent.name != "snapshots" or _COMMIT.fullmatch(path.name) is None:
        raise BackendError(f"{path} is not a commit snapshot of {name!r}")
    files: dict[str, str] = {}
    for file in sorted(path.rglob("*")):
        if not file.is_file() or not any(fnmatch.fnmatch(file.name, p) for p in SNAPSHOT_PATTERNS):
            continue
        if fnmatch.fnmatch(file.name, _WEIGHT_PATTERN) and not hash_weights:
            continue
        relative = file.relative_to(path).as_posix()
        digest = digest_file(file)
        if digest is None:
            raise BackendError(f"cannot hash snapshot file {relative}")
        files[relative] = digest
    return SnapshotIdentity(path=path, sha=path.name, files=files, weights_hashed=hash_weights)


def verify_expected_files(snapshot: SnapshotIdentity, expected: Mapping[str, Any]) -> None:
    """Compare the hashed files of a snapshot with the frozen expected digests (R-GEN-02).

    ``expected`` is ``protocol/model.json``: ``name``, ``revision`` and
    ``files`` (path in the repository to SHA-256). Every hashed file must be
    expected with the same digest and every expected file must be present, so
    a snapshot directory with other contents never passes for the revision.

    Raises:
        BackendError: If the revision or any file differs.
    """
    if expected.get("revision") != snapshot.sha:
        raise BackendError(f"model.json expects revision {expected.get('revision')}")
    wanted = dict(expected.get("files", {}))
    differing = sorted(
        name
        for name in set(wanted) | set(snapshot.files)
        if wanted.get(name) != snapshot.files.get(name)
    )
    if differing:
        raise BackendError(f"model files differ from the frozen digests: {differing[:5]}")


def official_chat_template(snapshot: SnapshotIdentity) -> str | None:
    """Read the conversation template published in a snapshot, byte for byte."""
    standalone = snapshot.path / "chat_template.jinja"
    if standalone.is_file():
        return standalone.read_bytes().decode("utf-8")
    configuration = snapshot.path / "tokenizer_config.json"
    if not configuration.is_file():
        return None
    template = json.loads(configuration.read_bytes().decode("utf-8")).get("chat_template")
    return template if isinstance(template, str) else None


def verify_chat_template(
    tokenizer: Tokenizer, snapshot: SnapshotIdentity, messages: list[dict[str, str]]
) -> None:
    """Check that a request is tokenised as the published template prescribes.

    The text rendered by the tokenizer in use must equal the text rendered from
    the template stored in the snapshot, and the token ids sent to the model
    must be the tokenisation of that text.

    Raises:
        ValueError: If the snapshot has no template, or either check fails.
    """
    official = official_chat_template(snapshot)
    if official is None:
        raise ValueError("the model snapshot publishes no conversation template")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected = tokenizer.apply_chat_template(
        messages, chat_template=official, tokenize=False, add_generation_prompt=True
    )
    if rendered != expected:
        raise ValueError("the tokenizer does not render the published conversation template")
    sent = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, truncation=False
    )
    direct = tokenizer(expected, add_special_tokens=False)["input_ids"]
    if list(sent) != list(direct):
        raise ValueError("template token ids differ from the tokenised published rendering")


def _loaded_from(thing: Any, snapshot: SnapshotIdentity) -> bool:
    origin = getattr(thing, "name_or_path", None)
    return isinstance(origin, str) and bool(origin) and Path(origin) == snapshot.path


def _as_dict(thing: Any) -> Any:
    return thing.to_dict() if hasattr(thing, "to_dict") else thing


def runtime_provenance(
    model: Model,
    tokenizer: Tokenizer,
    config: ModelConfig,
    snapshot: SnapshotIdentity | None = None,
) -> dict[str, Any]:
    """Record what was actually loaded, without inventing unresolved hashes.

    ``revision_verified`` holds only when a snapshot was resolved, its commit
    is the requested revision, and both the model and the tokenizer were loaded
    from that snapshot directory. ``model_generation_config`` is the set of
    defaults published with the model: it is recorded, never applied.
    """
    from pstparser.repro.manifest import digest_text

    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    template = getattr(tokenizer, "chat_template", None)
    official = official_chat_template(snapshot) if snapshot is not None else None
    published = getattr(model, "generation_config", None)
    quantization = getattr(getattr(model, "config", None), "quantization_config", None)
    return {
        "requested_model": config.name,
        "requested_revision": config.revision,
        "resolved_model_revision": model_revision,
        "resolved_tokenizer_revision": tokenizer_revision,
        "chat_template_digest": digest_text(str(template)) if template is not None else None,
        "backend": config.backend,
        "load_in_4bit": config.load_in_4bit,
        "dtype": str(getattr(model, "dtype", None)),
        "snapshot_sha": snapshot.sha if snapshot is not None else None,
        "files_sha256": dict(snapshot.files) if snapshot is not None else None,
        "weights_hashed": snapshot.weights_hashed if snapshot is not None else None,
        "model_generation_config": _as_dict(published) if published is not None else None,
        "quantization": _as_dict(quantization) if quantization is not None else None,
        "attn_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
        "device": str(getattr(model, "device", None)),
        "official_chat_template_digest": digest_text(official) if official is not None else None,
        "chat_template_matches_official": (
            template == official if snapshot is not None and official is not None else None
        ),
        "revision_verified": bool(
            snapshot is not None
            and config.revision is not None
            and snapshot.sha == config.revision
            and _loaded_from(model, snapshot)
            and _loaded_from(tokenizer, snapshot)
        ),
    }


def load_base(config: ModelConfig) -> tuple[Model, Tokenizer]:
    """Load the base model and its tokenizer, with the conversation template set.

    Args:
        config: Base model selection and loading options.

    Returns:
        The model and the tokenizer.

    Raises:
        BackendError: If the selected backend is not installed, or if the
            tokenizer carries no conversation template.
    """
    if config.backend == "unsloth":
        return _load_unsloth(config)
    return _load_hf(config)


def attach_adapter(
    model: Model, model_config: ModelConfig, lora_config: LoraConfig, seed: int
) -> Model:
    """Wrap a base model with trainable low-rank adapters.

    Args:
        model: The loaded base model.
        model_config: Determines which backend performs the wrapping.
        lora_config: Adapter parameters.
        seed: The experiment seed, governing the initial adapter weights. It is
            passed in rather than read from the adapter parameters so that one
            value governs every source of randomness in a run.

    Returns:
        The adapted model.

    Raises:
        BackendError: If the selected backend is not installed.
    """
    if model_config.backend == "unsloth":
        FastLanguageModel = _import_unsloth()
        return FastLanguageModel.get_peft_model(
            model,
            r=lora_config.r,
            target_modules=list(lora_config.target_modules),
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
            random_state=seed,
        )

    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
    except ImportError as exc:
        raise BackendError("peft is required to attach adapters with the 'hf' backend") from exc

    return get_peft_model(
        model,
        PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            target_modules=list(lora_config.target_modules),
            task_type="CAUSAL_LM",
        ),
    )


def load_untuned(config: ModelConfig) -> tuple[Model, Tokenizer]:
    """Load the base model for inference, with no adapter on top.

    This is what answers the same test set without having been trained on it.
    Its scores are the floor the fine-tuned ones have to be read against: on
    their own they say how well the task is done, not how much the training did.

    Args:
        config: Base model selection and loading options.

    Returns:
        The base model, switched to inference mode, and its tokenizer.

    Raises:
        BackendError: If the selected backend is not installed.
    """
    model, tokenizer = load_base(config)
    if config.backend == "unsloth":
        FastLanguageModel = _import_unsloth()
        FastLanguageModel.for_inference(model)
    else:
        model.eval()
    return model, tokenizer


def load_adapter(config: ModelConfig, adapter_dir: str | Path) -> tuple[Model, Tokenizer]:
    """Load a previously trained adapter on top of its base model.

    Args:
        config: Base model selection and loading options.
        adapter_dir: Directory holding the saved adapter and tokenizer.

    Returns:
        The adapted model, switched to inference mode, and its tokenizer.

    Raises:
        BackendError: If the selected backend is not installed.
        FileNotFoundError: If the directory does not exist.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")

    if config.backend == "unsloth":
        FastLanguageModel = _import_unsloth()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise BackendError("peft is required to load adapters with the 'hf' backend") from exc

    from transformers import AutoTokenizer

    base, _ = _load_hf(config)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, AutoTokenizer.from_pretrained(str(adapter_dir))


def save_adapter(model: Model, tokenizer: Tokenizer, directory: str | Path) -> Path:
    """Persist the adapter weights and the tokenizer.

    Args:
        model: The adapted model.
        tokenizer: The tokenizer used during training.
        directory: Destination directory, created if absent.

    Returns:
        The directory that was written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return directory


def resolve_precision(precision: Precision) -> tuple[bool, bool]:
    """Decide which reduced precision to train in.

    ``auto`` selects bfloat16 on devices that support it and float16 elsewhere.
    Both flags are false without a CUDA device, leaving training in float32.

    Args:
        precision: The requested precision.

    Returns:
        A pair of flags, ``(fp16, bf16)``.
    """
    if not torch.cuda.is_available():
        return False, False
    if precision == "fp16":
        return True, False
    if precision == "bf16":
        return False, True
    supports_bf16 = torch.cuda.is_bf16_supported()
    return not supports_bf16, supports_bf16


def trainable_parameters(model: Model) -> tuple[int, int]:
    """Count the trainable and total parameters of a model.

    Quantised weights are stored packed several to a byte, so the size of their
    tensor is smaller than the number of parameters it represents. Those are
    counted from the storage width rather than from the element count.

    Args:
        model: Any torch module.

    Returns:
        A pair, ``(trainable, total)``.
    """
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = _parameter_count(parameter)
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def _parameter_count(parameter: Any) -> int:
    """Return how many parameters a tensor holds, undoing any packing."""
    count: int = parameter.numel()
    if type(parameter).__name__ != "Params4bit":
        return count

    storage = getattr(parameter, "quant_storage", None)
    width = int(storage.itemsize) if storage is not None else int(parameter.element_size())
    # Two values share every byte of storage.
    return count * 2 * width


def _import_unsloth() -> Any:
    """Import the optimised backend, reporting a clear error when it is absent."""
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise BackendError(
            "the 'unsloth' backend is not installed; install the 'unsloth' extra "
            "or select model.backend='hf'"
        ) from exc
    return FastLanguageModel


def _load_unsloth(config: ModelConfig) -> tuple[Model, Tokenizer]:
    """Load through the optimised backend and apply its conversation template."""
    FastLanguageModel = _import_unsloth()
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        device_map="auto",
        revision=config.revision,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=config.chat_template)
    return model, tokenizer


def _load_hf(config: ModelConfig) -> tuple[Model, Tokenizer]:
    """Load through transformers, keeping the tokenizer's own conversation template."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    keywords: dict[str, Any] = {}
    if config.dtype is not None:
        keywords["dtype"] = getattr(torch, config.dtype)
    if config.load_in_4bit:
        keywords["quantization_config"] = _four_bit_config()
        keywords["device_map"] = "auto"

    if config.revision is None:
        model = AutoModelForCausalLM.from_pretrained(config.name, revision=None, **keywords)
        resolved_revision = getattr(model.config, "_commit_hash", None)
        tokenizer = AutoTokenizer.from_pretrained(config.name, revision=resolved_revision)
    else:
        # A pinned revision is loaded from its snapshot directory, model and
        # tokenizer alike, so that what runs is what resolve_snapshot identified.
        source = str(resolve_snapshot(config.name, config.revision).path)
        keywords["attn_implementation"] = "sdpa"
        if not config.load_in_4bit and torch.cuda.is_available():
            # The whole model on one device: "auto" may offload layers to the
            # CPU, and no device map at all loads everything there.
            keywords["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(source, **keywords)
        tokenizer = AutoTokenizer.from_pretrained(source)

    if getattr(tokenizer, "chat_template", None) is None:
        raise BackendError(
            f"tokenizer for {config.name!r} carries no conversation template; "
            "the 'hf' backend requires one"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _four_bit_config() -> Any:
    """Build the quantisation configuration used when loading in 4 bits."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise BackendError("4-bit loading requires bitsandbytes") from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
