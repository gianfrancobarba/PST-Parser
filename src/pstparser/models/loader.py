"""Loading of the base model, the tokenizer and the low-rank adapters.

Two backends are available. ``unsloth`` uses the optimised loader, which patches
transformers and TRL when it is imported; ``hf`` uses transformers together with
peft and imposes no such side effects. Every import of the optimised backend is
confined to this module so that the rest of the package, and the test suite, can
run without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pstparser.config import LoraConfig, ModelConfig
from pstparser.config.schema import Precision

Model = Any
Tokenizer = Any


class BackendError(RuntimeError):
    """Raised when a backend is unavailable or cannot honour the configuration."""


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

    keywords: dict[str, Any] = {"revision": config.revision}
    if config.dtype is not None:
        keywords["dtype"] = getattr(torch, config.dtype)
    if config.load_in_4bit:
        keywords["quantization_config"] = _four_bit_config()
        keywords["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.name, **keywords)
    tokenizer = AutoTokenizer.from_pretrained(config.name, revision=config.revision)

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
