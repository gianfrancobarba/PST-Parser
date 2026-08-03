"""Typed schema for experiment configuration.

Every operational parameter of the pipeline is declared here. Configurations are
written as YAML and validated against these models before any expensive resource
(dataset, tokenizer, model weights) is touched, so that malformed experiments
fail immediately rather than minutes into a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pstparser.pst.alignment import DEFAULT_LEVELS, MatchLevel

Precision = Literal["auto", "fp16", "bf16"]
ModelBackend = Literal["unsloth", "hf"]


class _Base(BaseModel):
    """Common behaviour for every configuration section."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class QualityConfig(_Base):
    """Thresholds for the dataset integrity check.

    Attributes:
        min_coverage_ratio: Minimum ratio between the length of the text
            reconstructed from the annotated segments and the length of the
            source prompt, after whitespace removal. Records below this ratio
            are reported as issues.
        fail_on_issues: When true, a non-empty issue list aborts the data
            preparation step instead of only being reported.
    """

    min_coverage_ratio: float = Field(default=0.90, gt=0.0, le=1.0)
    fail_on_issues: bool = False


class SplitConfig(_Base):
    """Parameters of the train/evaluation partition.

    Attributes:
        eval_fraction: Fraction of records held out for evaluation.
        random_state: Seed controlling the shuffle applied before partitioning.
        output_dir: Directory where the resulting identifier files are written.
    """

    eval_fraction: float = Field(default=0.09, gt=0.0, lt=1.0)
    random_state: int = 42
    output_dir: Path = Path("data/splits")


class DataConfig(_Base):
    """Location and layout of the annotated corpus.

    The corpus is only read while preparing the data, so its presence is checked
    at that point rather than here. Scoring a prediction file therefore needs no
    access to the annotations.

    Attributes:
        source_path: Spreadsheet holding the manual annotations.
        sheet_name: Worksheet to read within the spreadsheet.
        prompt_column: Column containing the raw, unsegmented prompt.
        column_mapping: Mapping from dotted leaf path of the target schema to
            the spreadsheet column supplying its content.
        processed_dir: Directory where the serialised records are written.
        quality: Integrity check thresholds.
        split: Partitioning parameters.
    """

    source_path: Path
    sheet_name: str
    prompt_column: str = "PROMPT"
    column_mapping: dict[str, str]
    processed_dir: Path = Path("data/processed")
    quality: QualityConfig = QualityConfig()
    split: SplitConfig = SplitConfig()

    @field_validator("column_mapping")
    @classmethod
    def _mapping_must_not_be_empty(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("column_mapping must declare at least one leaf")
        return value


class ModelConfig(_Base):
    """Base model selection and loading options.

    Attributes:
        backend: Loading strategy. ``unsloth`` uses the optimised loader,
            ``hf`` uses transformers together with peft.
        name: Model identifier on the Hugging Face Hub, or a local directory.
        revision: Commit hash to pin. When omitted the default branch is used.
        max_seq_length: Maximum number of tokens per training example.
        load_in_4bit: Whether to load the base weights quantised to 4 bits.
        dtype: Explicit compute dtype. When omitted the backend selects one.
        chat_template: Name of the conversation template applied to the
            tokenizer.
    """

    backend: ModelBackend = "unsloth"
    name: str
    revision: str | None = None
    max_seq_length: int = Field(default=4096, gt=0)
    load_in_4bit: bool = True
    dtype: str | None = None
    chat_template: str = "llama-3.1"


class LoraConfig(_Base):
    """Low-rank adapter parameters.

    Attributes:
        r: Rank of the decomposition.
        alpha: Scaling factor applied to the adapter output.
        dropout: Dropout probability on the adapter input.
        bias: Which bias terms remain trainable.
        target_modules: Names of the linear projections to adapt.
        use_gradient_checkpointing: Checkpointing strategy passed to the
            backend. Accepts a boolean or a backend-specific identifier.
        random_state: Seed for adapter weight initialisation.
    """

    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: bool | str = "unsloth"
    random_state: int = 3407

    @field_validator("target_modules")
    @classmethod
    def _modules_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("target_modules must list at least one projection")
        return value


class TrainingConfig(_Base):
    """Supervised fine-tuning hyperparameters.

    Attributes:
        output_dir: Root directory for run artefacts.
        per_device_train_batch_size: Examples per optimisation step, per device.
        per_device_eval_batch_size: Examples per evaluation step, per device.
        gradient_accumulation_steps: Number of micro-batches accumulated before
            an optimiser update.
        learning_rate: Peak learning rate.
        optim: Optimiser identifier understood by the trainer.
        max_steps: Total number of optimisation steps.
        warmup_steps: Steps spent linearly increasing the learning rate.
        seed: Seed governing data ordering and stochastic layers.
        eval_strategy: When to run evaluation during training.
        eval_steps: Interval between evaluations when ``eval_strategy`` is
            ``steps``.
        logging_steps: Interval between training metric emissions.
        completion_only_loss: When true, the loss ignores the prompt tokens and
            is computed on the assistant turn alone.
        prediction_loss_only: When true, evaluation reports the loss alone and
            does not retain the predicted distributions. Those tensors are the
            size of the vocabulary times the sequence length, so keeping them is
            what makes evaluation more demanding than training.
        load_best_model_at_end: When true, the checkpoint with the best
            validation loss is restored once training finishes.
        precision: Compute precision. ``auto`` selects bf16 where the device
            supports it and fp16 otherwise.
    """

    output_dir: Path = Path("outputs")
    per_device_train_batch_size: int = Field(default=2, gt=0)
    per_device_eval_batch_size: int = Field(default=2, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    optim: str = "adamw_8bit"
    max_steps: int = Field(default=300, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    seed: int = 3407
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = Field(default=50, gt=0)
    logging_steps: int = Field(default=10, gt=0)
    completion_only_loss: bool = False
    prediction_loss_only: bool = False
    load_best_model_at_end: bool = False
    precision: Precision = "auto"

    @model_validator(mode="after")
    def _warmup_must_fit_schedule(self) -> TrainingConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be smaller than "
                f"max_steps ({self.max_steps})"
            )
        return self

    @model_validator(mode="after")
    def _best_checkpoint_requires_evaluation(self) -> TrainingConfig:
        if self.load_best_model_at_end and self.eval_strategy == "no":
            raise ValueError("load_best_model_at_end requires eval_strategy other than 'no'")
        return self

    @property
    def effective_batch_size(self) -> int:
        """Number of examples contributing to a single optimiser update."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class InferenceConfig(_Base):
    """Decoding parameters used when producing predictions.

    Attributes:
        max_new_tokens: Upper bound on generated tokens per example.
        do_sample: Whether to sample instead of decoding greedily.
        temperature: Sampling temperature. Ignored when ``do_sample`` is false.
        top_p: Nucleus sampling threshold. Ignored when ``do_sample`` is false.
        structured_output: When true, decoding is constrained to the target
            JSON schema.
        batch_size: Number of prompts submitted to the model at once.
    """

    max_new_tokens: int = Field(default=4096, gt=0)
    do_sample: bool = False
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    structured_output: bool = False
    batch_size: int = Field(default=1, gt=0)


class EvaluationConfig(_Base):
    """Scoring options.

    Attributes:
        ted_failure_penalty: Distance assigned to a pair of trees that cannot be
            compared because the prediction is not parseable.
        confusion_top_k: Number of most frequent label confusions reported.
        alignment_levels: Normalisations attempted when locating a phrase in its
            prompt, from the most conservative. Widening the scale locates more
            phrases and makes the reconstruction rate less strict, so it belongs
            to the experiment rather than to the command line.
    """

    ted_failure_penalty: float = Field(default=100.0, ge=0.0)
    confusion_top_k: int = Field(default=5, gt=0)
    alignment_levels: tuple[MatchLevel, ...] = DEFAULT_LEVELS

    @field_validator("alignment_levels")
    @classmethod
    def _levels_must_not_be_empty(cls, value: tuple[MatchLevel, ...]) -> tuple[MatchLevel, ...]:
        if not value:
            raise ValueError("alignment_levels must list at least one normalisation")
        return value


class ProviderConfig(_Base):
    """Remote service used to expand the corpus.

    Attributes:
        base_url: Root of a chat completions API compatible with the OpenAI wire
            format.
        model: Identifier of the model to call.
        api_key_env: Environment variable holding the credential. The value is
            never stored in a configuration file.
        timeout: Seconds to wait for a response.
        attempts: How many times a failing request is retried.
    """

    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "meta/llama-3.1-70b-instruct"
    api_key_env: str = "NVIDIA_API_KEY"
    timeout: float = Field(default=120.0, gt=0.0)
    attempts: int = Field(default=3, gt=0)


class SynthConfig(_Base):
    """Generation of additional prompts for the reasoning paradigms.

    Attributes:
        seeds_path: File holding the seed prompts, grouped by paradigm.
        output_dir: Directory receiving the annotation sheet.
        sheet_name: Name given to the worksheet, matching the corpus layout.
        per_paradigm: How many prompts to request for each paradigm.
        temperature: Sampling temperature. Higher values widen the variety at
            the cost of adherence to the paradigm.
        provider: The service producing the completions.
    """

    seeds_path: Path = Path("data/seeds/paradigms.yaml")
    output_dir: Path = Path("data/synthetic")
    sheet_name: str = "synthetic"
    per_paradigm: int = Field(default=50, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    provider: ProviderConfig = ProviderConfig()


class ExperimentConfig(_Base):
    """Full description of a single experiment.

    Attributes:
        name: Identifier used to label runs and their artefacts.
        seed: Global seed applied to the standard library, NumPy and torch.
        system_prompt_path: File holding the system message prepended to every
            training and inference example.
        data: Corpus location and preparation options.
        model: Base model selection.
        lora: Adapter parameters.
        training: Fine-tuning hyperparameters.
        inference: Decoding parameters.
        evaluation: Scoring options.
        synth: Options for expanding the corpus.
    """

    name: str = Field(min_length=1)
    seed: int = 42
    system_prompt_path: Path
    data: DataConfig
    model: ModelConfig
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    synth: SynthConfig = SynthConfig()

    @field_validator("system_prompt_path")
    @classmethod
    def _prompt_must_exist(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError(f"system prompt file not found: {value}")
        return value

    @model_validator(mode="after")
    def _sequence_budget_must_be_reachable(self) -> ExperimentConfig:
        if self.inference.max_new_tokens > self.model.max_seq_length:
            raise ValueError(
                f"inference.max_new_tokens ({self.inference.max_new_tokens}) exceeds "
                f"model.max_seq_length ({self.model.max_seq_length})"
            )
        return self
