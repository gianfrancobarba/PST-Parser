"""Composition, override and validation of experiment configurations."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from pstparser.config import ConfigError, ExperimentConfig, dump_resolved, load_experiment


def test_composition_merges_every_base(config_dir: Path) -> None:
    config = load_experiment(config_dir / "valid.yaml", root=config_dir)

    # Values contributed by the data fragment.
    assert config.data.sheet_name == "corpus"
    assert len(config.data.column_mapping) == 9
    # Values contributed by the model fragment.
    assert config.model.backend == "hf"
    assert config.model.max_seq_length == 512
    # Values declared by the experiment itself.
    assert config.name == "fixture"
    assert config.seed == 7


def test_experiment_values_win_over_bases(config_dir: Path) -> None:
    config = load_experiment(config_dir / "valid.yaml", root=config_dir)

    assert config.lora.r == 2
    assert config.lora.alpha == 4


def test_defaults_apply_to_unspecified_sections(config_dir: Path) -> None:
    config = load_experiment(config_dir / "valid.yaml", root=config_dir)

    assert config.training.learning_rate == pytest.approx(2e-4)
    assert config.training.optim == "adamw_8bit"
    assert config.inference.do_sample is False


@pytest.mark.parametrize(
    ("override", "read"),
    [
        ("training.max_steps=50", lambda c: c.training.max_steps),
        ("lora.r=8", lambda c: c.lora.r),
        ("inference.do_sample=true", lambda c: c.inference.do_sample),
        ("model.name=other/model", lambda c: c.model.name),
    ],
)
def test_override_reaches_nested_keys(
    config_dir: Path,
    override: str,
    read: object,
) -> None:
    config = load_experiment(config_dir / "valid.yaml", overrides=[override], root=config_dir)

    expected = yaml.safe_load(override.split("=", 1)[1])
    assert read(config) == expected  # type: ignore[operator]


def test_override_rejects_missing_assignment(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match=re.escape("dotted.key=value")):
        load_experiment(config_dir / "valid.yaml", overrides=["lora.r"], root=config_dir)


def test_invalid_configuration_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        load_experiment(config_dir / "invalid.yaml", root=config_dir)


def test_circular_extends_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="circular"):
        load_experiment(config_dir / "circular_a.yaml", root=config_dir)


def test_missing_file_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_experiment(config_dir / "does_not_exist.yaml", root=config_dir)


def test_missing_corpus_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ValueError, match="corpus not found"):
        load_experiment(
            config_dir / "valid.yaml",
            overrides=["data.source_path=tests/assets/absent.xlsx"],
            root=config_dir,
        )


def test_unknown_key_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ValueError, match="typoed_key"):
        load_experiment(config_dir / "valid.yaml", overrides=["lora.typoed_key=1"], root=config_dir)


def test_effective_batch_size_multiplies_accumulation(config_dir: Path) -> None:
    config = load_experiment(
        config_dir / "valid.yaml",
        overrides=[
            "training.per_device_train_batch_size=4",
            "training.gradient_accumulation_steps=8",
        ],
        root=config_dir,
    )

    assert config.training.effective_batch_size == 32


def test_resolved_configuration_round_trips(config_dir: Path, tmp_path: Path) -> None:
    config = load_experiment(config_dir / "valid.yaml", root=config_dir)

    written = dump_resolved(config, tmp_path / "nested" / "resolved.yaml")
    reloaded = ExperimentConfig.model_validate(yaml.safe_load(written.read_text(encoding="utf-8")))

    assert reloaded == config


def test_baseline_experiment_is_valid() -> None:
    config = load_experiment("configs/experiments/baseline.yaml")

    assert config.name == "baseline"
    assert config.model.backend == "unsloth"
    assert config.training.effective_batch_size == 8
    assert len(config.data.column_mapping) == 9
