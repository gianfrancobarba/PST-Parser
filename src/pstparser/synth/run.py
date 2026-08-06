"""Orchestration of a corpus expansion run."""

from __future__ import annotations

import json
from pathlib import Path

from pstparser.config import ExperimentConfig
from pstparser.repro.manifest import build_manifest, write_manifest
from pstparser.repro.seeding import seed_everything
from pstparser.synth.generate import (
    SynthesisResult,
    generate_prompts,
    load_seeds,
    summarise,
    write_annotation_file,
)
from pstparser.synth.providers import Provider, provider_from_env

ANNOTATION_FILE = "annotation.yaml"
SUMMARY_FILE = "synthesis_summary.json"


def run_synthesis(config: ExperimentConfig, provider: Provider | None = None) -> SynthesisResult:
    """Generate prompts for every paradigm and write the annotation sheet.

    Args:
        config: The resolved experiment configuration.
        provider: The service producing the completions. Built from the
            configuration and the environment when omitted.

    Returns:
        The generated prompts and the path of the sheet awaiting annotation.

    Raises:
        FileNotFoundError: If the seed file is missing.
        ProviderError: If the credential is absent, or every request fails.
        ValueError: If the seed file is malformed.
    """
    seeds = seed_everything(config.seed)
    synth = config.synth

    seed_prompts = load_seeds(synth.seeds_path)
    if provider is None:
        provider = provider_from_env(
            base_url=synth.provider.base_url,
            model=synth.provider.model,
            api_key_env=synth.provider.api_key_env,
            timeout=synth.provider.timeout,
            attempts=synth.provider.attempts,
        )

    prompts, rejected = generate_prompts(seed_prompts, provider, synth)

    sheet_path = write_annotation_file(prompts, Path(synth.output_dir) / ANNOTATION_FILE)

    result = SynthesisResult(
        prompts=prompts,
        requested=synth.per_paradigm * len(seed_prompts),
        rejected=rejected,
        sheet_path=sheet_path,
    )

    summary = summarise(result)
    summary_path = Path(synth.output_dir) / SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_manifest(
        build_manifest(
            config=config,
            seeds=seeds,
            stage="synth",
            inputs={"seeds": synth.seeds_path},
            extra={"synthesis": summary},
        ),
        synth.output_dir,
    )

    return result
