"""Explicit artifact-based commands; no benchmark or experiment is selected implicitly."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer

from pstparser.empirical.analysis import AnalysisDesign, TaskSpec, analyze_paired, read_outcomes
from pstparser.empirical.contracts import CanonicalPrompt, PromptVariant, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.rendering import render_prompt_variants

app = typer.Typer(
    help="Prepared empirical workflow; real benchmark choices require protocol freeze."
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@app.command()
def render(source: Path, destination: Path) -> None:
    """Render reviewed canonical JSONL annotations into paired T/H-gold JSONL."""
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _rows(source):
        canonical = CanonicalPrompt(
            task_id=row["task_id"],
            text=row["text"],
            annotation_version=row["annotation_version"],
            annotations=tuple(SpanAnnotation(**span) for span in row["annotations"]),
        )
        if canonical.task_id in seen:
            raise ValueError("duplicate canonical task")
        seen.add(canonical.task_id)
        variants.extend(v.as_dict() for v in render_prompt_variants(canonical))
    if not variants:
        raise ValueError("empty corpus")
    _write_rows(destination, variants)
    typer.echo(f"Rendered {len(variants)} variants into {destination}")


@app.command()
def generate(variants: Path, configuration: Path, directory: Path) -> None:
    """Generate/resume only an explicitly configured, revision-pinned experiment."""
    # Imports remain local: rendering and analysis do not load torch/models.
    from pstparser.config import InferenceConfig, ModelConfig
    from pstparser.inference.generate import generate_messages, validate_token_budget
    from pstparser.models.loader import load_untuned, runtime_provenance
    from pstparser.repro.manifest import digest_file, digest_text
    from pstparser.repro.seeding import seed_everything

    config = json.loads(configuration.read_text(encoding="utf-8"))
    allowed = {
        "model",
        "inference",
        "seed",
        "replicate_ids",
        "system_prompt",
        "protocol_version",
        "dataset_revision",
        "purpose",
        "review_record",
    }
    if set(config) != allowed:
        raise ValueError(f"configuration requires exactly: {sorted(allowed)}")
    if config["purpose"] not in {"pilot", "final"}:
        raise ValueError("purpose must be pilot or final")
    for name in ("protocol_version", "dataset_revision", "system_prompt", "review_record"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"missing {name}")
    review = configuration.parent / config["review_record"]
    if not review.is_file():
        raise ValueError("an explicit review/freeze record is required")
    model_config = ModelConfig(**config["model"])
    inference = InferenceConfig(**config["inference"])
    if not re.fullmatch(r"[0-9a-f]{40}", model_config.revision or ""):
        raise ValueError("provide an immutable 40-character model revision")
    if inference.structured_output or inference.batch_size != 1:
        raise ValueError("code generation uses unconstrained output, batch_size=1")
    if not inference.do_sample and len(config["replicate_ids"]) != 1:
        raise ValueError("greedy repetitions are not independent samples")
    rendered = [PromptVariant(**row) for row in _rows(variants)]
    requests = request_plan(rendered, config["replicate_ids"], config["seed"])
    seed_everything(config["seed"])
    model, tokenizer = load_untuned(model_config)
    # Check BOTH variants for ALL tasks before generating the first condition.
    for variant in rendered:
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": variant.text},
        ]
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
        )
        validate_token_budget(
            model,
            tokenizer,
            len(tokens),
            inference.max_new_tokens,
            model_config.max_seq_length,
        )
    frozen = {
        **config,
        "review_record_sha256": digest_file(review),
        "variant_file_sha256": digest_file(variants),
        "system_prompt_sha256": digest_text(config["system_prompt"]),
        "runtime": runtime_provenance(model, tokenizer, model_config),
    }

    def complete(request: Any) -> dict[str, Any]:
        seed_everything(request["seed"])
        result = generate_messages(
            model,
            tokenizer,
            [
                {"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": request["text"]},
            ],
            inference,
            max_seq_length=model_config.max_seq_length,
        )
        return {
            "raw_output": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "finish_reason": result.finish_reason,
            "elapsed_seconds": result.elapsed_seconds,
        }

    records = run_journal(requests, frozen, directory, complete)
    typer.echo(f"{len(records)} recorded completions; {directory}")


@app.command()
def analyze(design: Path, outcomes: Path, destination: Path) -> None:
    """Analyse the explicit paired design, rejecting absent/duplicate/mixed rows."""
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    ensure_stage_outputs(destination.parent, "empirical-analysis", [destination.name])
    payload = json.loads(design.read_text(encoding="utf-8"))
    payload["tasks"] = tuple(TaskSpec(**task) for task in payload["tasks"])
    for key in ("replicate_ids", "conditions"):
        if key in payload:
            payload[key] = tuple(payload[key])
    report = analyze_paired(AnalysisDesign(**payload), read_outcomes(outcomes))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_manifest(
        {
            "stage": "empirical-analysis",
            "analysis_version": "paired-v1",
            "design_sha256": digest_file(design),
            "outcomes_sha256": digest_file(outcomes),
            "report_sha256": digest_file(destination),
            "source": source_snapshot(),
        },
        destination.parent,
    )
    typer.echo(str(destination))


@app.command()
def check(
    generations: Path,
    oracles: Path,
    directory: Path,
    image: str,
    analysis_id: str,
) -> None:
    """Calibrate supplied functional oracles, then evaluate in pinned local Docker.

    This does not import a benchmark or choose its test suite. Oracles must be
    trusted/versioned JSONL records matching FunctionalOracle, with both controls.
    """
    from dataclasses import asdict

    from pstparser.empirical.analysis import write_outcomes
    from pstparser.empirical.sandbox import SandboxConfig, run_python_check
    from pstparser.empirical.workflow import (
        FunctionalOracle,
        calibrate_oracles,
        evaluate_functional,
    )
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    ensure_stage_outputs(
        directory,
        "empirical-check",
        [
            "outcomes.jsonl",
            "details.jsonl",
            "calibration.json",
        ],
    )
    config = SandboxConfig(image)
    trusted = [FunctionalOracle(**row) for row in _rows(oracles)]
    generated = _rows(generations)
    if not generated or {r["task_id"] for r in generated} != {o.task_id for o in trusted}:
        raise ValueError("generation and oracle task sets must be identical and non-empty")

    def execute(code: str, harness: str) -> Any:
        return run_python_check(code, harness, config)

    calibration = calibrate_oracles(trusted, execute)
    with (directory / "calibration.json").open("x", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
    calibrated = all(row["calibrated"] for row in calibration.values())
    if calibrated:
        outcomes, details = evaluate_functional(
            generated, trusted, analysis_id, execute, calibration
        )
        write_outcomes(outcomes, directory / "outcomes.jsonl")
        _write_rows(directory / "details.jsonl", details)
    write_manifest(
        {
            "stage": "empirical-check",
            "analysis_id": analysis_id,
            "calibrated": calibrated,
            "sandbox": asdict(config),
            "retry_count": 0,
            "generations_sha256": digest_file(generations),
            "oracles_sha256": digest_file(oracles),
            "outcomes_sha256": digest_file(directory / "outcomes.jsonl"),
            "source": source_snapshot(),
        },
        directory,
    )
    if not calibrated:
        raise ValueError("oracle calibration failed; no candidate outcomes produced")
    typer.echo(str(directory / "outcomes.jsonl"))
