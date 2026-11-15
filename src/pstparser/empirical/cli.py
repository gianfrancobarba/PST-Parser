"""Explicit artifact-based commands; no benchmark or experiment is selected implicitly."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer

from pstparser.empirical.analysis import AnalysisDesign, TaskSpec, analyze_paired, read_outcomes
from pstparser.empirical.contracts import CanonicalPrompt, PromptVariant, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.messages import chat_messages, validate_user_template
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


@app.command("import-benchmark")
def import_benchmark(
    benchmark: str,
    source: Path,
    revision: Annotated[str, typer.Option(help="Pinned revision of the dataset repository.")],
    out: Annotated[Path, typer.Option(help="New JSON Lines file for the canonical sources.")],
    split: Annotated[str, typer.Option(help="BigCodeBench split.")] = "v0.1.4",
    selection: Annotated[str, typer.Option(help="LLMSecEval selection.")] = "S2",
    id_map: Annotated[
        Path | None, typer.Option(help="LLMSecEval: new file mapping internal IDs to records.")
    ] = None,
) -> None:
    """Import a pinned benchmark file into canonical sources, without oracle fields."""
    from collections import Counter

    from pstparser.empirical import benchmarks
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    if benchmark not in {benchmarks.BIGCODEBENCH, benchmarks.LLMSECEVAL}:
        raise ValueError("benchmark must be bigcodebench or llmseceval")
    if id_map is not None and benchmark != benchmarks.LLMSECEVAL:
        raise ValueError("--id-map applies to llmseceval only")
    # Everything is loaded and checked before the first file is created.
    details: dict[str, Any]
    mapping: list[dict[str, Any]] | None = None
    if benchmark == benchmarks.BIGCODEBENCH:
        records = benchmarks.load_bigcodebench(source, revision, split)
        details = {"split": split}
    else:
        records = benchmarks.load_llmseceval(source, revision, selection=selection)
        details = {
            "selection": selection,
            "families": len({record.family_id for record in records}),
        }
        if id_map is not None:
            mapping = benchmarks.llmseceval_id_map(source, revision)
    if not records:
        raise ValueError("empty import")
    ensure_stage_outputs(out.parent, "empirical-import", [out.name])
    if id_map is not None and id_map.exists():
        raise FileExistsError(f"the identifier map already exists: {id_map}")
    benchmarks.write_sources(records, out)
    if id_map is not None:
        id_map.parent.mkdir(parents=True, exist_ok=True)
        with id_map.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "benchmark": benchmark,
                    "dataset_revision": revision,
                    "dataset_file_sha256": records[0].dataset_file_sha256,
                    "records": mapping,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")
    applied = Counter(transform for record in records for transform in record.transforms)
    write_manifest(
        {
            "stage": "empirical-import",
            "benchmark": benchmark,
            "dataset_revision": revision,
            "dataset_file_sha256": records[0].dataset_file_sha256,
            **details,
            "text_field": records[0].text_field,
            "records": len(records),
            "transforms": [
                {
                    "kind": transform.kind,
                    "old": transform.old,
                    "new": transform.new,
                    "count": transform.count,
                    "records": count,
                }
                for transform, count in applied.items()
            ],
            "sources_sha256": digest_file(out),
            "id_map_sha256": digest_file(id_map) if id_map is not None else None,
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"Imported {len(records)} sources into {out}")


@app.command()
def generate(variants: Path, configuration: Path, directory: Path) -> None:
    """Generate/resume only an explicitly configured, revision-pinned experiment."""
    # Imports remain local: rendering and analysis do not load torch/models.
    from pstparser.config import InferenceConfig, ModelConfig
    from pstparser.inference.generate import (
        decoding_parameters,
        generate_messages,
        validate_token_budget,
    )
    from pstparser.models.loader import (
        load_untuned,
        resolve_snapshot,
        runtime_provenance,
        verify_chat_template,
    )
    from pstparser.repro.manifest import digest_file, digest_text, environment
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
    # A configuration without format_version is version 1 and keeps its closed
    # key set; the keys added since are admitted only by declaring version 2.
    versioned = "format_version" in config
    if versioned:
        if config["format_version"] != 2:
            raise ValueError("format_version must be 2 when present")
        allowed |= {"format_version", "user_template", "conditions"}
    if set(config) != allowed:
        raise ValueError(f"configuration requires exactly: {sorted(allowed)}")
    user_template: str | None = None
    if versioned:
        user_template = config["user_template"]
        if user_template is not None:
            if not isinstance(user_template, str):
                raise ValueError("user_template must be a string or null")
            validate_user_template(user_template)
        declared = config["conditions"]
        if (
            not isinstance(declared, list)
            or len(declared) < 2
            or len(set(declared)) != len(declared)
            or not all(isinstance(name, str) and name for name in declared)
        ):
            raise ValueError("conditions must list at least two distinct condition names")
        if not {"backend", "chat_template"} <= set(config["model"]):
            raise ValueError("model.backend and model.chat_template must be explicit")
        if config["model"]["backend"] == "hf" and config["model"]["chat_template"] != "native":
            raise ValueError("the hf backend keeps the model's own template: use 'native'")
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
    if versioned and {variant.condition for variant in rendered} != set(config["conditions"]):
        raise ValueError("the variants do not hold exactly the configured conditions")
    requests = request_plan(rendered, config["replicate_ids"], config["seed"])
    seed_everything(config["seed"])
    final = config["purpose"] == "final"
    # Only the transformers backend loads the repository it is asked for; the
    # optimised one substitutes its own, so its revision cannot be verified.
    snapshot = (
        resolve_snapshot(model_config.name, model_config.revision or "", hash_weights=final)
        if model_config.backend == "hf"
        else None
    )
    if final and (snapshot is None or snapshot.sha != model_config.revision):
        raise ValueError("final runs require the model snapshot of the requested revision")
    model, tokenizer = load_untuned(model_config)
    runtime = runtime_provenance(model, tokenizer, model_config, snapshot)
    if final and not runtime["revision_verified"]:
        raise ValueError("final runs require a verified model revision")
    if final and runtime["chat_template_matches_official"] is not True:
        raise ValueError("final runs require the published conversation template")
    # Check BOTH variants for ALL tasks before generating the first condition.
    template_checked: set[str] = set()
    for variant in rendered:
        messages = chat_messages(config["system_prompt"], user_template, variant.text)
        if snapshot is not None and variant.condition not in template_checked:
            verify_chat_template(tokenizer, snapshot, messages)
            template_checked.add(variant.condition)
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
        "user_template_sha256": digest_text(user_template) if user_template is not None else None,
        "decoding": decoding_parameters(inference),
        "runtime": runtime,
        "environment": environment(),
    }

    def complete(request: Any) -> dict[str, Any]:
        seed_everything(request["seed"])
        messages = chat_messages(config["system_prompt"], user_template, request["text"])
        result = generate_messages(
            model,
            tokenizer,
            messages,
            inference,
            max_seq_length=model_config.max_seq_length,
        )
        return {
            "user_message_sha256": digest_text(messages[1]["content"]),
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
