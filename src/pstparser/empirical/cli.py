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


@app.command("calibrate-bigcodebench")
def calibrate_bigcodebench(
    parquet: Path,
    revision: Annotated[str, typer.Option(help="Pinned revision of the full dataset.")],
    hard_parquet: Annotated[Path, typer.Option(help="Parquet file of BigCodeBench-Hard.")],
    hard_revision: Annotated[str, typer.Option(help="Pinned revision of the Hard dataset.")],
    image: Annotated[str, typer.Option(help="Harness image, as name@sha256:<digest>.")],
    nltk_dir: Annotated[Path, typer.Option(help="Local nltk data, mounted read-only.")],
    directory: Annotated[Path, typer.Option(help="New directory for the calibration runs.")],
    out: Annotated[Path, typer.Option(help="New file for the frozen exclusions.")],
    split: Annotated[str, typer.Option(help="BigCodeBench split.")] = "v0.1.4",
    parallel: Annotated[int, typer.Option(help="Fixed number of harness workers.")] = 8,
    task_ids: Annotated[
        str | None, typer.Option(help="Comma-separated subset, for a smoke run only.")
    ] = None,
) -> None:
    """Apply exclusion rules E1 to E4 with the official harness, before any generation.

    The exclusions file is written only when the calibration covers every task
    and leaves no reference failure unexplained.
    """
    from pstparser.empirical import bigcodebench
    from pstparser.empirical.benchmarks import load_bigcodebench_hard_ids
    from pstparser.repro.manifest import (
        digest_directory,
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    if out.exists():
        raise FileExistsError(f"the exclusions file already exists: {out}")
    ensure_stage_outputs(directory, "empirical-calibrate-bcb", ["calibration.json", "data", "runs"])
    hard_ids = load_bigcodebench_hard_ids(hard_parquet, hard_revision, split)
    dataset = directory / "data" / f"BigCodeBench-{split}.jsonl"
    dataset_sha256 = bigcodebench.write_harness_dataset(parquet, revision, split, dataset)
    nltk_sha256 = digest_directory(nltk_dir)
    if nltk_sha256 is None:
        raise ValueError(f"missing nltk data directory: {nltk_dir}")
    config = bigcodebench.HarnessConfig(
        image=image,
        dataset_jsonl=dataset,
        dataset_sha256=dataset_sha256,
        nltk_dir=nltk_dir,
        nltk_sha256=nltk_sha256,
        parallel=parallel,
    )
    environment = {
        "harness_version": bigcodebench.harness_version(image),
        "docker": bigcodebench.docker_resources(),
    }
    report = bigcodebench.calibrate(
        config,
        hard_ids,
        directory / "runs",
        task_ids=task_ids.split(",") if task_ids else None,
    )
    payload = {
        "format_version": 1,
        "benchmark": "bigcodebench",
        "dataset_revision": revision,
        "split": split,
        "dataset_file_sha256": digest_file(parquet),
        "hard_revision": hard_revision,
        "hard_file_sha256": digest_file(hard_parquet),
        **report,
        **environment,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with (directory / "calibration.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    conclusive = report["scope"] == "all" and not report["unexplained"]
    if conclusive:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    write_manifest(
        {
            "stage": "empirical-calibrate-bcb",
            "conclusive": conclusive,
            "calibration_sha256": digest_file(directory / "calibration.json"),
            "exclusions_sha256": digest_file(out) if conclusive else None,
            "harness": config.as_dict(),
            **environment,
            "source": source_snapshot(),
        },
        directory,
    )
    typer.echo(
        f"{report['excluded_count']} excluded, {report['eligible']} eligible, "
        f"{report['eligible_hard']} eligible Hard"
    )
    if report["unexplained"]:
        raise ValueError(f"unexplained reference failures: {report['unexplained']}")


@app.command("merge-calibrations")
def merge_calibrations(
    calibrations: list[Path],
    out: Annotated[Path, typer.Option(help="New file for the frozen exclusions.")],
) -> None:
    """Freeze the exclusions as the union of complete BigCodeBench calibrations.

    A task is excluded when any calibration excludes it: a reference solution
    that fails in one run out of several is not a usable oracle.
    """
    from pstparser.empirical import bigcodebench
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    if len(set(calibrations)) != len(calibrations):
        raise ValueError("a calibration is listed twice")
    merged = bigcodebench.merge_calibrations([_read_document(path) for path in calibrations])
    digests = [digest_file(path) for path in calibrations]
    if len(set(digests)) != len(digests):
        raise ValueError("two calibration files are identical")
    ensure_stage_outputs(out.parent, "empirical-merge-calibrations", [out.name])
    _write_document(out, {**merged, "calibration_sha256": digests})
    write_manifest(
        {
            "stage": "empirical-merge-calibrations",
            "calibration_sha256": digests,
            "exclusions_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(
        f"{merged['excluded_count']} excluded, {merged['eligible']} eligible, "
        f"{merged['eligible_hard']} eligible Hard"
    )


def _write_document(path: Path, document: dict[str, Any]) -> None:
    """Write a frozen JSON document to a new file, with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _read_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return document


@app.command()
def sample(
    sources: Path,
    exclude: Annotated[Path, typer.Option(help="Frozen exclusions of the calibration.")],
    seed: Annotated[int, typer.Option(help="Sampling seed, fixed before seeing any data.")],
    permutation_out: Annotated[Path, typer.Option(help="New file for the frozen permutation.")],
    out: Annotated[Path, typer.Option(help="New file for the final sample.")],
    n: Annotated[int, typer.Option(help="Size of the final sample, a prefix.")] = 800,
    strata: Annotated[str, typer.Option(help="Named stratification rule.")] = (
        "bigcodebench-exceptions-doctest-v1"
    ),
    hard_parquet: Annotated[Path | None, typer.Option(help="Parquet of the Hard subset.")] = None,
    hard_revision: Annotated[str | None, typer.Option(help="Revision of the Hard subset.")] = None,
    split: Annotated[str, typer.Option(help="BigCodeBench split.")] = "v0.1.4",
) -> None:
    """Freeze the stratified permutation of the eligible tasks and its final prefix."""
    from pstparser.empirical import sampling
    from pstparser.empirical.benchmarks import load_bigcodebench_hard_ids, read_sources
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    if strata not in sampling.STRATA:
        raise ValueError(f"unknown strata rule {strata!r}; choose one of {sorted(sampling.STRATA)}")
    if (hard_parquet is None) != (hard_revision is None):
        raise ValueError("give both --hard-parquet and --hard-revision, or neither")
    records = read_sources(sources)
    if not records:
        raise ValueError("empty sources")
    if records[0].benchmark == "bigcodebench" and hard_parquet is None:
        raise ValueError("BigCodeBench needs its Hard subset: the pilot is drawn outside it")
    dataset_sha256 = records[0].dataset_file_sha256
    calibration = _read_document(exclude)
    # Exclusions must come from a complete, conclusive calibration of this very file.
    if (
        calibration.get("format_version") != 1
        or calibration.get("scope") != "all"
        or calibration.get("unexplained") != []
    ):
        raise ValueError("the exclusions are not a complete, conclusive calibration")
    if calibration["dataset_file_sha256"] != dataset_sha256:
        raise ValueError("the exclusions were calibrated on another dataset file")
    if calibration["tasks"] != len(records):
        raise ValueError("the exclusions do not cover every source")
    excluded = {row["task_id"]: row["reason"] for row in calibration["excluded"]}
    hard: frozenset[str] = frozenset()
    if hard_parquet is not None and hard_revision is not None:
        hard = load_bigcodebench_hard_ids(hard_parquet, hard_revision, split)
        if not hard <= {record.task_id for record in records}:
            raise ValueError("a Hard task is not among the sources")
    permutation = sampling.stratified_permutation(
        records,
        dataset_sha256=dataset_sha256,
        seed=seed,
        strata=sampling.STRATA[strata],
        exclude=excluded,
    )
    final = sampling.sample(permutation, n)
    if permutation_out.exists():
        raise FileExistsError(f"the permutation already exists: {permutation_out}")
    ensure_stage_outputs(out.parent, "empirical-sample", [out.name])
    _write_document(
        permutation_out,
        {
            **sampling.permutation_document(
                permutation,
                benchmark=records[0].benchmark,
                dataset_revision=records[0].dataset_revision,
                strata_rule=strata,
                hard_ids=hard,
            ),
            "sources_sha256": digest_file(sources),
            "exclusions_sha256": digest_file(exclude),
        },
    )
    _write_document(
        out,
        sampling.sample_document(
            "final-sample",
            final,
            permutation,
            hard,
            benchmark=records[0].benchmark,
            permutation_sha256=digest_file(permutation_out),
        ),
    )
    write_manifest(
        {
            "stage": "empirical-sample",
            "seed": seed,
            "n": n,
            "strata_rule": strata,
            "sources_sha256": digest_file(sources),
            "exclusions_sha256": digest_file(exclude),
            "permutation_sha256": digest_file(permutation_out),
            "sample_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(permutation.order)} eligible; final sample of {n} in {out}")


@app.command("sample-pilot")
def sample_pilot(
    permutation: Path,
    final: Path,
    seed: Annotated[int, typer.Option(help="Seed of the pilot draw.")],
    out: Annotated[Path, typer.Option(help="New file for the pilot pool.")],
    full_out: Annotated[Path, typer.Option(help="New file for every eligible task but the pilot.")],
    n: Annotated[int, typer.Option(help="Number of pilot tasks.")] = 30,
) -> None:
    """Draw the pilot pool outside the final sample and outside Hard.

    The final sample must already be frozen: the pilot is what it left out.
    """
    from pstparser.empirical import sampling
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    frozen, hard = sampling.read_permutation_document(_read_document(permutation))
    final_document = _read_document(final)
    if final_document.get("permutation_sha256") != digest_file(permutation):
        raise ValueError("the final sample was not drawn from this permutation file")
    final_sample = sampling.read_sample_document(final_document, "final-sample")
    pilot = sampling.pilot_pool(frozen, final_sample, n, seed=seed, hard_ids=hard)
    full = sampling.full_minus(frozen, pilot)
    if full_out.exists():
        raise FileExistsError(f"the full set already exists: {full_out}")
    ensure_stage_outputs(out.parent, "empirical-sample-pilot", [out.name])
    provenance = {
        "benchmark": final_document["benchmark"],
        "permutation_sha256": digest_file(permutation),
        "final_sha256": digest_file(final),
    }
    _write_document(
        out,
        sampling.sample_document(
            "pilot-sample",
            pilot,
            frozen,
            hard,
            purpose="pilot",
            hard_excluded=len(hard),
            **provenance,
        ),
    )
    _write_document(
        full_out,
        sampling.sample_document(
            "full-minus-pilot", full, frozen, hard, pilot_sha256=digest_file(out), **provenance
        ),
    )
    write_manifest(
        {
            "stage": "empirical-sample-pilot",
            "seed": seed,
            "n": n,
            **provenance,
            "pilot_sha256": digest_file(out),
            "full_sha256": digest_file(full_out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"pilot pool of {n} in {out}; {len(full.task_ids)} tasks in {full_out}")


@app.command("sample-families")
def sample_families(
    sources: Path,
    families: Annotated[int, typer.Option(help="Number of whole families to draw.")],
    seed: Annotated[int, typer.Option(help="Seed of the draw.")],
    disjoint_from: Annotated[Path, typer.Option(help="Sources no drawn family may appear in.")],
    out: Annotated[Path, typer.Option(help="New file for the drawn families.")],
    family_size: Annotated[
        int | None, typer.Option(help="Draw only among families of this size.")
    ] = None,
) -> None:
    """Draw whole families as pilot material, disjoint by family from another set."""
    from collections import Counter

    from pstparser.empirical import sampling
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    records = read_sources(sources)
    others = read_sources(disjoint_from)
    if not records:
        raise ValueError("empty sources")
    dataset_sha256 = records[0].dataset_file_sha256
    sizes = Counter(record.family_id for record in records)
    drawn = sampling.family_sample(
        records,
        families=families,
        seed=seed,
        dataset_sha256=dataset_sha256,
        family_size=family_size,
    )
    split = sampling.shared_families(drawn, others)
    if split or {record.task_id for record in drawn} & {record.task_id for record in others}:
        raise ValueError(f"families shared with the other set: {sorted(split)}")
    ensure_stage_outputs(out.parent, "empirical-sample-families", [out.name])
    _write_document(
        out,
        {
            "format_version": 1,
            "kind": "family-sample",
            "benchmark": records[0].benchmark,
            "dataset_revision": records[0].dataset_revision,
            "dataset_file_sha256": dataset_sha256,
            "text_field": records[0].text_field,
            "method": sampling.ORDER_METHOD,
            "purpose": "pilot-families",
            "seed": seed,
            "family_size": family_size,
            "candidate_families": sum(
                1 for size in sizes.values() if family_size is None or size == family_size
            ),
            "families": list(dict.fromkeys(record.family_id for record in drawn)),
            "n": len(drawn),
            "sources_sha256": digest_file(sources),
            "disjoint_from_sha256": digest_file(disjoint_from),
            "tasks": [
                {
                    "task_id": record.task_id,
                    "family_id": record.family_id,
                    "source_sha256": record.source_sha256,
                }
                for record in drawn
            ],
        },
    )
    write_manifest(
        {
            "stage": "empirical-sample-families",
            "seed": seed,
            "families": families,
            "family_size": family_size,
            "sources_sha256": digest_file(sources),
            "disjoint_from_sha256": digest_file(disjoint_from),
            "sample_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(drawn)} prompts in {families} families in {out}")


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
