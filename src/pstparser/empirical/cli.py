"""Explicit artifact-based commands; no benchmark or experiment is selected implicitly."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from pstparser.empirical.analysis import AnalysisDesign, TaskSpec, analyze_paired, read_outcomes
from pstparser.empirical.contracts import CanonicalPrompt, PromptVariant, SpanAnnotation
from pstparser.empirical.journal import request_plan, run_journal
from pstparser.empirical.messages import chat_messages, validate_user_template
from pstparser.empirical.rendering import (
    render_envelope_variant,
    render_flat_variant,
    render_prompt_variants,
)

app = typer.Typer(
    help="Prepared empirical workflow; real benchmark choices require protocol freeze."
)


def _purpose(purpose: str, freeze: Path | None) -> dict[str, Any]:
    """Purpose of a downstream command and the hash of the verified freeze it ran under.

    A ``final`` purpose requires a protocol freeze that verifies against the
    repository (R-GEN-01); an ``exploratory`` one records ``None`` when no
    freeze is given.

    Raises:
        ValueError: If the purpose is unknown, a final purpose has no freeze,
            or the freeze does not verify.
    """
    if purpose not in {"exploratory", "final"}:
        raise ValueError("purpose must be exploratory or final")
    if freeze is None:
        if purpose == "final":
            raise ValueError("a final purpose requires --freeze with a verified protocol freeze")
        return {"purpose": purpose, "freeze_sha256": None}
    from pstparser.empirical.freeze import verify_freeze
    from pstparser.repro.manifest import digest_file

    mismatches = verify_freeze(freeze)
    if mismatches:
        shown = ", ".join(f"{m.kind} {m.path}" for m in mismatches[:5])
        raise ValueError(f"the protocol freeze does not verify: {shown}")
    return {"purpose": purpose, "freeze_sha256": digest_file(freeze)}


_PURPOSE_HELP = "exploratory, or final: final requires --freeze (a verified protocol freeze)."
_FREEZE_HELP = "protocol-freeze.json the command runs under; verified before anything is read."
#: S_gold, the tasks with reviewed gold, relative to protocol/samples: the 600-task
#: prefix of the frozen permutation (plan of 18/11/2026, P-02).
S_GOLD_FILE = "prefix-600/bcb-prefix-600.json"
#: Prespecified fallback, the nested 400-task prefix: chosen only for time (pilot
#: stopwatch above 3.0 min per prompt, or gold review late at CP4), never for
#: outcomes; switching to it is recorded before any outcome is seen.
S_GOLD_FALLBACK_FILE = "prefix-400/bcb-prefix-400.json"


def _rows(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines split on the line feed only, skipping blank lines."""
    text = path.read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def _with_prefill(
    rows: list[dict[str, Any]], generations: Path
) -> tuple[list[dict[str, Any]], str | None]:
    """Put the S-PREFILL prefill back in front of every output (Q-25).

    The journal records only the continuation; the prefill is read from the
    configuration in the manifest next to the journal and prepended, as the
    start of the assistant turn, before extraction and evaluation.
    """
    manifest = generations.parent / "empirical-generate_manifest.json"
    if not manifest.is_file():
        return rows, None
    configuration = json.loads(manifest.read_text(encoding="utf-8")).get("configuration", {})
    prefill = configuration.get("assistant_prefill")
    if prefill is None:
        return rows, None
    return [{**row, "raw_output": prefill + row["raw_output"]} for row in rows], prefill


def _json_lines(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines split on the line feed only.

    A model output may hold other Unicode line separators, which JSON leaves
    unescaped and ``str.splitlines`` would cut.
    """
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line]


def _require_complete(rows: list[dict[str, Any]]) -> None:
    """Refuse a journal in which a task lacks, or repeats, a condition or a replicate."""
    cells: dict[str, set[tuple[str, int]]] = {}
    for row in rows:
        cells.setdefault(row["task_id"], set()).add((row["condition"], row["replicate_id"]))
    if (
        len(rows) != sum(len(cell) for cell in cells.values())
        or len({frozenset(cell) for cell in cells.values()}) != 1
    ):
        raise ValueError("every task needs each condition and replicate exactly once")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@app.command()
def render(
    source: Path,
    destination: Path,
    envelope: Annotated[
        bool, typer.Option(help="Also write E (envelope-json-v1) after T and H-gold.")
    ] = False,
    condition: Annotated[
        str, typer.Option(help="Label of the hierarchical variant: H-gold, or H-auto.")
    ] = "H-gold",
    traditional: Annotated[
        bool, typer.Option(help="Write T before the hierarchical variant.")
    ] = True,
    tasks: Annotated[
        Path | None,
        typer.Option(help="S_parse document: render only its tasks, verified by hash."),
    ] = None,
    flat: Annotated[
        bool, typer.Option(help="Also write F (pst-flat-json-v1) after the hierarchical variant.")
    ] = False,
    purpose: Annotated[str, typer.Option(help=_PURPOSE_HELP)] = "exploratory",
    freeze: Annotated[Path | None, typer.Option(help=_FREEZE_HELP)] = None,
) -> None:
    """Render reviewed canonical JSONL annotations into T/H-gold (and E) JSONL.

    With ``--condition H-auto`` the hierarchical variant of parsed spans is
    labelled H-auto (M19); ``--no-traditional`` leaves T out, so that it can be
    joined with the gold rendering of the same tasks; ``--tasks`` keeps only
    the tasks of an S_parse document; ``--flat`` adds the extension F.
    """
    if condition not in {"H-gold", "H-auto"}:
        raise ValueError("condition must be H-gold or H-auto")
    frozen = _purpose(purpose, freeze)
    selected: set[str] | None = None
    if tasks is not None:
        from pstparser.empirical.auto import load_s_parse

        selected = set(load_s_parse(_read_document(tasks)))
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
        if selected is not None and canonical.task_id not in selected:
            continue
        plain, hierarchical = render_prompt_variants(canonical)
        if condition == "H-auto":
            from pstparser.empirical.auto import as_hauto

            hierarchical = as_hauto(hierarchical)
        if traditional:
            variants.append(plain.as_dict())
        variants.append(hierarchical.as_dict())
        if flat:
            variants.append(render_flat_variant(canonical).as_dict())
        if envelope:
            variants.append(render_envelope_variant(canonical).as_dict())
    if selected is not None and not selected <= seen:
        raise ValueError("the S_parse document lists tasks absent from the source")
    if not variants:
        raise ValueError("empty corpus")
    from pstparser.repro.manifest import digest_file

    # State of the gold the canonical file was joined in (R-TRA-02), read from the
    # canonicalize manifest when it describes this very file.
    gold_status: str | None = None
    annotations_sha256: str | None = None
    joined = source.parent / "empirical-canonicalize_manifest.json"
    if joined.is_file():
        record = json.loads(joined.read_text(encoding="utf-8"))
        if record.get("canonical_sha256") == digest_file(source):
            gold_status = record.get("require_status")
            annotations_sha256 = record.get("annotations_sha256")
    if frozen["purpose"] == "final" and condition == "H-gold" and gold_status != "frozen":
        raise ValueError("a final rendering of H-gold requires frozen gold (canonicalize)")
    _write_rows(destination, variants)
    with Path(f"{destination}.render.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "format": "render-record-v1",
                    "variants_sha256": digest_file(destination),
                    "canonical_sha256": digest_file(source),
                    "gold_status": gold_status,
                    "annotations_sha256": annotations_sha256,
                    "h_gold_prompt_hashes": sorted(
                        row["prompt_hash"] for row in variants if row["condition"] == "H-gold"
                    ),
                    **frozen,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    typer.echo(
        f"Rendered {len(variants)} variants into {destination} "
        f"({frozen['purpose']}, freeze {frozen['freeze_sha256']})"
    )


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


@app.command("merge-journals")
def merge_journals_command(
    journals: Annotated[list[Path], typer.Argument(help="Journal directories of one session.")],
    sample: Annotated[Path, typer.Option(help="Frozen sample the core rows must cover.")],
    out: Annotated[Path, typer.Option(help="New directory for the merged generations.")],
    s_parse: Annotated[
        Path | None, typer.Option(help="S_parse document the H-auto rows must cover exactly.")
    ] = None,
) -> None:
    """Verified union of the journals of one session, before check-bigcodebench (C-26).

    Writes ``generations-core.jsonl`` (the conditions of every journal, on the
    sample) and, if any, ``generations-extra.jsonl`` (H-auto on S_parse).
    """
    from pstparser.empirical.auto import load_s_parse
    from pstparser.empirical.journal import merge_journals
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    merged = merge_journals(journals, _sample_ids(sample, None))
    if s_parse is not None:
        expected = set(load_s_parse(_read_document(s_parse)))
        hauto = {row["task_id"] for row in merged["extra"] if row["condition"] == "H-auto"}
        if hauto != expected:
            raise ValueError("the H-auto rows do not cover exactly the S_parse tasks")
    names = ["generations-core.jsonl", "generations-extra.jsonl"]
    ensure_stage_outputs(out, "empirical-merge-journals", names)
    _write_rows(out / names[0], merged["core"])
    if merged["extra"]:
        _write_rows(out / names[1], merged["extra"])
    write_manifest(
        {
            "stage": "empirical-merge-journals",
            "journals": [
                {
                    "directory": str(path),
                    "manifest_sha256": digest_file(path / "empirical-generate_manifest.json"),
                    "generations_sha256": digest_file(path / "generations.jsonl"),
                }
                for path in journals
            ],
            "sample_sha256": digest_file(sample),
            "s_parse_sha256": None if s_parse is None else digest_file(s_parse),
            "core_conditions": merged["core_conditions"],
            "extra_conditions": merged["extra_conditions"],
            "sessions": merged["sessions"],
            "gpu": merged["gpu"],
            "environment_digest": merged["environment_digest"],
            "core_sha256": digest_file(out / names[0]),
            "extra_sha256": digest_file(out / names[1]) if merged["extra"] else None,
            "source": source_snapshot(),
        },
        out,
    )
    typer.echo(f"{len(merged['core'])} core and {len(merged['extra'])} extra rows in {out}")


@app.command("check-bigcodebench")
def check_bigcodebench(
    generations: Path,
    parquet: Annotated[Path, typer.Option(help="Pinned parquet file of the dataset.")],
    revision: Annotated[str, typer.Option(help="Pinned revision of the dataset.")],
    image: Annotated[str, typer.Option(help="Harness image, as name@sha256:<digest>.")],
    nltk_dir: Annotated[Path, typer.Option(help="Local nltk data, mounted read-only.")],
    sample: Annotated[Path, typer.Option(help="Frozen sample the generations must cover.")],
    exclusions: Annotated[Path, typer.Option(help="Frozen exclusions of the calibration.")],
    directory: Annotated[Path, typer.Option(help="New directory for runs and outcomes.")],
    analysis_id: Annotated[str, typer.Option(help="Identifier written in every outcome.")],
    order_seed: Annotated[int, typer.Option(help="Seed of the order of the sample lines.")],
    split: Annotated[str, typer.Option(help="BigCodeBench split.")] = "v0.1.4",
    parallel: Annotated[int, typer.Option(help="Fixed number of harness workers.")] = 8,
    runs: Annotated[int, typer.Option(help="Independent runs before the majority rule.")] = 2,
    accept_discordance: Annotated[
        bool,
        typer.Option(
            help="Write the outcomes even if more than 1% of the programs changed outcome "
            "between two runs (recorded in the manifest)."
        ),
    ] = False,
) -> None:
    """Judge BigCodeBench generations with the official harness, offline.

    Code is extracted by the official sanitize inside the image. Every program
    runs twice, in two line orders; where the runs disagree it runs a third
    time and the majority decides, with a flaky flag.
    """
    from pstparser.empirical import bigcodebench
    from pstparser.empirical.analysis import write_outcomes
    from pstparser.repro.manifest import (
        digest_directory,
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    if runs != 2:
        raise ValueError("the frozen rule is two runs, with a third only on disagreement")
    rows = _json_lines(generations)
    sample_ids = set(_sample_ids(sample, None))
    excluded = {row["task_id"] for row in _read_document(exclusions)["excluded"]}
    tasks = {row["task_id"] for row in rows}
    if tasks != sample_ids:
        raise ValueError("the generations do not cover exactly the tasks of the sample")
    if tasks & excluded:
        raise ValueError(f"generations for excluded tasks: {sorted(tasks & excluded)[:5]}")
    _require_complete(rows)
    rows, prefill = _with_prefill(rows, generations)
    ensure_stage_outputs(
        directory,
        "empirical-check-bcb",
        ["outcomes.jsonl", "details.jsonl", "attempts.jsonl", "data", "sanitize"],
    )
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
    # The exclusions hold for the settings they were calibrated with, no others.
    calibrated_with = _read_document(exclusions).get("harness") or {}
    settings = config.as_dict()
    differing = sorted(
        key
        for key in ("image", "dataset_sha256", "nltk_sha256", "parallel", "calibrated", "network")
        if calibrated_with.get(key) != settings[key]
    )
    if differing:
        raise ValueError(
            f"harness settings differ from the calibration of the exclusions: {differing}"
        )
    environment = {
        "harness_version": bigcodebench.harness_version(image),
        "docker": bigcodebench.docker_resources(),
    }
    result = bigcodebench.evaluate_generations(
        config,
        rows,
        directory,
        analysis_id=analysis_id,
        order_seed=order_seed,
        accept_discordance=accept_discordance,
    )
    write_outcomes(result["outcomes"], directory / "outcomes.jsonl")
    _write_rows(directory / "details.jsonl", result["details"])
    _write_rows(directory / "attempts.jsonl", result["attempts"])
    write_manifest(
        {
            "stage": "empirical-check-bcb",
            "analysis_id": analysis_id,
            "assistant_prefill_prepended": prefill is not None,
            "harness": config.as_dict(),
            **environment,
            **result["summary"],
            "generations_sha256": digest_file(generations),
            "sample_sha256": digest_file(sample),
            "exclusions_sha256": digest_file(exclusions),
            "outcomes_sha256": digest_file(directory / "outcomes.jsonl"),
            "details_sha256": digest_file(directory / "details.jsonl"),
            "attempts_sha256": digest_file(directory / "attempts.jsonl"),
            "source": source_snapshot(),
        },
        directory,
    )
    summary = result["summary"]
    typer.echo(
        f"{summary['generations']} generations, {summary['executed']} executed, "
        f"{summary['discordant']} discordant; {directory / 'outcomes.jsonl'}"
    )
    if summary["discordance_accepted"]:
        typer.echo("accepted: more than 1% of the programs changed outcome between two runs")


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


@app.command("sample-hard")
def sample_hard(
    permutation: Path,
    pilot: Annotated[Path, typer.Option(help="Frozen pilot pool, which must lie outside Hard.")],
    out: Annotated[Path, typer.Option(help="New file for the Hard census (S_gold).")],
) -> None:
    """Write the census of the eligible Hard tasks, in permutation order (S_gold).

    Nothing is drawn: the Hard subset is the benchmark's own, and the frozen
    exclusions already left out the ineligible tasks.
    """
    from pstparser.empirical import sampling
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    frozen, hard = sampling.read_permutation_document(_read_document(permutation))
    pilot_document = _read_document(pilot)
    if pilot_document.get("permutation_sha256") != digest_file(permutation):
        raise ValueError("the pilot pool was not drawn from this permutation file")
    pool = sampling.read_sample_document(pilot_document, "pilot-sample")
    census = sampling.hard_census(frozen, hard, pool)
    ensure_stage_outputs(out.parent, "empirical-sample-hard", [out.name])
    provenance = {
        "benchmark": pilot_document["benchmark"],
        "permutation_sha256": digest_file(permutation),
        "pilot_sha256": digest_file(pilot),
    }
    _write_document(
        out,
        sampling.sample_document("hard-census", census, frozen, hard, **provenance),
    )
    write_manifest(
        {
            "stage": "empirical-sample-hard",
            **provenance,
            "n": len(census.task_ids),
            "census_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(census.task_ids)} eligible Hard tasks in {out}")


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


@app.command("sample-reannotation")
def sample_reannotation(
    sources: Path,
    rule: Annotated[str, typer.Option(help="Draft rule of the benchmark (judgement units).")],
    n: Annotated[int, typer.Option(help="Number of tasks to re-annotate from scratch.")],
    seed: Annotated[int, typer.Option(help="Seed of the draw, fixed before the gold review.")],
    out: Annotated[Path, typer.Option(help="New file for the re-annotation sample.")],
    sample: Annotated[
        Path | None, typer.Option(help="Frozen sample the tasks must belong to (BigCodeBench).")
    ] = None,
    by_family: Annotated[
        bool, typer.Option(help="One prompt from each of n seeded families (LLMSecEval).")
    ] = False,
) -> None:
    """Draw the re-annotation sample among the tasks with at least one judgement unit.

    Drawn with a seed before the gold review, so that the choice of tasks cannot
    depend on how the review went (plan of 18/11/2026, P-07).
    """
    from pstparser.empirical import sampling
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.empirical.covariates import JUDGEMENT_UNITS_RULE, judgement_regions
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    records = read_sources(sources)
    if not records:
        raise ValueError("empty sources")
    if sample is not None:
        allowed = set(_sample_ids(sample, None))
        if not allowed <= {record.task_id for record in records}:
            raise ValueError("a task of the frozen sample is not among the sources")
        records = [record for record in records if record.task_id in allowed]
    dataset_sha256 = records[0].dataset_file_sha256
    candidates = [record for record in records if judgement_regions(record.text, rule)]
    drawn = sampling.reannotation_sample(
        candidates, n=n, seed=seed, dataset_sha256=dataset_sha256, by_family=by_family
    )
    ensure_stage_outputs(out.parent, "empirical-sample-reannotation", [out.name])
    provenance = {
        "sources_sha256": digest_file(sources),
        "sample_sha256": None if sample is None else digest_file(sample),
    }
    _write_document(
        out,
        {
            "format_version": 1,
            "kind": "reannotation-sample",
            "benchmark": records[0].benchmark,
            "dataset_revision": records[0].dataset_revision,
            "dataset_file_sha256": dataset_sha256,
            "text_field": records[0].text_field,
            "method": sampling.ORDER_METHOD,
            "purpose": "reannotation",
            "seed": seed,
            "rule": rule,
            "judgement_units_rule": JUDGEMENT_UNITS_RULE,
            "by_family": by_family,
            "population": len(records),
            "candidates": len(candidates),
            "candidate_families": len({record.family_id for record in candidates}),
            "n": len(drawn),
            **provenance,
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
            "stage": "empirical-sample-reannotation",
            "seed": seed,
            "n": n,
            "rule": rule,
            "by_family": by_family,
            **provenance,
            "reannotation_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(drawn)} of {len(candidates)} candidate tasks in {out}")


def _sample_ids(path: Path, selection: str | None) -> list[str]:
    """Read the task identifiers of a frozen sample, of S_parse, or of one LLMSecEval selection."""
    document = _read_document(path)
    if document.get("format") == "s-parse-v1" and selection is None:
        from pstparser.empirical.auto import load_s_parse

        # The H-auto rows are judged on S_parse, whose hash is checked here.
        return load_s_parse(document)
    if "tasks" in document:
        if selection is not None:
            raise ValueError("--selection applies to an identifier map only")
        return [row["task_id"] for row in document["tasks"]]
    if "records" in document and selection is not None:
        return [row["task_id"] for row in document["records"] if selection in row["selections"]]
    raise ValueError(f"{path}: not a sample, or an identifier map without --selection")


@app.command("propose-spans")
def propose_spans(
    sources: Path,
    rule: Annotated[str, typer.Option(help="Named pre-annotation rule.")],
    out: Annotated[Path, typer.Option(help="New annotation file; every row is a draft.")],
    sample: Annotated[
        Path | None, typer.Option(help="Draft only the tasks of this sample.")
    ] = None,
    selection: Annotated[str | None, typer.Option(help="Selection of an identifier map.")] = None,
) -> None:
    """Propose draft spans from fixed markers; a draft is never gold."""
    from pstparser.empirical import annotations
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    by_id = {record.task_id: record for record in read_sources(sources)}
    wanted = _sample_ids(sample, selection) if sample is not None else list(by_id)
    absent = [task for task in wanted if task not in by_id]
    if absent or not wanted:
        raise ValueError(f"sample tasks without a source: {absent[:5]}")
    rows = [annotations.propose(by_id[task], rule) for task in wanted]
    # A proposal that the renderer would refuse is a defect of the rule: stop.
    annotations.canonicalize(
        [by_id[task] for task in wanted], rows, require_status="draft", sample_ids=wanted
    )
    ensure_stage_outputs(out.parent, "empirical-propose-spans", [out.name])
    annotations.write_annotations(rows, out)
    write_manifest(
        {
            "stage": "empirical-propose-spans",
            "rule": rule,
            "splitter": annotations.SPLITTER_VERSION,
            "annotation_version": annotations.ANNOTATION_VERSION,
            "status": "draft",
            "rows": len(rows),
            "sources_sha256": digest_file(sources),
            "sample_sha256": digest_file(sample) if sample is not None else None,
            "selection": selection,
            "annotations_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(rows)} draft annotations in {out}")


@app.command("gold-diff")
def gold_diff_command(
    sources: Path,
    drafts: Path,
    gold: Path,
    rule: Annotated[str, typer.Option(help="Rule that proposed the drafts.")],
    out: Annotated[Path, typer.Option(help="New diff report (JSON).")],
    sample: Annotated[
        Path | None, typer.Option(help="Frozen sample the gold must cover exactly.")
    ] = None,
) -> None:
    """Diff between the rule drafts and the reviewed gold: units and characters by rule.

    A prespecified analysis (plan of 18/11/2026, P-07): share of tasks changed,
    characters and units changed for each rule, and the share of characters the
    rules leave to the reviewer.
    """
    from pstparser.empirical import annotations
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.empirical.golddiff import gold_diff
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    gold_rows = annotations.load_annotations(gold)
    if sample is not None and sorted(_sample_ids(sample, None)) != sorted(
        row.task_id for row in gold_rows
    ):
        raise ValueError("the gold does not cover exactly the tasks of the frozen sample")
    report = gold_diff(read_sources(sources), annotations.load_annotations(drafts), gold_rows, rule)
    ensure_stage_outputs(out.parent, "empirical-gold-diff", [out.name])
    _write_document(out, report)
    write_manifest(
        {
            "stage": "empirical-gold-diff",
            "rule": rule,
            "sources_sha256": digest_file(sources),
            "drafts_sha256": digest_file(drafts),
            "gold_sha256": digest_file(gold),
            "sample_sha256": None if sample is None else digest_file(sample),
            "report_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{report['tasks_changed']} of {report['tasks']} tasks changed; {out}")


@app.command("gold-lint")
def gold_lint(sources: Path, annotation_file: Path) -> None:
    """List the blocking defects of an annotation file; fail if there is any."""
    from pstparser.empirical import annotations
    from pstparser.empirical.benchmarks import read_sources

    by_id = {record.task_id: record for record in read_sources(sources)}
    defects = 0
    rows = annotations.load_annotations(annotation_file)
    for row in rows:
        source = by_id.get(row.task_id)
        problems = ["no such source"] if source is None else annotations.lint(source, row)
        for problem in problems:
            defects += 1
            typer.echo(f"{row.task_id}: {problem}")
    states = sorted({row.status for row in rows})
    typer.echo(f"{len(rows)} annotations, states {states}, {defects} defects")
    if defects:
        raise typer.Exit(code=1)


@app.command()
def canonicalize(
    sources: Path,
    annotation_file: Path,
    sample: Annotated[Path, typer.Option(help="Frozen sample the annotations must cover.")],
    out: Annotated[Path, typer.Option(help="New canonical file, input of the render command.")],
    selection: Annotated[str | None, typer.Option(help="Selection of an identifier map.")] = None,
    require_status: Annotated[
        str, typer.Option(help="State every annotation must be in.")
    ] = "frozen",
) -> None:
    """Join annotations in the required state to their sources, for rendering."""
    from pstparser.empirical import annotations
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    prompts = annotations.canonicalize(
        read_sources(sources),
        annotations.load_annotations(annotation_file),
        require_status=require_status,
        sample_ids=_sample_ids(sample, selection),
    )
    ensure_stage_outputs(out.parent, "empirical-canonicalize", [out.name])
    _write_rows(
        out,
        [
            {
                "task_id": prompt.task_id,
                "text": prompt.text,
                "annotation_version": prompt.annotation_version,
                "annotations": [
                    {"path": s.path, "start": s.start, "end": s.end, "phrase": s.phrase}
                    for s in prompt.annotations
                ],
            }
            for prompt in prompts
        ],
    )
    write_manifest(
        {
            "stage": "empirical-canonicalize",
            "require_status": require_status,
            "rows": len(prompts),
            "sources_sha256": digest_file(sources),
            "annotations_sha256": digest_file(annotation_file),
            "sample_sha256": digest_file(sample),
            "selection": selection,
            "canonical_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(prompts)} canonical prompts in {out}")


def _reset_peak_vram() -> bool:
    """Start a peak-memory window on the GPU; False when there is no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats()
    return True


def _peak_vram_bytes() -> int:
    """Peak GPU memory allocated by torch since the last reset (pilot criterion G11)."""
    import torch

    return int(torch.cuda.max_memory_allocated())


@app.command()
def generate(variants: Path, configuration: Path, directory: Path) -> None:
    """Generate/resume only an explicitly configured, revision-pinned experiment."""
    # Imports remain local: rendering and analysis do not load torch/models.
    import uuid

    from pstparser.config import InferenceConfig, ModelConfig
    from pstparser.inference.generate import (
        decoding_parameters,
        generate_messages,
        prefill_token_ids,
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
        # The one GPU of the whole campaign (C-01, R-PIA-02): a run elsewhere is refused.
        "gpu",
    }
    # A configuration without format_version is version 1 and keeps its closed
    # key set; the keys added since are admitted only by declaring version 2.
    # assistant_prefill (S-PREFILL, 00 T-3) is the one optional key of version 2.
    versioned = "format_version" in config
    if versioned:
        if config["format_version"] != 2:
            raise ValueError("format_version must be 2 when present")
        allowed |= {"format_version", "user_template", "conditions"}
    if set(config) - {"assistant_prefill"} != allowed or (
        "assistant_prefill" in config and not versioned
    ):
        raise ValueError(f"configuration requires exactly: {sorted(allowed)}")
    assistant_prefill = config.get("assistant_prefill")
    if assistant_prefill is not None and (
        not isinstance(assistant_prefill, str) or not assistant_prefill
    ):
        raise ValueError("assistant_prefill must be a non-empty string or null")
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
    for name in ("protocol_version", "dataset_revision", "system_prompt", "review_record", "gpu"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"missing {name}")
    review = configuration.parent / config["review_record"]
    if not review.is_file():
        raise ValueError("an explicit review/freeze record is required")
    if config["purpose"] == "final":
        from pstparser.empirical.freeze import require_final_freeze

        # Before any model is loaded: a final run belongs to a verified freeze
        # that lists this configuration and these variants (M12).
        require_final_freeze(review, variants, configuration)
    model_config = ModelConfig(**config["model"])
    inference = InferenceConfig(**config["inference"])
    if not re.fullmatch(r"[0-9a-f]{40}", model_config.revision or ""):
        raise ValueError("provide an immutable 40-character model revision")
    # A pilot measures the treatment of the final runs (R-GEN-03): only the hf
    # backend loads the requested snapshot and keeps the published template.
    if model_config.backend != "hf" or model_config.chat_template != "native":
        raise ValueError("generation requires the hf backend with the 'native' template")
    if inference.structured_output or inference.batch_size != 1:
        raise ValueError("code generation uses unconstrained output, batch_size=1")
    if not inference.do_sample and len(config["replicate_ids"]) != 1:
        raise ValueError("greedy repetitions are not independent samples")
    # Before anything is loaded: pilot, final runs, determinism and H-auto share
    # one declared GPU, since greedy outputs may differ across GPUs (R-PIA-02).
    measured_environment = environment()
    found = measured_environment.get("accelerator", {}).get("devices") or []
    if not found or any(device != config["gpu"] for device in found):
        raise ValueError(f"the run requires the declared GPU {config['gpu']!r}; found {found}")
    # Q-03 (b): the kernel string and the NVIDIA driver may change between Colab
    # sessions on the same GPU. They are recorded per completion, not frozen.
    volatile = {
        "platform": measured_environment.pop("platform", None),
        "gpu_driver": measured_environment.get("accelerator", {}).pop("driver", None),
    }
    rendered = [PromptVariant(**row) for row in _rows(variants)]
    if versioned and {variant.condition for variant in rendered} != set(config["conditions"]):
        raise ValueError("the variants do not hold exactly the configured conditions")
    conditions = tuple(config["conditions"]) if versioned else ("T", "H-gold")
    requests = request_plan(rendered, config["replicate_ids"], config["seed"], conditions)
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
    if final:
        from pstparser.empirical.freeze import default_root, load_freeze
        from pstparser.models.loader import verify_expected_files

        # The loaded files must be the frozen ones (Q-02 b, R-GEN-02), before loading.
        expected = default_root(review) / "protocol" / "model.json"
        listed = set(load_freeze(review)["items"].values())
        if digest_file(expected) not in listed:
            raise ValueError("final runs require protocol/model.json in the protocol freeze")
        assert snapshot is not None
        try:
            verify_expected_files(snapshot, _read_document(expected))
        except Exception as error:
            raise ValueError(str(error)) from error
    model, tokenizer = load_untuned(model_config)
    runtime = runtime_provenance(model, tokenizer, model_config, snapshot)
    if final and not runtime["revision_verified"]:
        raise ValueError("final runs require a verified model revision")
    if runtime["chat_template_matches_official"] is not True:
        raise ValueError("generation requires the published conversation template")
    # Check BOTH variants for ALL tasks before generating the first condition.
    template_checked: set[str] = set()
    prefill_tokens = (
        len(prefill_token_ids(tokenizer, assistant_prefill)) if assistant_prefill is not None else 0
    )
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
            len(tokens) + prefill_tokens,
            inference.max_new_tokens,
            model_config.max_seq_length,
        )
    frozen = {
        **config,
        "review_record_sha256": digest_file(review),
        "variant_file_sha256": digest_file(variants),
        "system_prompt_sha256": digest_text(config["system_prompt"]),
        "user_template_sha256": digest_text(user_template) if user_template is not None else None,
        # Present only with a prefill, so the fingerprint of every other journal is unchanged.
        **(
            {"assistant_prefill_sha256": digest_text(assistant_prefill)}
            if assistant_prefill is not None
            else {}
        ),
        "decoding": decoding_parameters(inference),
        "runtime": runtime,
        "environment": measured_environment,
    }

    # Identity of this process (M18): a resume in another session leaves a second
    # session_id in the journal; the environment itself must not change (S03).
    devices = frozen["environment"].get("accelerator", {}).get("devices") or []
    session = {
        "session_id": uuid.uuid4().hex,
        "gpu_name": devices[0] if devices else None,
        "environment_digest": digest_text(
            json.dumps(frozen["environment"], sort_keys=True, ensure_ascii=False)
        ),
        **volatile,
    }

    def complete(request: Any) -> dict[str, Any]:
        seed_everything(request["seed"])
        messages = chat_messages(config["system_prompt"], user_template, request["text"])
        measured = _reset_peak_vram()
        result = generate_messages(
            model,
            tokenizer,
            messages,
            inference,
            max_seq_length=model_config.max_seq_length,
            assistant_prefill=assistant_prefill,
        )
        record = {
            "user_message_sha256": digest_text(messages[1]["content"]),
            "raw_output": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "finish_reason": result.finish_reason,
            "elapsed_seconds": result.elapsed_seconds,
        }
        if measured:
            record["peak_vram_bytes"] = _peak_vram_bytes()
        return {**record, **session}

    records = run_journal(requests, frozen, directory, complete)
    typer.echo(f"{len(records)} recorded completions; {directory}")


@app.command()
def power(
    grid: Annotated[
        Path, typer.Option(help="YAML or JSON grid: seed, paired and clustered cells.")
    ],
    out: Annotated[Path, typer.Option(help="New CSV file for the power table.")],
) -> None:
    """Exact mid-p power and MDE (paired cells); sign-flip power by simulation (clustered)."""
    import csv

    import yaml

    from pstparser.empirical.power import COLUMNS, power_grid
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    ensure_stage_outputs(out.parent, "empirical-power", [out.name])
    rows = power_grid(yaml.safe_load(grid.read_bytes().decode("utf-8")))
    with out.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                ""
                if row.get(column) is None
                else repr(row[column])
                if isinstance(row[column], float)
                else str(row[column])
                for column in COLUMNS
            )
    write_manifest(
        {
            "stage": "empirical-power",
            "grid_sha256": digest_file(grid),
            "table_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(str(out))


@app.command("pilot-report")
def pilot_report_command(
    journal: Annotated[Path, typer.Argument(help="Directory of one generation journal.")],
    out: Annotated[Path, typer.Option(help="New JSON file for the report.")],
    outcomes: Annotated[
        list[Path] | None,
        typer.Option(
            help="Outcomes of a check run of this journal; repeat with --merged for the "
            "check runs of the merged files of a session (core and extra)."
        ),
    ] = None,
    details: Annotated[
        list[Path] | None,
        typer.Option(help="Details written by the same check run(s), in the same order."),
    ] = None,
    merged: Annotated[
        bool,
        typer.Option(
            help="The check runs cover the merged files of two journals: keep only the "
            "cells and requests of this journal (C-26), then check exact coverage."
        ),
    ] = False,
) -> None:
    """Report the technical pilot criteria and b + c, never a difference between conditions."""
    from pstparser.empirical.pilot_report import pilot_report
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    outcome_files = outcomes or []
    detail_files = details or []
    if len(outcome_files) != len(detail_files):
        raise ValueError("give one --details file for each --outcomes file")
    if len(outcome_files) > 1 and not merged:
        raise ValueError("several check runs are read only with --merged")
    if merged and not outcome_files:
        raise ValueError("--merged needs --outcomes and --details")
    ensure_stage_outputs(out.parent, "empirical-pilot-report", [out.name])
    manifest_path = journal / "empirical-generate_manifest.json"
    journal_path = journal / "generations.jsonl"
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    rows = _json_lines(journal_path)
    report = pilot_report(
        manifest,
        rows,
        [r for path in outcome_files for r in read_outcomes(path)] if outcome_files else None,
        [d for path in detail_files for d in _json_lines(path)] if detail_files else None,
        restrict=merged,
    )
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    write_manifest(
        {
            "stage": "empirical-pilot-report",
            "report_version": report["report_version"],
            "generate_manifest_sha256": digest_file(manifest_path),
            "journal_sha256": digest_file(journal_path),
            "outcomes_sha256": [digest_file(path) for path in outcome_files] or None,
            "details_sha256": [digest_file(path) for path in detail_files] or None,
            "merged": merged,
            "report_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(str(out))


@app.command("blind-sample")
def blind_sample(
    details: Annotated[
        list[Path], typer.Argument(help="details.jsonl of check-security (both measures).")
    ],
    journal: Annotated[Path, typer.Option(help="The security journal (generations.jsonl).")],
    sources: Annotated[Path, typer.Option(help="LLMSecEval sources: the prompt in its T form.")],
    seed: Annotated[int, typer.Option(help="Seed of the sample, fixed before opening files.")],
    out: Annotated[Path, typer.Option(help="New directory: files/, sheet.jsonl, key.jsonl.")],
    form: Annotated[
        str, typer.Option(help="full, or reduced (only the U and targeted V cases).")
    ] = "full",
) -> None:
    """Draw the blind review (04 §3.10): programs renamed by hash, one sheet, a separate key.

    Move ``key.jsonl`` out of the reviewer's sight before opening the sheet.
    """
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.empirical.blind import BLIND_VERSION, SHEET_FIELDS, draw_cases
    from pstparser.empirical.extraction import extract_fenced_blocks
    from pstparser.repro.manifest import digest_file, source_snapshot, write_manifest

    if out.exists():
        raise FileExistsError(f"{out} exists: each draw needs a new directory")
    rows = [row for path in details for row in _json_lines(path)]
    cases = draw_cases(rows, seed, form)
    outputs = {row["request_id"]: row["raw_output"] for row in _json_lines(journal)}
    prompts = {record.task_id: record.text for record in read_sources(sources)}
    (out / "files").mkdir(parents=True)
    sheet, key = [], []
    for case in cases:
        extracted = extract_fenced_blocks(outputs[case["request_id"]])
        name = f"{case['case_id']}.py" if extracted.code is not None else f"{case['case_id']}.txt"
        text = extracted.code if extracted.code is not None else outputs[case["request_id"]]
        with (out / "files" / name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        sheet.append(
            {
                "case_id": case["case_id"],
                "prompt": prompts[case["task_id"]],
                "file": f"files/{name}",
                **dict.fromkeys(SHEET_FIELDS),
            }
        )
        key.append(case)
    _write_rows(out / "sheet.jsonl", sheet)
    _write_rows(out / "key.jsonl", key)
    write_manifest(
        {
            "stage": "empirical-blind-sample",
            "version": BLIND_VERSION,
            "form": form,
            "seed": seed,
            "details_sha256": [digest_file(path) for path in details],
            "journal_sha256": digest_file(journal),
            "cases": len(cases),
            "sheet_sha256": digest_file(out / "sheet.jsonl"),
            "key_sha256": digest_file(out / "key.jsonl"),
            "source": source_snapshot(),
        },
        out,
    )
    typer.echo(f"{len(cases)} cases in {out}; move key.jsonl away before reviewing")


@app.command("blind-score")
def blind_score(
    sheet: Annotated[Path, typer.Argument(help="The completed sheet.jsonl.")],
    key: Annotated[Path, typer.Argument(help="The key.jsonl of the same draw.")],
    out: Annotated[Path, typer.Option(help="New score report (JSON).")],
) -> None:
    """Score the blind review: FP and FN by condition, kappa U2/U4, accuracy of the blind."""
    from pstparser.empirical.blind import score_review
    from pstparser.repro.manifest import digest_file

    report = score_review(_json_lines(sheet), _json_lines(key))
    report["inputs"] = {"sheet_sha256": digest_file(sheet), "key_sha256": digest_file(key)}
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_document(out, report)
    typer.echo(f"{report['cases']} cases scored; {out}")


@app.command("determinism-sample")
def determinism_sample(
    variants: Annotated[Path, typer.Argument(help="Variants of the greedy journal.")],
    journal: Annotated[Path, typer.Argument(help="The greedy journal (generations.jsonl).")],
    seed: Annotated[int, typer.Option(help="Seed of the selection.")],
    out: Annotated[Path, typer.Option(help="New variants file of the selected tasks.")],
) -> None:
    """Select whole tasks holding at least 5% of the greedy requests (minimum 20), 00 G-6.

    The tasks are ordered by SHA-256(seed | task) and taken until the share is
    reached; every condition of a chosen task is kept, so generate on the
    output in a separate session reproduces the same request identifiers.
    """
    import hashlib
    import math

    from pstparser.repro.manifest import digest_file, source_snapshot, write_manifest

    rows = _json_lines(journal)
    per_task: dict[str, int] = {}
    for row in rows:
        per_task[row["task_id"]] = per_task.get(row["task_id"], 0) + 1
    wanted = max(20, math.ceil(0.05 * len(rows)))
    order = sorted(per_task, key=lambda task: hashlib.sha256(f"{seed}|{task}".encode()).hexdigest())
    chosen: list[str] = []
    for task in order:
        if sum(per_task[name] for name in chosen) >= wanted:
            break
        chosen.append(task)
    selected = set(chosen)
    kept = [row for row in _json_lines(variants) if row["task_id"] in selected]
    if {row["task_id"] for row in kept} != selected:
        raise ValueError("the variants do not hold every selected task")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    _write_rows(out, kept)
    write_manifest(
        {
            "stage": "empirical-determinism-sample",
            "seed": seed,
            "journal_sha256": digest_file(journal),
            "variants_sha256": digest_file(variants),
            "greedy_requests": len(rows),
            "wanted_requests": wanted,
            "tasks": sorted(selected),
            "selected_variants_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(selected)} tasks, {len(kept)} variants in {out}")


@app.command("determinism-compare")
def determinism_compare(
    original: Annotated[Path, typer.Argument(help="The greedy journal of the run.")],
    regenerated: Annotated[Path, typer.Argument(help="Journal regenerated in another session.")],
    out: Annotated[Path, typer.Option(help="New determinism report (JSON).")],
) -> None:
    """Compare the regenerated outputs with the original ones, byte for byte (00 G-6)."""
    from pstparser.repro.manifest import digest_file

    first = {row["request_id"]: row["raw_output"] for row in _json_lines(original)}
    second = {row["request_id"]: row["raw_output"] for row in _json_lines(regenerated)}
    unknown = sorted(set(second) - set(first))
    if unknown or not second:
        raise ValueError(f"regenerated requests absent from the original run: {unknown[:5]}")
    differing = sorted(request for request in second if second[request] != first[request])
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_document(
        out,
        {
            "format": "determinism-v1",
            "original_sha256": digest_file(original),
            "regenerated_sha256": digest_file(regenerated),
            "compared": len(second),
            "identical": len(second) - len(differing),
            "identical_share": (len(second) - len(differing)) / len(second),
            "differing": differing,
        },
    )
    typer.echo(f"{len(second) - len(differing)} of {len(second)} identical; {out}")


@app.command("model-digests")
def model_digests(
    name: Annotated[str, typer.Option(help="Model repository on the Hub.")],
    revision: Annotated[str, typer.Option(help="Immutable 40-character commit.")],
    out: Annotated[Path, typer.Option(help="New model.json with the SHA-256 of every file.")],
) -> None:
    """Hash every file of the model snapshot, weights included, into model.json (Q-02 b).

    Run once where the model is downloaded (Colab); the file goes to
    protocol/model.json and into the freeze, and generate final compares the
    loaded snapshot with it.
    """
    from pstparser.models.loader import resolve_snapshot

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("provide an immutable 40-character model revision")
    snapshot = resolve_snapshot(name, revision, hash_weights=True)
    if snapshot.sha != revision:
        raise ValueError(f"the snapshot holds {snapshot.sha}, not {revision}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        document = {
            "format": "model-digests-v1",
            "name": name,
            "revision": snapshot.sha,
            "files": dict(sorted(snapshot.files.items())),
        }
        handle.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    typer.echo(f"{len(snapshot.files)} files of {name}@{revision} in {out}")


@app.command()
def design(
    sample: Annotated[
        Path,
        typer.Argument(
            help="Frozen sample (prefix-600/bcb-prefix-600.json) or LLMSecEval identifier map."
        ),
    ],
    analysis_id: Annotated[str, typer.Option(help="analysis_id of the outcome ledger.")],
    endpoint: Annotated[str, typer.Option(help="functional or security.")],
    conditions: Annotated[str, typer.Option(help="Baseline and treatment, e.g. T,H-gold.")],
    out: Annotated[Path, typer.Option(help="New design file (JSON), input of analyze.")],
    selection: Annotated[
        str | None, typer.Option(help="Selection of the identifier map (security), e.g. S2.")
    ] = None,
    replicates: Annotated[str, typer.Option(help="Replicate IDs, e.g. 0 or 0,1,2,3,4,5.")] = "0",
) -> None:
    """Write the analysis design of a frozen sample (R-STA-04): tasks, families, replicates.

    Functional designs take the tasks of a frozen sample, in its order, without
    families; security designs take the prompts of one selection of the
    LLMSecEval identifier map with their family, the unit of inference.
    """
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    document = _read_document(sample)
    pair = conditions.split(",")
    if len(pair) != 2:
        raise ValueError("conditions must name a baseline and a treatment")
    replicate_ids = [int(value) for value in replicates.split(",")]
    if endpoint == "functional":
        if "tasks" not in document or selection is not None:
            raise ValueError("a functional design reads a frozen sample, without --selection")
        tasks = [{"task_id": row["task_id"]} for row in document["tasks"]]
    elif endpoint == "security":
        if "records" not in document or selection is None:
            raise ValueError("a security design reads the identifier map with --selection")
        tasks = [
            {"task_id": row["task_id"], "family_id": row["family_id"]}
            for row in document["records"]
            if selection in row["selections"]
        ]
    else:
        raise ValueError("endpoint must be functional or security")
    if not tasks:
        raise ValueError("the sample has no task for this design")
    payload = {
        "analysis_id": analysis_id,
        "endpoint": endpoint,
        "tasks": tasks,
        "replicate_ids": replicate_ids,
        "conditions": pair,
    }
    # The same checks analyze will apply, before anything is written.
    AnalysisDesign(
        analysis_id,
        endpoint,  # type: ignore[arg-type]
        tuple(TaskSpec(**task) for task in tasks),
        tuple(replicate_ids),
        (pair[0], pair[1]),
    )
    ensure_stage_outputs(out.parent, "empirical-design", [out.name])
    _write_document(out, payload)
    write_manifest(
        {
            "stage": "empirical-design",
            "sample_sha256": digest_file(sample),
            "selection": selection,
            "design_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(tasks)} tasks in {out}")


@app.command()
def analyze(
    design: Path,
    outcomes: Path,
    destination: Path,
    ledger_conditions: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated conditions of a ledger holding more than the analysed "
            "pair (e.g. T,H-gold,E); rows of the others are counted and left out."
        ),
    ] = None,
    purpose: Annotated[str, typer.Option(help=_PURPOSE_HELP)] = "exploratory",
    freeze: Annotated[Path | None, typer.Option(help=_FREEZE_HELP)] = None,
    details: Annotated[
        Path | None,
        typer.Option(help="details.jsonl of check-bigcodebench, for the S-FLAKY sensitivity."),
    ] = None,
    near_duplicates: Annotated[
        Path | None,
        typer.Option(help="Groups of near-duplicates, for the S-NEARDUP sensitivity."),
    ] = None,
) -> None:
    """Analyse the explicit paired design, rejecting absent/duplicate/mixed rows."""
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    frozen = _purpose(purpose, freeze)
    ensure_stage_outputs(destination.parent, "empirical-analysis", [destination.name])
    payload = json.loads(design.read_text(encoding="utf-8"))
    payload["tasks"] = tuple(TaskSpec(**task) for task in payload["tasks"])
    for key in ("replicate_ids", "conditions"):
        if key in payload:
            payload[key] = tuple(payload[key])
    declared = tuple(ledger_conditions.split(",")) if ledger_conditions is not None else None
    report = analyze_paired(
        AnalysisDesign(**payload),
        read_outcomes(outcomes),
        ledger_conditions=declared,
        details=_json_lines(details) if details is not None else None,
        near_duplicate_groups=(
            _read_document(near_duplicates)["groups"] if near_duplicates is not None else None
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_manifest(
        {
            "stage": "empirical-analysis",
            "analysis_version": "paired-v2",
            "design_sha256": digest_file(design),
            "outcomes_sha256": digest_file(outcomes),
            "report_sha256": digest_file(destination),
            **frozen,
            "details_sha256": digest_file(details) if details is not None else None,
            "near_duplicates_sha256": (
                digest_file(near_duplicates) if near_duplicates is not None else None
            ),
            "source": source_snapshot(),
            **({"ledger_conditions": list(declared)} if declared is not None else {}),
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
    from pstparser.empirical.retries import MAX_ATTEMPTS, RETRY_POLICY, with_retries
    from pstparser.empirical.sandbox import ExecutionResult, SandboxConfig, run_python_check
    from pstparser.empirical.workflow import (
        FunctionalOracle,
        calibrate_oracles,
        evaluate_functional,
    )
    from pstparser.repro.manifest import (
        digest_file,
        digest_text,
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
            "attempts.jsonl",
        ],
    )
    config = SandboxConfig(image)
    trusted = [FunctionalOracle(**row) for row in _rows(oracles)]
    generated = _rows(generations)
    if not generated or {r["task_id"] for r in generated} != {o.task_id for o in trusted}:
        raise ValueError("generation and oracle task sets must be identical and non-empty")

    attempt_log: list[dict[str, Any]] = []

    def failure(result: ExecutionResult) -> str | None:
        if result.status != "infrastructure_error":
            return None
        return result.detail[-500:] or "infrastructure_error"

    def execute(code: str, harness: str) -> Any:
        # Only infrastructure errors are retried; a failed or timed-out program is final.
        result, attempts = with_retries(lambda _: run_python_check(code, harness, config), failure)
        attempt_log.extend(
            {"code_sha256": digest_text(code), "harness_sha256": digest_text(harness), **attempt}
            for attempt in attempts
        )
        return result

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
    _write_rows(directory / "attempts.jsonl", attempt_log)
    write_manifest(
        {
            "stage": "empirical-check",
            "analysis_id": analysis_id,
            "calibrated": calibrated,
            "sandbox": asdict(config),
            "retry_count": sum(1 for attempt in attempt_log if attempt["attempt"] > 1),
            "retry_policy": RETRY_POLICY,
            "max_attempts": MAX_ATTEMPTS,
            "attempts_sha256": digest_file(directory / "attempts.jsonl"),
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


@app.command("calibrate-security")
def calibrate_security(
    codeql: Annotated[Path, typer.Option(help="Frozen settings of the analyser.")],
    bundle_dir: Annotated[Path, typer.Option(help="Local copy of the unpacked bundle.")],
    cards: Annotated[Path, typer.Option(help="Scenario cards, with the controls next to them.")],
    directory: Annotated[Path, typer.Option(help="New directory for analyses and report.")],
    bundle_archive: Annotated[
        Path | None, typer.Option(help="Bundle archive, to check its hash.")
    ] = None,
    secure_samples: Annotated[
        Path | None, typer.Option(help="Secure code samples folder of the pinned dataset.")
    ] = None,
    workers: Annotated[int, typer.Option(help="Programs analysed at the same time.")] = 1,
) -> None:
    """Analyse the controls of every family with the exact settings of the collection.

    Writes ``calibration.json`` with what each control received and the state
    observed for each family, next to the state the cards declare.
    """
    from pstparser.empirical import codeql as oracle
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    config = oracle.read_config(codeql, bundle_dir, bundle_archive)
    scenario_cards = oracle.read_cards(cards)
    controls = oracle.read_controls(cards, secure_samples)
    ensure_stage_outputs(directory, "empirical-calibrate-security", ["calibration.json"])
    version = oracle.codeql_version(config)
    report = oracle.calibrate(config, scenario_cards, controls, directory, workers=workers)
    _write_document(directory / "calibration.json", report)
    write_manifest(
        {
            "stage": "empirical-calibrate-security",
            "codeql": config.as_dict(),
            "codeql_version": version,
            "bundle_archive_verified": bundle_archive is not None,
            "secure_samples_included": secure_samples is not None,
            "controls": len(controls),
            "totals": report["totals"],
            "mismatches": report["mismatches"],
            "configuration_sha256": digest_file(codeql),
            "cards_sha256": digest_file(cards),
            "calibration_sha256": digest_file(directory / "calibration.json"),
            "source": source_snapshot(),
        },
        directory,
    )
    detected, vulnerable = report["totals"]["vulnerable_detected_by_target_rules"]
    typer.echo(
        f"{len(controls)} controls; target queries report {detected} of {vulnerable} "
        f"vulnerable controls; {directory / 'calibration.json'}"
    )
    if report["mismatches"]:
        typer.echo(f"families whose observed state differs: {', '.join(report['mismatches'])}")
        raise typer.Exit(1)


@app.command("check-security")
def check_security(
    generations: Path,
    codeql: Annotated[Path, typer.Option(help="Frozen settings of the analyser.")],
    bundle_dir: Annotated[Path, typer.Option(help="Local copy of the unpacked bundle.")],
    cards: Annotated[Path, typer.Option(help="Scenario cards.")],
    calibration: Annotated[
        Path,
        typer.Option(help="Frozen calibration.json of calibrate-security backing the cards."),
    ],
    id_map: Annotated[Path, typer.Option(help="Identifier map giving the family of each task.")],
    measure: Annotated[str, typer.Option(help="general or targeted.")],
    directory: Annotated[Path, typer.Option(help="New directory for the outcomes.")],
    programs: Annotated[
        Path, typer.Option(help="Folder of the per-program analyses, shared by the measures.")
    ],
    analysis_id: Annotated[str, typer.Option(help="Identifier written in every outcome.")],
    order_seed: Annotated[int, typer.Option(help="Seed of the order of the analyses.")],
    sensitivity: Annotated[
        bool,
        typer.Option(help="Targeted measure: add the families with documented false positives."),
    ] = False,
    workers: Annotated[int, typer.Option(help="Programs analysed at the same time.")] = 1,
) -> None:
    """Judge LLMSecEval generations with CodeQL, one database per program.

    The general measure counts any alert of the default suite; the targeted
    one counts the queries of the weakness of the scenario, on the calibrated
    families only. Each measure gets its own outcomes and analysis identifier.
    """
    from pstparser.empirical import codeql as oracle
    from pstparser.empirical.analysis import write_outcomes
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    if measure not in oracle.MEASURES:
        raise ValueError(f"measure must be one of {', '.join(oracle.MEASURES)}")
    if sensitivity and measure != "targeted":
        raise ValueError("--sensitivity applies to the targeted measure only")
    config = oracle.read_config(codeql, bundle_dir)
    scenario_cards = oracle.read_cards(cards)
    # The family states come from a frozen calibration, not from the cards alone (R-ORA-10).
    oracle.require_calibration(_read_document(calibration), config, scenario_cards)
    families = {row["task_id"]: row["family_id"] for row in _read_document(id_map)["records"]}
    rows = _json_lines(generations)
    unknown = sorted({row["task_id"] for row in rows} - set(families))
    if unknown or not rows:
        raise ValueError(f"generations for tasks outside the identifier map: {unknown[:5]}")
    _require_complete(rows)
    rows, prefill = _with_prefill(rows, generations)
    without_card = sorted({families[row["task_id"]] for row in rows} - set(scenario_cards))
    if without_card:
        raise ValueError(f"families without a scenario card: {without_card[:5]}")
    ensure_stage_outputs(
        directory, "empirical-check-security", ["outcomes.jsonl", "details.jsonl", "attempts.jsonl"]
    )
    version = oracle.codeql_version(config)
    states = (oracle.CALIBRATED, oracle.CALIBRATED_FP) if sensitivity else (oracle.CALIBRATED,)
    result = oracle.evaluate_generations(
        config,
        scenario_cards,
        rows,
        families,
        directory,
        measure=measure,  # type: ignore[arg-type]
        analysis_id=analysis_id,
        order_seed=order_seed,
        workers=workers,
        programs=programs,
        states=states,
    )
    write_outcomes(result["outcomes"], directory / "outcomes.jsonl")
    _write_rows(directory / "details.jsonl", result["details"])
    _write_rows(directory / "attempts.jsonl", result["attempts"])
    write_manifest(
        {
            "stage": "empirical-check-security",
            "analysis_id": analysis_id,
            "assistant_prefill_prepended": prefill is not None,
            "codeql": config.as_dict(),
            "codeql_version": version,
            "family_states": list(states) if measure == "targeted" else None,
            **result["summary"],
            "generations_sha256": digest_file(generations),
            "configuration_sha256": digest_file(codeql),
            "cards_sha256": digest_file(cards),
            "calibration_sha256": digest_file(calibration),
            "id_map_sha256": digest_file(id_map),
            "outcomes_sha256": digest_file(directory / "outcomes.jsonl"),
            "details_sha256": digest_file(directory / "details.jsonl"),
            "attempts_sha256": digest_file(directory / "attempts.jsonl"),
            "source": source_snapshot(),
        },
        directory,
    )
    summary = result["summary"]
    typer.echo(
        f"{summary['generations']} generations, {summary['analysed']} analysed, "
        f"states {summary['states']}; {directory / 'outcomes.jsonl'}"
    )


def _load_tokenizer(snapshot: Path) -> Any:
    """Load the generator's tokenizer from a local snapshot, without the network."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)


@app.command()
def covariates(
    canonical: Path,
    variants: Path,
    rule: Annotated[str, typer.Option(help="Draft rule of the benchmark (judgement units).")],
    generate_config: Annotated[
        Path, typer.Option(help="Generate configuration: system message and wrapper.")
    ],
    tokenizer_snapshot: Annotated[
        Path, typer.Option(help="Local snapshot of the generator at the frozen revision.")
    ],
    out: Annotated[Path, typer.Option(help="New covariates file (JSON Lines).")],
) -> None:
    """Compute the frozen prompt covariates (M13, M25) before any generation.

    Token lengths count the whole model input (system message, wrapper and
    request, with the generation prompt) with the generator's tokenizer.
    """
    from pstparser.empirical.covariates import COVARIATES_VERSION, covariate_table
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    config = json.loads(generate_config.read_text(encoding="utf-8"))
    system = config["system_prompt"]
    template = config.get("user_template")
    if template is not None:
        validate_user_template(template)
    prompts = [
        CanonicalPrompt(
            row["task_id"],
            row["text"],
            tuple(SpanAnnotation(**span) for span in row["annotations"]),
            row["annotation_version"],
        )
        for row in _rows(canonical)
    ]
    rendered = [PromptVariant(**row) for row in _rows(variants)]
    ensure_stage_outputs(out.parent, "empirical-covariates", [out.name])
    tokenizer = _load_tokenizer(tokenizer_snapshot)

    def count_tokens(text: str) -> int:
        tokens = tokenizer.apply_chat_template(
            chat_messages(system, template, text),
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
        )
        return len(tokens)

    rows = covariate_table(prompts, rendered, count_tokens, rule)
    _write_rows(out, rows)
    write_manifest(
        {
            "stage": "empirical-covariates",
            "covariates_version": COVARIATES_VERSION,
            "rule": rule,
            "rows": len(rows),
            "canonical_sha256": digest_file(canonical),
            "variants_sha256": digest_file(variants),
            "generate_config_sha256": digest_file(generate_config),
            "tokenizer_files_sha256": {
                name: digest_file(tokenizer_snapshot / name)
                for name in ("tokenizer.json", "tokenizer_config.json")
                if (tokenizer_snapshot / name).is_file()
            },
            "covariates_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(rows)} tasks in {out}")


@app.command()
def contamination(
    generations: Path,
    sanitized: Path,
    dataset: Path,
    config: Annotated[Path, typer.Option(help="Frozen contamination configuration.")],
    tokenizer_snapshot: Annotated[
        Path, typer.Option(help="Local snapshot of the generator at the frozen revision.")
    ],
    out: Annotated[Path, typer.Option(help="New overlap file (JSON Lines).")],
) -> None:
    """Measure how much of each reference solution the oracle's programs reproduce.

    GENERATIONS is a journal, SANITIZED the programs given to the harness
    (``request_id``, ``sanitized``), DATASET the harness dataset that holds the
    reference solutions. Analysis only: run after the oracle, never before.
    """
    from pstparser.empirical.bigcodebench import read_harness_dataset
    from pstparser.empirical.contamination import contamination_rows, load_contamination_config
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    frozen = load_contamination_config(config)
    journal = _json_lines(generations)
    programs = {row["request_id"]: row["sanitized"] for row in _json_lines(sanitized)}
    if set(programs) != {row["request_id"] for row in journal} or len(programs) != len(journal):
        raise ValueError("the sanitized programs must cover the journal's requests once")
    solutions = {row["task_id"]: row["canonical_solution"] for row in read_harness_dataset(dataset)}
    ensure_stage_outputs(out.parent, "empirical-contamination", [out.name])
    tokenizer = _load_tokenizer(tokenizer_snapshot)

    def tokenize(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows = contamination_rows(
        [
            {
                "task_id": row["task_id"],
                "condition": row["condition"],
                "replicate_id": row["replicate_id"],
                "program": programs[row["request_id"]],
            }
            for row in journal
        ],
        solutions,
        tokenize,
        frozen,
    )
    _write_rows(out, rows)
    write_manifest(
        {
            "stage": "empirical-contamination",
            "config": {"n": frozen.n, "threshold": frozen.threshold, "tokenizer": frozen.tokenizer},
            "config_sha256": digest_file(config),
            "generations_sha256": digest_file(generations),
            "sanitized_sha256": digest_file(sanitized),
            "dataset_sha256": digest_file(dataset),
            "overlap_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(rows)} programs in {out}")


@app.command("near-duplicates")
def near_duplicates(
    sources: Path,
    out: Annotated[Path, typer.Option(help="New file of near-duplicate groups (JSON).")],
    threshold: Annotated[float, typer.Option(help="Jaccard threshold (03 B15).")] = 0.8,
) -> None:
    """Group BigCodeBench prompts that are near duplicates (S-NEARDUP clusters)."""
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.empirical.contamination import NEAR_DUPLICATE_RULE, near_duplicate_groups
    from pstparser.repro.manifest import digest_file, source_snapshot, write_manifest

    records = read_sources(sources)
    groups = near_duplicate_groups({r.task_id: r.text for r in records}, threshold)
    _write_document(
        out,
        {
            "format_version": 1,
            "rule": NEAR_DUPLICATE_RULE,
            "threshold": threshold,
            "tasks": len(records),
            "groups": groups,
        },
    )
    write_manifest(
        {
            "stage": "empirical-near-duplicates",
            "sources_sha256": digest_file(sources),
            "groups_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(groups)} groups, {sum(map(len, groups))} tasks in {out}")


@app.command()
def agreement(
    sources: Path,
    first: Path,
    second: Path,
    rule: Annotated[str, typer.Option(help="Draft rule of the benchmark (judgement units).")],
    first_date: Annotated[str, typer.Option(help="Date of the first annotation (YYYY-MM-DD).")],
    second_date: Annotated[
        str, typer.Option(help="Date of the re-annotation from scratch (YYYY-MM-DD).")
    ],
    out: Annotated[Path, typer.Option(help="New agreement report (JSON).")],
) -> None:
    """Character kappa and boundary F1 of two annotation files of the same tasks.

    The primary measures count only the judgement units; the measures on every
    non-whitespace character are descriptive. The second file is the
    re-annotation from scratch: no row may come from a rule, and the interval
    between the two annotations is recorded (plan of 18/11/2026, P-07).
    """
    from datetime import date

    from pstparser.empirical import annotations
    from pstparser.empirical.agreement import (
        AGREEMENT_VERSION,
        AgreementItem,
        boundary_f1,
        char_kappa,
    )
    from pstparser.empirical.benchmarks import read_sources
    from pstparser.empirical.covariates import (
        JUDGEMENT_UNITS_RULE,
        UNIT_TYPES,
        judgement_mask,
        unit_type_mask,
    )
    from pstparser.repro.manifest import digest_file, source_snapshot, write_manifest

    texts = {record.task_id: record for record in read_sources(sources)}
    left = {row.task_id: row for row in annotations.load_annotations(first)}
    right = {row.task_id: row for row in annotations.load_annotations(second)}
    ruled = sorted(task for task, row in right.items() if row.annotator.startswith("rule:"))
    if ruled:
        raise ValueError(f"the second file must be a human re-annotation; rule rows: {ruled[:3]}")
    interval_days = (date.fromisoformat(second_date) - date.fromisoformat(first_date)).days
    if interval_days < 0:
        raise ValueError("the second date comes before the first date")
    if set(left) != set(right) or not set(left) <= set(texts):
        raise ValueError("both files must annotate the same tasks of the sources")
    for task in left:
        if {left[task].source_sha256, right[task].source_sha256} != {texts[task].source_sha256}:
            raise ValueError(f"{task}: an annotation does not belong to this source")
    tasks = sorted(left)
    judged = [
        AgreementItem(
            task,
            texts[task].text,
            left[task].spans,
            right[task].spans,
            judgement_mask(texts[task].text, rule),
        )
        for task in tasks
    ]
    every = [AgreementItem(i.task_id, i.text, i.first, i.second) for i in judged]
    # Kappa per unit type (R-TRA-01): the first sentence is rule-fixed and only descriptive.
    by_type = {}
    for unit_type in UNIT_TYPES:
        typed = [
            AgreementItem(
                i.task_id, i.text, i.first, i.second, unit_type_mask(i.text, rule, unit_type)
            )
            for i in judged
        ]
        by_type[unit_type] = {
            "tasks_with_units": sum(1 for i in typed if i.mask is not None and any(i.mask)),
            "char_kappa": char_kappa(typed),
            "boundary_f1": boundary_f1(typed),
        }
    _write_document(
        out,
        {
            "agreement_version": AGREEMENT_VERSION,
            "judgement_units_rule": JUDGEMENT_UNITS_RULE,
            "first_date": first_date,
            "second_date": second_date,
            "interval_days": interval_days,
            "tasks": len(tasks),
            "judgement_units": {
                "char_kappa": char_kappa(judged),
                "boundary_f1": boundary_f1(judged),
            },
            "tasks_without_judgement_units": sum(
                1 for i in judged if i.mask is not None and not any(i.mask)
            ),
            "by_unit_type": by_type,
            "all_characters": {"char_kappa": char_kappa(every), "boundary_f1": boundary_f1(every)},
            "per_task_judgement_kappa": {item.task_id: char_kappa([item]) for item in judged},
        },
    )
    write_manifest(
        {
            "stage": "empirical-agreement",
            "sources_sha256": digest_file(sources),
            "first_sha256": digest_file(first),
            "second_sha256": digest_file(second),
            "report_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(str(out))


@app.command()
def freeze(
    spec: Annotated[Path, typer.Option(help="Freeze specification (freeze-spec-v1, YAML).")],
    out: Annotated[Path, typer.Option(help="New freeze file, inside <root>/protocol/.")],
    root: Annotated[Path, typer.Option(help="Repository root of the listed items.")] = Path(),
) -> None:
    """Freeze the protocol: hash every listed file, the package source and the lock file.

    The file is written once; a freeze that already exists is never replaced.
    """
    from pstparser.empirical.freeze import build_freeze, read_spec, verify_freeze, write_freeze
    from pstparser.repro.manifest import digest_file

    if out.resolve().parent != (root / "protocol").resolve():
        raise ValueError("the freeze must be written in <root>/protocol/")
    items, meta = read_spec(spec)
    write_freeze(build_freeze(root, items, meta), out)
    mismatches = verify_freeze(out, root)
    if mismatches:
        raise ValueError(f"the new freeze does not verify: {mismatches[:5]}")
    typer.echo(f"{len(items)} items frozen in {out} (sha256 {digest_file(out)})")


@app.command("freeze-verify")
def freeze_verify(
    freeze_file: Annotated[Path, typer.Argument(help="protocol-freeze.json to verify.")],
    root: Annotated[
        Path | None, typer.Option(help="Repository root; default: the parent of protocol/.")
    ] = None,
) -> None:
    """Verify a protocol freeze against the repository; exit with an error on any difference."""
    from pstparser.empirical.freeze import verify_freeze

    mismatches = verify_freeze(freeze_file, root)
    for mismatch in mismatches:
        typer.echo(f"{mismatch.kind}: {mismatch.path}", err=True)
    if mismatches:
        raise typer.Exit(code=1)
    typer.echo(f"{freeze_file}: verified")


_RUN_HELP = "Run directory holding run-identity.json (results/downstream/<run_id>)."
_OUT_HELP = "New directory for the files; it must not exist."


@app.command()
def tables(
    kind: Annotated[
        str,
        typer.Argument(
            help="samples, treatment, tokens, freeze, diagnostics, paired, security or parser."
        ),
    ],
    run: Annotated[Path, typer.Option(help=_RUN_HELP)],
    out: Annotated[Path, typer.Option(help=_OUT_HELP)],
    sample: Annotated[list[Path] | None, typer.Option(help="Sample file (samples).")] = None,
    exclusions: Annotated[Path | None, typer.Option(help="Frozen exclusions (samples).")] = None,
    variants: Annotated[list[Path] | None, typer.Option(help="Variants JSONL (treatment).")] = None,
    journal: Annotated[
        Path | None, typer.Option(help="generations.jsonl (tokens, diagnostics).")
    ] = None,
    details: Annotated[
        Path | None, typer.Option(help="details.jsonl of one check run (diagnostics).")
    ] = None,
    freeze_file: Annotated[
        Path | None, typer.Option("--freeze", help="protocol-freeze.json (freeze).")
    ] = None,
    report: Annotated[list[Path] | None, typer.Option(help="report.json (paired).")] = None,
    confirmatory: Annotated[
        list[str] | None,
        typer.Option(help="H1=<analysis_id>, and H3=<analysis_id> if H3 is in the family."),
    ] = None,
    outcomes: Annotated[
        list[Path] | None, typer.Option(help="Security outcomes.jsonl (security).")
    ] = None,
    score: Annotated[Path | None, typer.Option(help="Parser score results.json (parser).")] = None,
) -> None:
    """Write one kind of thesis table as CSV and LaTeX, with run and input hashes."""
    from pstparser.empirical.thesis import Outputs, build_tables, parse_confirmatory, write

    outputs = Outputs(run)
    build_tables(
        outputs,
        kind,
        samples=sample or (),
        exclusions=exclusions,
        variants=variants or (),
        journal=journal,
        details=details,
        freeze=freeze_file,
        reports=report or (),
        confirmatory=parse_confirmatory(confirmatory or ()) or None,
        outcomes=outcomes or (),
        score=score,
    )
    written = write(outputs, out)
    typer.echo(f"{len(written)} files in {out}")


@app.command()
def figures(
    kind: Annotated[str, typer.Argument(help="forest, effects, leaves or tokens.")],
    run: Annotated[Path, typer.Option(help=_RUN_HELP)],
    out: Annotated[Path, typer.Option(help=_OUT_HELP)],
    report: Annotated[
        list[Path] | None, typer.Option(help="report.json (forest: one or more).")
    ] = None,
    variants: Annotated[list[Path] | None, typer.Option(help="Variants JSONL (leaves).")] = None,
    journal: Annotated[Path | None, typer.Option(help="generations.jsonl (tokens).")] = None,
) -> None:
    """Draw one kind of thesis figure as PDF and PNG (needs the analysis extra)."""
    from pstparser.empirical.thesis import Outputs, build_figures, write

    outputs = Outputs(run)
    build_figures(outputs, kind, reports=report or (), variants=variants or (), journal=journal)
    written = write(outputs, out)
    typer.echo(f"{len(written)} files in {out}")


@app.command("thesis-tables")
def thesis_tables(
    run: Annotated[Path, typer.Option(help=_RUN_HELP)],
    out: Annotated[Path, typer.Option(help=_OUT_HELP)],
    protocol: Annotated[
        Path, typer.Option(help="Protocol directory with samples/ and oracles/.")
    ] = Path("protocol"),
    confirmatory: Annotated[
        list[str] | None,
        typer.Option(help="H1=<analysis_id>, and H3=<analysis_id> if H3 is in the family."),
    ] = None,
    gate: Annotated[
        Path | None,
        typer.Option(help="parser-gate report deciding H3 (default: parser/gate/gate.json)."),
    ] = None,
    purpose: Annotated[str, typer.Option(help=_PURPOSE_HELP)] = "exploratory",
    freeze: Annotated[Path | None, typer.Option(help=_FREEZE_HELP)] = None,
    family: Annotated[
        Path | None,
        typer.Option(
            help="Frozen confirmatory-family-v1 document (default: "
            "<protocol>/analysis/confirmatory-family.json, if present): H1 alone at 0.05, "
            "H3 exploratory. Only for the prespecified time cut; otherwise the gate decides."
        ),
    ] = None,
) -> None:
    """Every table and figure the files of a run allow; the skipped kinds go in the manifest.

    The parser gate decides whether H3 is in the confirmatory family. Only when
    the prespecified time cut applies (no complete parse by its deadline) does a
    frozen ``confirmatory-family-v1`` document fix the family to H1 alone; the
    gate then stays a descriptive report.
    """
    from pstparser.empirical.thesis import build_thesis, parse_confirmatory, write

    frozen = _purpose(purpose, freeze)
    if frozen["purpose"] == "final":
        # Thesis tables of a final run read only final analyses under this freeze (R-STA-04).
        for manifest in sorted((run / "analysis").glob("*/empirical-analysis_manifest.json")):
            upstream = _read_document(manifest)
            if (upstream.get("purpose"), upstream.get("freeze_sha256")) != (
                "final",
                frozen["freeze_sha256"],
            ):
                raise ValueError(f"{manifest}: not a final analysis under this freeze")
    if family is None and (protocol / "analysis" / "confirmatory-family.json").is_file():
        family = protocol / "analysis" / "confirmatory-family.json"
    if family is not None:
        from pstparser.empirical.tables import FAMILY_FORMAT

        if _read_document(family).get("format") != FAMILY_FORMAT:
            raise ValueError(f"{family}: not a {FAMILY_FORMAT} document")
        if freeze is not None:
            from pstparser.empirical.freeze import load_freeze
            from pstparser.repro.manifest import digest_file

            # The family is fixed by the protocol only if the freeze lists it.
            if digest_file(family) not in set(load_freeze(freeze)["items"].values()):
                raise ValueError("the confirmatory family document is not in the freeze")
        decider: Path | None = family
    else:
        if gate is None and (run / "parser" / "gate" / "gate.json").is_file():
            gate = run / "parser" / "gate" / "gate.json"
        if freeze is not None and gate is not None:
            from pstparser.empirical.freeze import load_freeze

            # The thresholds the gate was decided on must be the frozen ones (R-STA-01).
            thresholds = _read_document(gate).get("inputs", {}).get("thresholds_sha256")
            if thresholds not in set(load_freeze(freeze)["items"].values()):
                raise ValueError("the parser gate was not decided on thresholds the freeze lists")
        decider = gate
    outputs = build_thesis(run, protocol, parse_confirmatory(confirmatory or ()) or None, decider)
    written = write(outputs, out, frozen)
    for kind, reason in outputs.skipped.items():
        typer.echo(f"skipped {kind}: {reason}", err=True)
    typer.echo(f"{len(written)} files in {out}")


@app.command("auto-annotate")
def auto_annotate_command(
    predictions: Annotated[Path, typer.Argument(help="predictions.jsonl of parse-prompts.")],
    id_map: Annotated[Path, typer.Argument(help="id-map.json of the same parse-prompts run.")],
    sources: Annotated[Path, typer.Argument(help="Source prompts (task_id, text) JSONL.")],
    policy: Annotated[
        str,
        typer.Option(
            help="auto-strict-v1 (primary), auto-attach-v1 (sensitivity) or auto-strict-v2 "
            "(primary only by the C-22 rule)."
        ),
    ],
    out: Annotated[Path, typer.Option(help="New H-auto canonical JSONL.")],
    failures: Annotated[Path, typer.Option(help="New JSONL of the parser failures.")],
    s_parse: Annotated[Path, typer.Option(help="New S_parse document (s-parse-v1).")],
) -> None:
    """Convert parser predictions into H-auto spans; write S_parse before any generation.

    The conversion reads the source text and the raw prediction only, never the
    gold target stored beside the prediction.
    """
    from pstparser.empirical.auto import auto_annotate
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    directory = out.parent
    if failures.parent != directory or s_parse.parent != directory:
        raise ValueError("the three outputs belong in one directory")
    ensure_stage_outputs(
        directory, "empirical-auto-annotate", [out.name, failures.name, s_parse.name]
    )
    conversion = auto_annotate(
        _json_lines(predictions),
        _read_document(id_map),
        _rows(sources),
        policy=policy,
        inputs={
            "predictions_sha256": digest_file(predictions),
            "id_map_sha256": digest_file(id_map),
            "sources_sha256": digest_file(sources),
        },
    )
    _write_rows(
        out,
        [
            {
                "task_id": prompt.task_id,
                "text": prompt.text,
                "annotation_version": prompt.annotation_version,
                "annotations": [asdict(span) for span in prompt.annotations],
            }
            for prompt in conversion.canonical
        ],
    )
    _write_rows(failures, list(conversion.failures))
    _write_document(s_parse, conversion.s_parse)
    write_manifest(
        {
            "stage": "empirical-auto-annotate",
            "policy": policy,
            **conversion.s_parse["inputs"],
            "canonical_sha256": digest_file(out),
            "failures_sha256": digest_file(failures),
            "s_parse_sha256": digest_file(s_parse),
            "s_parse_tasks_sha256": conversion.s_parse["tasks_sha256"],
            "source": source_snapshot(),
        },
        directory,
    )
    document = conversion.s_parse
    typer.echo(
        f"{document['converted']}/{document['n']} converted; "
        f"failure rate {document['failure_rate']:.4f}; S_parse {document['tasks_sha256']}"
    )


@app.command("merge-ledgers")
def merge_ledgers_command(
    ledgers: Annotated[
        list[Path], typer.Argument(help="Outcome ledgers of the check runs of one session.")
    ],
    analysis_id: Annotated[str, typer.Option(help="analysis_id of the merged ledger.")],
    out: Annotated[Path, typer.Option(help="New merged ledger (JSON Lines).")],
) -> None:
    """Merge the core and extra ledgers of a session for hauto-ledgers (C-26), unchanged."""
    from pstparser.empirical.analysis import write_outcomes
    from pstparser.empirical.ledgers import merge_ledgers
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    merged = merge_ledgers([read_outcomes(path) for path in ledgers], analysis_id)
    ensure_stage_outputs(out.parent, "empirical-merge-ledgers", [out.name])
    if out.exists():
        raise FileExistsError(f"{out} exists: a merged ledger is never overwritten")
    write_outcomes(merged, out)
    write_manifest(
        {
            "stage": "empirical-merge-ledgers",
            "analysis_id": analysis_id,
            "ledgers_sha256": [digest_file(path) for path in ledgers],
            "records": len(merged),
            "merged_sha256": digest_file(out),
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"{len(merged)} records in {out}")


@app.command("hauto-ledgers")
def hauto_ledgers_command(
    outcomes: Annotated[Path, typer.Argument(help="Outcome ledger with T and H-auto rows.")],
    s_parse: Annotated[Path, typer.Option(help="S_parse document of the strict conversion.")],
    failures: Annotated[Path, typer.Option(help="Parser failures of the same conversion.")],
    directory: Annotated[Path, typer.Option(help="New directory for the two ledgers.")],
) -> None:
    """Write the ledgers of the two H3 estimates: on S_parse, and end-to-end."""
    from pstparser.empirical.analysis import write_outcomes
    from pstparser.empirical.auto import hauto_ledgers, load_s_parse
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    names = ["outcomes-s-parse.jsonl", "outcomes-end-to-end.jsonl"]
    ensure_stage_outputs(directory, "empirical-hauto-ledgers", names)
    primary, end_to_end = hauto_ledgers(
        read_outcomes(outcomes), load_s_parse(_read_document(s_parse)), _json_lines(failures)
    )
    write_outcomes(primary, directory / names[0])
    write_outcomes(end_to_end, directory / names[1])
    write_manifest(
        {
            "stage": "empirical-hauto-ledgers",
            "outcomes_sha256": digest_file(outcomes),
            "s_parse_sha256": digest_file(s_parse),
            "failures_sha256": digest_file(failures),
            "s_parse_ledger_sha256": digest_file(directory / names[0]),
            "end_to_end_ledger_sha256": digest_file(directory / names[1]),
            "source": source_snapshot(),
        },
        directory,
    )
    typer.echo(f"{len(primary)} and {len(end_to_end)} records in {directory}")


@app.command("parser-gate")
def parser_gate_command(
    s_parse: Annotated[
        Path, typer.Argument(help="S_parse of the primary strict policy on S_gold.")
    ],
    score: Annotated[Path, typer.Argument(help="score results.json of the same predictions.")],
    thresholds: Annotated[
        Path, typer.Option(help="Frozen thresholds: max_strict_failure, min_macro_f1.")
    ],
    out: Annotated[Path, typer.Option(help="New gate report.")],
    id_map: Annotated[
        Path, typer.Option(help="id-map.json of the parse-prompts run behind S_parse.")
    ],
    sample: Annotated[
        Path,
        typer.Option(
            help=f"Frozen sample of the parsed tasks; final: protocol/samples/{S_GOLD_FILE}."
        ),
    ],
    purpose: Annotated[str, typer.Option(help=_PURPOSE_HELP)] = "exploratory",
    freeze: Annotated[Path | None, typer.Option(help=_FREEZE_HELP)] = None,
    c22: Annotated[
        Path | None,
        typer.Option(
            help="S_parse of auto-strict-v1 on the pilot tasks (D-01): the C-22 rule fixes "
            "the primary policy. Required, and listed in the freeze, for a final gate."
        ),
    ] = None,
) -> None:
    """Decide the parser gate (00 T-7) before any H-auto generation.

    The score report, S_parse and the index map must come from the same
    predictions, and the tasks must be exactly those of the sample (R-HAU-02).
    """
    from pstparser.empirical.auto import parser_gate
    from pstparser.empirical.freeze import default_root
    from pstparser.repro.manifest import (
        digest_file,
        ensure_stage_outputs,
        source_snapshot,
        write_manifest,
    )

    frozen = _purpose(purpose, freeze)
    if purpose == "final":
        assert freeze is not None
        s_gold = default_root(freeze) / "protocol" / "samples" / S_GOLD_FILE
        if digest_file(sample) != digest_file(s_gold):
            raise ValueError(f"a final gate is measured on S_gold: protocol/samples/{S_GOLD_FILE}")
        if c22 is None:
            raise ValueError("a final gate needs --c22, the v1 conversion of the pilot tasks")
        from pstparser.empirical.freeze import load_freeze

        if digest_file(c22) not in set(load_freeze(freeze)["items"].values()):
            raise ValueError("the C-22 document is not in the freeze")
    ensure_stage_outputs(out.parent, "empirical-parser-gate", [out.name])
    sample_tasks = [row["task_id"] for row in _read_document(sample)["tasks"]]
    report = parser_gate(
        _read_document(s_parse),
        _read_document(score),
        _read_document(thresholds),
        id_map=_read_document(id_map),
        id_map_sha256=digest_file(id_map),
        sample_tasks=sample_tasks,
        c22=None if c22 is None else _read_document(c22),
    )
    report["tasks"] = {"n": len(sample_tasks), "sample_sha256": digest_file(sample)}
    report["inputs"] = {
        "s_parse_sha256": digest_file(s_parse),
        "score_sha256": digest_file(score),
        "thresholds_sha256": digest_file(thresholds),
        "id_map_sha256": digest_file(id_map),
        "sample_sha256": digest_file(sample),
        "c22_sha256": None if c22 is None else digest_file(c22),
    }
    _write_document(out, report)
    write_manifest(
        {
            "stage": "empirical-parser-gate",
            **report["inputs"],
            "gate_sha256": digest_file(out),
            "status": report["status"],
            **frozen,
            "source": source_snapshot(),
        },
        out.parent,
    )
    typer.echo(f"parser gate: {report['status']}; H3 {report['h3']}")
