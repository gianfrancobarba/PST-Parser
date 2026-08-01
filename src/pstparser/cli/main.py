"""Command line interface.

Commands are thin wrappers: they parse arguments, load the configuration and
delegate to the corresponding module. No pipeline logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from pstparser import __version__
from pstparser.config import ConfigError, ExperimentConfig, load_experiment
from pstparser.data import CorpusError, prepare_corpus

app = typer.Typer(
    name="pstparser",
    help="Fine-tuning and evaluation pipeline for Prompt Syntax Tree segmentation.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
error_console = Console(stderr=True)

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Experiment configuration file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]

SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override a configuration value, as dotted.key=value. Repeatable.",
        metavar="KEY=VALUE",
    ),
]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"pstparser {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Entry point for the pstparser command line."""


def _load(config: Path, overrides: list[str] | None) -> ExperimentConfig:
    """Load a configuration, reporting failures as a clean command line error."""
    try:
        return load_experiment(config, overrides=overrides)
    except ConfigError as exc:
        error_console.print(f"[bold red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        error_console.print(f"[bold red]Invalid configuration:[/]\n{exc}")
        raise typer.Exit(code=1) from exc


@app.command("validate-config")
def validate_config(config: ConfigOption, set_: SetOption = None) -> None:
    """Compose and validate a configuration without running anything."""
    experiment = _load(config, set_)

    table = Table(title=f"Experiment: {experiment.name}", show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("corpus", f"{experiment.data.source_path} [{experiment.data.sheet_name}]")
    table.add_row("leaves", str(len(experiment.data.column_mapping)))
    table.add_row("model", f"{experiment.model.name} ({experiment.model.backend})")
    table.add_row("adapter", f"r={experiment.lora.r}, alpha={experiment.lora.alpha}")
    table.add_row(
        "batch",
        f"{experiment.training.per_device_train_batch_size}"
        f" x {experiment.training.gradient_accumulation_steps}"
        f" = {experiment.training.effective_batch_size}",
    )
    table.add_row("steps", str(experiment.training.max_steps))
    table.add_row("learning rate", f"{experiment.training.learning_rate:g}")

    console.print(table)
    console.print("[bold green]Configuration is valid.[/]")


@app.command("prepare-data")
def prepare_data(config: ConfigOption, set_: SetOption = None) -> None:
    """Convert the annotated corpus into records and partition it."""
    experiment = _load(config, set_)

    try:
        result = prepare_corpus(experiment.data)
    except CorpusError as exc:
        error_console.print(f"[bold red]Corpus error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Prepared corpus", show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("records", str(len(result.records)))
    table.add_row("training", str(len(result.split.train)))
    table.add_row("evaluation", str(len(result.split.eval)))
    table.add_row("integrity issues", str(len(result.quality.issues)))
    table.add_row("records file", str(result.records_path))
    table.add_row("integrity report", str(result.report_path))
    table.add_row("split directory", str(result.split_dir))
    console.print(table)

    if result.quality.passed:
        console.print("[bold green]Integrity check passed.[/]")
        return

    console.print(
        f"[bold yellow]Integrity check reported {len(result.quality.issues)} records[/] "
        f"below a coverage ratio of {result.quality.min_coverage_ratio:g}."
    )
    for issue in result.quality.issues[:10]:
        console.print(f"  row {issue.index}: {issue.detail}")
    if len(result.quality.issues) > 10:
        console.print(f"  ... {len(result.quality.issues) - 10} more, see {result.report_path}")


@app.command("train")
def train(
    config: ConfigOption,
    set_: SetOption = None,
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Directory for the run artefacts.", dir_okay=True),
    ] = None,
) -> None:
    """Fine-tune adapters on the prepared corpus."""
    experiment = _load(config, set_)

    from pstparser.models import BackendError
    from pstparser.training import run_training

    try:
        outcome = run_training(experiment, run_dir=run_dir)
    except BackendError as exc:
        error_console.print(f"[bold red]Backend error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        error_console.print(f"[bold red]Missing input:[/] {exc}\nRun 'prepare-data' first.")
        raise typer.Exit(code=1) from exc

    share = outcome.trainable / outcome.total if outcome.total else 0.0
    table = Table(title="Training complete", show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("training examples", str(outcome.train_size))
    table.add_row("evaluation examples", str(outcome.eval_size))
    table.add_row(
        "trainable parameters", f"{outcome.trainable:,} of {outcome.total:,} ({share:.2%})"
    )
    table.add_row("adapter", str(outcome.adapter_dir))
    table.add_row("manifest", str(outcome.manifest_path))
    table.add_row("metrics", str(outcome.metrics_path))
    console.print(table)

    losses = [entry for entry in outcome.log_history if "eval_loss" in entry]
    if losses:
        best = min(losses, key=lambda entry: float(entry["eval_loss"]))
        console.print(
            f"lowest evaluation loss [bold]{float(best['eval_loss']):.4f}[/] "
            f"at step {int(best.get('step', 0))}"
        )


@app.command("generate")
def generate(
    config: ConfigOption,
    adapter: Annotated[
        Path,
        typer.Option("--adapter", "-a", help="Directory holding the trained adapter."),
    ],
    set_: SetOption = None,
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Directory for the predictions."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Stop after this many records.", min=1),
    ] = None,
) -> None:
    """Produce predictions for the evaluation split."""
    experiment = _load(config, set_)

    from pstparser.inference import run_generation
    from pstparser.models import BackendError

    try:
        outcome = run_generation(experiment, adapter_dir=adapter, run_dir=run_dir, limit=limit)
    except BackendError as exc:
        error_console.print(f"[bold red]Backend error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        error_console.print(f"[bold red]Missing input:[/] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Generation complete", show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("predictions", str(len(outcome.predictions)))
    table.add_row("predictions file", str(outcome.predictions_path))
    table.add_row("manifest", str(outcome.manifest_path))
    console.print(table)


@app.command("score")
def score(
    config: ConfigOption,
    predictions: Annotated[
        Path,
        typer.Option(
            "--predictions",
            "-p",
            help="File holding the predictions to score.",
            exists=True,
            dir_okay=False,
        ),
    ],
    set_: SetOption = None,
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Directory for the results."),
    ] = None,
) -> None:
    """Score predictions against their references. Runs without an accelerator."""
    experiment = _load(config, set_)

    from pstparser.data import load_predictions
    from pstparser.evaluation import evaluate, write_report

    records = load_predictions(predictions)
    if not records:
        error_console.print(f"[bold red]No predictions found in[/] {predictions}")
        raise typer.Exit(code=1)

    report = evaluate(
        predictions=[record.prediction for record in records],
        references=[record.target for record in records],
        prompts=[record.prompt for record in records],
        config=experiment.evaluation,
        indices=[record.index for record in records],
    )
    results_path, details_path = write_report(
        report,
        run_dir if run_dir is not None else predictions.parent,
        top_confusions=experiment.evaluation.confusion_top_k,
    )

    summary = Table(title="Evaluation", show_header=False, box=None)
    summary.add_column(style="cyan")
    summary.add_column(justify="right")
    summary.add_row("predictions", str(report.total))
    summary.add_row("JSON validity", f"{report.json_validity_rate:.2%}")
    summary.add_row("coverage", _format(report.mean_coverage_score, "{:.2%}"))
    summary.add_row("hallucination", _format(report.mean_hallucination_rate, "{:.2%}"))
    summary.add_row("tree edit distance", _format(report.mean_tree_edit_distance, "{:.2f}"))
    console.print(summary)

    if report.field_f1:
        fields = Table(title="Field F1", box=None)
        fields.add_column("field", style="cyan")
        fields.add_column("score", justify="right")
        for field, value in sorted(report.field_f1.items()):
            fields.add_row(field, f"{value:.4f}")
        console.print(fields)

    confusions = report.confusion.top_confusions(experiment.evaluation.confusion_top_k)
    if confusions:
        console.print("\n[bold]Most frequent misassignments[/]")
        for expected, predicted, count in confusions:
            console.print(f"  {count} x {expected} -> {predicted}")

    console.print(f"\nresults  {results_path}\ndetails  {details_path}")


def _format(value: float | None, template: str) -> str:
    """Render an optional metric, marking the absence of a value."""
    return template.format(value) if value is not None else "n/a"


if __name__ == "__main__":
    app()
