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


if __name__ == "__main__":
    app()
