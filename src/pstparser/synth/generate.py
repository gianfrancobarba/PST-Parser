"""Expansion of the corpus with prompts exercising the reasoning paradigms.

The corpus drawn from real conversations is dominated by direct requests: the
branches of the taxonomy that describe reasoning are almost unattested. This
stage produces additional prompts for those paradigms, using hand-written seeds
as templates.

What comes out are *prompts*, not annotations. The result is an annotation sheet
with the prompt column filled and the rest empty, ready for a human to label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pstparser.config import SynthConfig
from pstparser.pst.taxonomy import LEAF_PATHS
from pstparser.synth.providers import Provider, ProviderError

#: Instruction given to the generating model.
SYSTEM_PROMPT = """\
You write realistic prompts that software developers send to a coding assistant.

You will be shown several example prompts that all use the same reasoning
paradigm. Write one new prompt that uses the same paradigm, on a different
subject.

Rules:
1. Return only the prompt itself. No preamble, no quotes, no explanation.
2. Match the paradigm of the examples, not their topic. Do not reuse their
   subject matter.
3. Write the way a developer writes: concrete, occasionally terse, sometimes
   including a snippet or an error message.
4. Keep it under two hundred words.
"""


@dataclass(frozen=True)
class SyntheticPrompt:
    """One generated prompt.

    Attributes:
        paradigm: The reasoning paradigm it was generated for.
        text: The prompt itself.
    """

    paradigm: str
    text: str


@dataclass(frozen=True)
class SynthesisResult:
    """Outcome of a generation run.

    Attributes:
        prompts: The accepted prompts, grouped in generation order.
        requested: How many prompts were asked for in total.
        rejected: How many completions were discarded as empty or duplicated.
        sheet_path: The annotation sheet that was written.
    """

    prompts: list[SyntheticPrompt]
    requested: int
    rejected: int
    sheet_path: Path

    def per_paradigm(self) -> dict[str, int]:
        """Count the accepted prompts of each paradigm."""
        counts: dict[str, int] = {}
        for prompt in self.prompts:
            counts[prompt.paradigm] = counts.get(prompt.paradigm, 0) + 1
        return counts


def load_seeds(path: str | Path) -> dict[str, list[str]]:
    """Read the seed prompts, grouped by paradigm.

    Args:
        path: YAML file mapping each paradigm to its seeds.

    Returns:
        The seeds, keyed by paradigm.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a mapping of paradigm to list of strings,
            or if a paradigm has no seeds.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"seed file not found: {path}")

    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"seed file must map paradigms to lists of prompts: {path}")

    seeds: dict[str, list[str]] = {}
    for paradigm, entries in content.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"paradigm {paradigm!r} has no seeds in {path}")
        seeds[str(paradigm)] = [str(entry).strip() for entry in entries]
    return seeds


def generate_prompts(
    seeds: Mapping[str, Sequence[str]],
    provider: Provider,
    config: SynthConfig,
) -> tuple[list[SyntheticPrompt], int]:
    """Ask the provider for new prompts in each paradigm.

    Completions that are empty, or that repeat a seed or an earlier completion,
    are discarded rather than retried, so a run always terminates.

    Args:
        seeds: Seed prompts, keyed by paradigm.
        provider: The service producing the completions.
        config: Generation options.

    Returns:
        The accepted prompts and the number of discarded completions.
    """
    accepted: list[SyntheticPrompt] = []
    rejected = 0

    for paradigm, examples in seeds.items():
        seen = {_fingerprint(example) for example in examples}
        instruction = _instruction(paradigm, examples)

        for _ in range(config.per_paradigm):
            try:
                text = provider.complete(SYSTEM_PROMPT, instruction, config.temperature).strip()
            except ProviderError:
                rejected += 1
                continue

            fingerprint = _fingerprint(text)
            if not fingerprint or fingerprint in seen:
                rejected += 1
                continue

            seen.add(fingerprint)
            accepted.append(SyntheticPrompt(paradigm=paradigm, text=text))

    return accepted, rejected


def write_annotation_sheet(
    prompts: Sequence[SyntheticPrompt],
    destination: str | Path,
    column_mapping: Mapping[str, str],
    prompt_column: str,
    sheet_name: str,
) -> Path:
    """Write the generated prompts as a spreadsheet ready for annotation.

    The layout matches the annotated corpus, so the completed sheet can be fed
    straight back into the preparation stage.

    Args:
        prompts: The prompts to write, one per row.
        destination: File to write. Parent directories are created.
        column_mapping: Mapping from leaf path to column name, used to lay out
            the empty annotation columns.
        prompt_column: Name of the column holding the prompt.
        sheet_name: Name given to the worksheet.

    Returns:
        The path that was written.
    """
    from openpyxl import Workbook

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    headers = [prompt_column, *(column_mapping[path] for path in LEAF_PATHS)]
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append(headers)
    worksheet.freeze_panes = "A2"

    for prompt in prompts:
        worksheet.append([prompt.text, *([None] * len(LEAF_PATHS))])

    workbook.save(destination)
    return destination


def _fingerprint(text: str) -> str:
    """Reduce a prompt to a form suitable for equality checks."""
    return " ".join(text.casefold().split())


def _instruction(paradigm: str, examples: Sequence[str]) -> str:
    """Build the request shown to the generating model."""
    numbered = "\n\n".join(f"Example {n}:\n{text}" for n, text in enumerate(examples, start=1))
    readable = paradigm.replace("_", " ")
    return f"Paradigm: {readable}\n\n{numbered}\n\nWrite one new prompt using this paradigm."


def summarise(result: SynthesisResult) -> dict[str, Any]:
    """Return a JSON-serialisable summary of a generation run.

    Args:
        result: The outcome to summarise.

    Returns:
        A mapping suitable for logging or serialisation.
    """
    return {
        "requested": result.requested,
        "accepted": len(result.prompts),
        "rejected": result.rejected,
        "per_paradigm": result.per_paradigm(),
        "sheet": str(result.sheet_path),
    }
