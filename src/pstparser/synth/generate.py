"""Expansion of the corpus with prompts exercising the reasoning paradigms.

The corpus drawn from real conversations is dominated by direct requests: the
branches of the taxonomy that describe reasoning are almost unattested. This
stage produces additional prompts for those paradigms, using hand-written seeds
as templates.

What comes out are *prompts*, not annotations. The result is an annotation sheet
with the prompt column filled and the rest empty, ready for a human to label.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from pstparser.config import SynthConfig
from pstparser.data.annotations import write_annotation_skeleton
from pstparser.synth.providers import Provider, ProviderError

#: How many requests may be spent for each prompt asked for, before a paradigm
#: is left short. It is what keeps the count a target without letting a
#: provider that repeats itself run forever.
ATTEMPT_RATIO: Final = 3

#: How many seeds are shown in one request. Fewer than the paradigm holds, so
#: that no two requests are quite the same.
SEEDS_PER_REQUEST: Final = 4

#: How many requests may fail in a row before a run gives up. A provider that
#: has refused several times running is down rather than unlucky, and going on
#: only spends the retry backoff to arrive at nothing.
CONSECUTIVE_FAILURES: Final = 5

#: Pairs of marks a model wraps an answer in when asked for text and nothing
#: else. They belong to no prompt, and an annotator copying them in would
#: record an extraction that cannot be found in its own source. The curly pairs
#: are written as escapes, being indistinguishable from their straight
#: counterparts in a source file.
QUOTE_PAIRS: Final = (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"))

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
3. Reproduce their structure, not only their intent. Where the examples carry
   worked demonstrations, carry worked demonstrations too, showing the
   intermediate reasoning and keeping the labels they use. Where they set out
   alternatives to choose between, set out alternatives. A prompt that keeps
   only the closing question has not used the paradigm.
4. Write the way a developer writes: concrete, occasionally terse, sometimes
   including a snippet or an error message.
5. Keep it under two hundred words.
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

    The configured count is how many prompts are wanted, not how many requests
    are spent: a completion that is empty, or that repeats a seed or an earlier
    completion, is discarded and asked for again. Each paradigm may spend
    :data:`ATTEMPT_RATIO` requests per prompt before being left short, which is
    what keeps a provider that has run out of things to say from looping.

    Every request shows a different handful of the paradigm's seeds, drawn from
    the global generator that the run seeds from the experiment. Showing all of
    them every time made fifty identical requests and asked temperature alone
    for the variety.

    Args:
        seeds: Seed prompts, keyed by paradigm.
        provider: The service producing the completions.
        config: Generation options.

    Returns:
        The accepted prompts and the number of discarded completions.

    Raises:
        ProviderError: If :data:`CONSECUTIVE_FAILURES` requests fail in a row,
            which is a service that is down rather than one that is flaky.
    """
    accepted: list[SyntheticPrompt] = []
    rejected = 0
    failures = 0

    for paradigm, examples in seeds.items():
        seen = {_fingerprint(example) for example in examples}
        produced = 0
        budget = config.per_paradigm * ATTEMPT_RATIO

        while produced < config.per_paradigm and budget > 0:
            budget -= 1
            instruction = _instruction(paradigm, _sample(examples))
            try:
                answer = provider.complete(SYSTEM_PROMPT, instruction, config.temperature)
            except ProviderError as exc:
                rejected += 1
                failures += 1
                if failures >= CONSECUTIVE_FAILURES:
                    raise ProviderError(
                        f"gave up after {failures} requests failed in a row: {exc}"
                    ) from exc
                continue

            failures = 0
            text = _unquote(answer.strip())
            fingerprint = _fingerprint(text)
            if not fingerprint or fingerprint in seen:
                rejected += 1
                continue

            seen.add(fingerprint)
            accepted.append(SyntheticPrompt(paradigm=paradigm, text=text))
            produced += 1

    return accepted, rejected


def write_annotation_file(prompts: Sequence[SyntheticPrompt], destination: str | Path) -> Path:
    """Write the generated prompts as a file awaiting annotation.

    The layout is the one the preparation reads, so what gets filled in here is
    fed back without being copied between formats. Copying is where an
    extraction stops being exact, and the check downstream would then report a
    transcription slip as an annotation error.

    Args:
        prompts: The prompts to write, in order.
        destination: File to write. Parent directories are created.

    Returns:
        The path that was written.
    """
    return write_annotation_skeleton(
        ((prompt.paradigm, prompt.text) for prompt in prompts), destination
    )


def _fingerprint(text: str) -> str:
    """Reduce a prompt to a form suitable for equality checks."""
    return " ".join(text.casefold().split())


def _sample(examples: Sequence[str]) -> list[str]:
    """Pick the seeds shown in one request.

    Args:
        examples: Every seed of the paradigm.

    Returns:
        A subset of them, or all of them when there are no more than
        :data:`SEEDS_PER_REQUEST`.
    """
    if len(examples) <= SEEDS_PER_REQUEST:
        return list(examples)
    return random.sample(list(examples), SEEDS_PER_REQUEST)


def _unquote(text: str) -> str:
    """Remove one pair of marks enclosing the whole answer.

    The pair is only removed when the closing mark occurs nowhere else, so a
    prompt that genuinely opens and closes on quoted passages is left alone.

    Args:
        text: The answer as the provider returned it.

    Returns:
        The answer without the marks that enclose all of it.
    """
    for opening, closing in QUOTE_PAIRS:
        if len(text) > len(opening) + len(closing) and text.startswith(opening):
            inner = text[len(opening) :]
            if inner.endswith(closing) and closing not in inner[: -len(closing)]:
                return inner[: -len(closing)].strip()
    return text


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
