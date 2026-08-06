"""Corpus expansion, exercised against a stand-in provider."""

from __future__ import annotations

import http.client
import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from pstparser.config import SynthConfig
from pstparser.config.loader import load_experiment
from pstparser.data import read_annotations
from pstparser.pst import LEAF_PATHS
from pstparser.synth import (
    ProviderError,
    SyntheticPrompt,
    generate_prompts,
    load_seeds,
    provider_from_env,
    run_synthesis,
    write_annotation_file,
)
from pstparser.synth import providers as providers_module
from pstparser.synth import run as run_module

SEEDS = {
    "zero_shot_cot": ["Think through the migration before fixing it."],
    "tree_of_thoughts": ["Generate three options and choose one."],
}


class ScriptedProvider:
    """Returns queued answers in order, recording what it was asked."""

    def __init__(self, answers: list[str]) -> None:
        """Queue the answers the provider will hand out."""
        self.answers = list(answers)
        self.calls: list[tuple[str, str, float]] = []

    def complete(self, system: str, user: str, temperature: float) -> str:
        """Pop the next queued answer."""
        self.calls.append((system, user, temperature))
        return self.answers.pop(0) if self.answers else ""


class FailingProvider:
    """Always refuses."""

    def complete(self, system: str, user: str, temperature: float) -> str:
        """Raise, as a provider does when the service is unreachable."""
        raise ProviderError("unreachable")


class FlakyProvider:
    """Refuses a given number of times, then answers."""

    def __init__(self, failures: int, answers: list[str]) -> None:
        """Queue the refusals and the answers that follow them."""
        self.remaining = failures
        self.answers = list(answers)

    def complete(self, system: str, user: str, temperature: float) -> str:
        """Refuse while refusals remain, then pop the next answer."""
        if self.remaining > 0:
            self.remaining -= 1
            raise ProviderError("connection reset")
        return self.answers.pop(0) if self.answers else ""


class _Response:
    """A stand-in for the object ``urlopen`` returns."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *excinfo: object) -> bool:
        return False

    def read(self) -> bytes:
        """Return the body, as the real handle does."""
        return json.dumps(self.body).encode("utf-8")


def _answer(content: str) -> _Response:
    """Build a response carrying one completion."""
    return _Response({"choices": [{"message": {"content": content}}]})


def test_seeds_are_read_and_grouped() -> None:
    seeds = load_seeds("data/seeds/paradigms.yaml")

    assert set(seeds) == {"zero_shot_cot", "few_shot_cot", "tree_of_thoughts"}
    assert all(len(entries) == 10 for entries in seeds.values())
    assert all(text.strip() for entries in seeds.values() for text in entries)


def test_missing_seed_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="seed file not found"):
        load_seeds(tmp_path / "absent.yaml")


def test_seed_file_without_entries_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "seeds.yaml"
    path.write_text(yaml.safe_dump({"zero_shot_cot": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="has no seeds"):
        load_seeds(path)


def test_every_paradigm_receives_the_requested_count() -> None:
    provider = ScriptedProvider(["a", "b", "c", "d"])

    prompts, rejected = generate_prompts(SEEDS, provider, SynthConfig(per_paradigm=2))

    assert len(prompts) == 4
    assert rejected == 0
    assert len(provider.calls) == 4


def test_generated_prompts_carry_their_paradigm() -> None:
    provider = ScriptedProvider(["one", "two"])

    prompts, _ = generate_prompts(SEEDS, provider, SynthConfig(per_paradigm=1))

    assert [prompt.paradigm for prompt in prompts] == ["zero_shot_cot", "tree_of_thoughts"]


def test_request_shows_the_seeds_of_that_paradigm() -> None:
    provider = ScriptedProvider(["one", "two"])

    generate_prompts(SEEDS, provider, SynthConfig(per_paradigm=1))

    first_request = provider.calls[0][1]
    assert "zero shot cot" in first_request
    assert SEEDS["zero_shot_cot"][0] in first_request
    assert SEEDS["tree_of_thoughts"][0] not in first_request


def test_temperature_is_passed_through() -> None:
    provider = ScriptedProvider(["one", "two"])

    generate_prompts(SEEDS, provider, SynthConfig(per_paradigm=1, temperature=0.4))

    assert all(call[2] == pytest.approx(0.4) for call in provider.calls)


def test_duplicate_completions_are_discarded_and_asked_for_again() -> None:
    provider = ScriptedProvider(["same", "SAME", "other", "x"])

    prompts, rejected = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=3)
    )

    # The repeat costs a request, not a prompt: three were asked for, three came.
    assert [prompt.text for prompt in prompts] == ["same", "other", "x"]
    assert rejected == 1


def test_completion_repeating_a_seed_is_discarded() -> None:
    provider = ScriptedProvider([SEEDS["zero_shot_cot"][0], "fresh"])

    prompts, rejected = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=1)
    )

    assert [prompt.text for prompt in prompts] == ["fresh"]
    assert rejected == 1


def test_empty_completion_is_discarded() -> None:
    provider = ScriptedProvider(["", "   ", "kept"])

    prompts, rejected = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=1)
    )

    assert [prompt.text for prompt in prompts] == ["kept"]
    assert rejected == 2


def test_the_requested_count_is_a_target_not_a_budget() -> None:
    # Every other answer repeats the one before it, so reaching four prompts
    # takes seven requests.
    provider = ScriptedProvider(["a", "a", "b", "b", "c", "c", "d"])

    prompts, rejected = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=4)
    )

    assert [prompt.text for prompt in prompts] == ["a", "b", "c", "d"]
    assert rejected == 3
    assert len(provider.calls) == 7


def test_a_provider_that_repeats_itself_stops_at_the_attempt_cap() -> None:
    provider = ScriptedProvider(["same"] * 100)

    prompts, _ = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=5)
    )

    assert [prompt.text for prompt in prompts] == ["same"]
    assert len(provider.calls) == 5 * 3


def test_an_occasional_failure_costs_a_request_and_not_the_run() -> None:
    provider = FlakyProvider(failures=2, answers=["first", "second"])

    prompts, rejected = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=2)
    )

    assert [prompt.text for prompt in prompts] == ["first", "second"]
    assert rejected == 2


def test_a_provider_that_has_stopped_answering_ends_the_run() -> None:
    with pytest.raises(ProviderError, match="failed in a row"):
        generate_prompts(SEEDS, FailingProvider(), SynthConfig(per_paradigm=50))


def test_marks_enclosing_the_whole_answer_are_removed() -> None:
    provider = ScriptedProvider(['"Refactor this loop."'])

    prompts, _ = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=1)
    )

    assert [prompt.text for prompt in prompts] == ["Refactor this loop."]


def test_marks_inside_the_answer_are_left_alone() -> None:
    quoted = '"Top product" means the largest quantity. Explain the query below.'
    provider = ScriptedProvider([quoted])

    prompts, _ = generate_prompts(
        {"zero_shot_cot": SEEDS["zero_shot_cot"]}, provider, SynthConfig(per_paradigm=1)
    )

    assert [prompt.text for prompt in prompts] == [quoted]


def test_requests_do_not_all_show_the_same_seeds() -> None:
    paradigm = {"zero_shot_cot": [f"seed number {n}" for n in range(10)]}
    provider = ScriptedProvider([f"generated {n}" for n in range(20)])

    generate_prompts(paradigm, provider, SynthConfig(per_paradigm=20))

    requests = {call[1] for call in provider.calls}
    assert len(requests) > 1
    # A request carries a handful of the seeds, never all ten.
    shown = [sum(seed in call[1] for seed in paradigm["zero_shot_cot"]) for call in provider.calls]
    assert set(shown) == {4}


def test_generated_prompts_are_written_in_the_format_preparation_reads(tmp_path: Path) -> None:
    prompts = [
        SyntheticPrompt(paradigm="zero_shot_cot", text="first"),
        SyntheticPrompt(paradigm="tree_of_thoughts", text="second"),
    ]

    path = write_annotation_file(prompts, tmp_path / "nested" / "annotation.yaml")
    records = read_annotations(path)

    assert [record.prompt for record in records] == ["first", "second"]
    assert [record.paradigm for record in records] == ["zero_shot_cot", "tree_of_thoughts"]
    assert all(set(record.leaves) == set(LEAF_PATHS) for record in records)
    # The leaves are left for a human to fill in.
    assert all(not any(record.leaves.values()) for record in records)


def test_missing_credential_is_reported() -> None:
    with pytest.raises(ProviderError, match="ABSENT_VARIABLE is not set"):
        provider_from_env("https://example.invalid/v1", "some/model", "ABSENT_VARIABLE")


def test_credential_is_read_with_the_configured_transport_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTH_TEST_KEY", "secret")

    provider = provider_from_env(
        "https://example.invalid/v1", "some/model", "SYNTH_TEST_KEY", timeout=7.5, attempts=9
    )

    assert provider.timeout == pytest.approx(7.5)
    assert provider.attempts == 9


def test_a_connection_closed_midway_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    # RemoteDisconnected is a reset socket and a malformed status line at once,
    # and belongs to neither of the families the obvious clauses name. It ended
    # a whole generation before the boundary was sealed.
    seen: list[float] = []

    def urlopen(request: object, timeout: float) -> _Response:
        seen.append(timeout)
        if len(seen) == 1:
            raise http.client.RemoteDisconnected("closed without response")
        return _answer("recovered")

    monkeypatch.setattr(providers_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(providers_module, "RETRY_BACKOFF_SECONDS", 0.0)
    provider = providers_module.ChatCompletionsProvider(
        "https://example.invalid/v1", "some/model", "secret", timeout=3.0
    )

    assert provider.complete("system", "user", 1.0) == "recovered"
    assert seen == [3.0, 3.0]


def test_a_connection_that_never_recovers_is_reported_as_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def urlopen(request: object, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        raise http.client.RemoteDisconnected("closed without response")

    monkeypatch.setattr(providers_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(providers_module, "RETRY_BACKOFF_SECONDS", 0.0)
    provider = providers_module.ChatCompletionsProvider(
        "https://example.invalid/v1", "some/model", "secret", attempts=3
    )

    with pytest.raises(ProviderError, match="RemoteDisconnected"):
        provider.complete("system", "user", 1.0)
    assert calls == 3


def test_a_failure_no_clause_names_still_leaves_as_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The point of the guard is that the list of transport failures is never
    # complete: whatever arrives, one request costs one prompt, not the run.
    def urlopen(request: object, timeout: float) -> _Response:
        raise RuntimeError("something the clauses do not name")

    monkeypatch.setattr(providers_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(providers_module, "RETRY_BACKOFF_SECONDS", 0.0)
    provider = providers_module.ChatCompletionsProvider(
        "https://example.invalid/v1", "some/model", "secret", attempts=1
    )

    with pytest.raises(ProviderError, match="RuntimeError"):
        provider.complete("system", "user", 1.0)


def test_a_refusal_the_service_would_repeat_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def urlopen(request: object, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        raise providers_module.urllib.error.HTTPError(
            "https://example.invalid/v1", 401, "Unauthorized", {}, io.BytesIO(b"bad key")
        )

    monkeypatch.setattr(providers_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(providers_module, "RETRY_BACKOFF_SECONDS", 0.0)
    provider = providers_module.ChatCompletionsProvider(
        "https://example.invalid/v1", "some/model", "secret", attempts=3
    )

    with pytest.raises(ProviderError, match="HTTP 401"):
        provider.complete("system", "user", 1.0)
    assert calls == 1


def test_run_writes_sheet_summary_and_manifest(tmp_path: Path) -> None:
    config = load_experiment(
        "configs/experiments/baseline.yaml",
        overrides=[
            f"synth.output_dir={(tmp_path / 'synthetic').as_posix()}",
            "synth.per_paradigm=1",
        ],
    )
    provider = ScriptedProvider([f"generated {n}" for n in range(10)])

    result = run_synthesis(config, provider=provider)

    assert result.requested == 3
    assert len(result.prompts) == 3
    assert result.per_paradigm() == {
        "zero_shot_cot": 1,
        "few_shot_cot": 1,
        "tree_of_thoughts": 1,
    }
    assert result.sheet_path.is_file()

    summary = json.loads((tmp_path / "synthetic" / "synthesis_summary.json").read_text("utf-8"))
    assert summary["accepted"] == 3

    manifest = json.loads((tmp_path / "synthetic" / "run_manifest.json").read_text("utf-8"))
    assert manifest["stage"] == "synth"
    assert manifest["inputs"]["seeds"]
    assert manifest["synthesis"]["accepted"] == 3


def test_run_builds_the_provider_the_configuration_describes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The two transport options were validated and recorded in the manifest
    # while never reaching the provider, which no test could see.
    captured: dict[str, object] = {}

    def build(**kwargs: object) -> ScriptedProvider:
        captured.update(kwargs)
        return ScriptedProvider([f"generated {n}" for n in range(10)])

    monkeypatch.setattr(run_module, "provider_from_env", build)
    config = load_experiment(
        "configs/experiments/baseline.yaml",
        overrides=[
            f"synth.output_dir={(tmp_path / 'synthetic').as_posix()}",
            "synth.per_paradigm=1",
            "synth.provider.timeout=7.5",
            "synth.provider.attempts=9",
        ],
    )

    run_synthesis(config)

    assert captured["timeout"] == pytest.approx(7.5)
    assert captured["attempts"] == 9
    assert captured["api_key_env"] == "NVIDIA_API_KEY"
