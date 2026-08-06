"""Expansion of the corpus with prompts exercising the reasoning paradigms."""

from pstparser.synth.generate import (
    SynthesisResult,
    SyntheticPrompt,
    generate_prompts,
    load_seeds,
    summarise,
    write_annotation_file,
)
from pstparser.synth.providers import (
    ChatCompletionsProvider,
    Provider,
    ProviderError,
    provider_from_env,
)
from pstparser.synth.run import run_synthesis

__all__ = [
    "ChatCompletionsProvider",
    "Provider",
    "ProviderError",
    "SynthesisResult",
    "SyntheticPrompt",
    "generate_prompts",
    "load_seeds",
    "provider_from_env",
    "run_synthesis",
    "summarise",
    "write_annotation_file",
]
