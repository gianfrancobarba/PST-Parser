"""Generation configurations of the pilot and of the core (plan of 18/11/2026, P-04, P-08, P-11).

One file per journal. The BigCodeBench core has two journals in one session,
{T, H-gold, E, H-auto} on S_parse and {T, H-gold, E} on the other tasks, each
with and without F (F enters only by the CP3-F time rule); the pilot always has
F. Security: greedy and sampled, {T, H-gold}. Values from the frozen decisions:
SYS and wrapper W (00 T-1, T-2), greedy 1280 tokens (G-2), sampling 0.7/0.8/20
with replicates 1-5 (G-3), repetition_penalty 1.0 everywhere (G-4), one L4
(G-1), seed 7319 (decided by the student on 18/11/2026).
"""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.cli import app
from pstparser.empirical.journal import MERGE_FREE_KEYS

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "protocol" / "configs"
SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
W = (
    "Please provide a self-contained Python script that solves the following problem in a "
    "markdown code block:\n{prompt}\n"
)
GREEDY = {
    "max_new_tokens": 1280,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": None,
    "repetition_penalty": 1.0,
    "structured_output": False,
    "batch_size": 1,
}
SAMPLED = {**GREEDY, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
CORE = ["T", "H-gold", "E"]
FILES = {
    "bcb-greedy-hauto.json": [*CORE, "H-auto"],
    "bcb-greedy.json": CORE,
    "bcb-greedy-hauto-f.json": [*CORE, "F", "H-auto"],
    "bcb-greedy-f.json": [*CORE, "F"],
    "lse-greedy.json": ["T", "H-gold"],
    "lse-sampled.json": ["T", "H-gold"],
}
PILOT = {name: c for name, c in FILES.items() if name.startswith("lse") or "-f" in name}
REVIEW = {"final": "protocol-freeze.json", "pilot": "render-record.json"}


def load(purpose: str, name: str) -> dict:
    return json.loads((CONFIGS / purpose / name).read_text(encoding="utf-8"))


def every() -> list[tuple[str, str]]:
    return [("final", name) for name in FILES] + [("pilot", name) for name in PILOT]


def test_exactly_the_planned_files_exist() -> None:
    assert sorted(p.name for p in (CONFIGS / "final").iterdir()) == sorted(FILES)
    assert sorted(p.name for p in (CONFIGS / "pilot").iterdir()) == sorted(PILOT)


@pytest.mark.parametrize(("purpose", "name"), every())
def test_each_file_holds_the_frozen_values(purpose: str, name: str) -> None:
    config = load(purpose, name)
    assert config["conditions"] == FILES[name]
    assert config["purpose"] == purpose and config["review_record"] == REVIEW[purpose]
    assert config["system_prompt"] == SYS and config["user_template"] == W
    assert config["gpu"] == "NVIDIA L4" and config["seed"] == 7319
    assert config["inference"] == (SAMPLED if "sampled" in name else GREEDY)
    assert config["replicate_ids"] == ([1, 2, 3, 4, 5] if "sampled" in name else [0])
    assert config["model"] == {
        "backend": "hf",
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "max_seq_length": 8192,
        "load_in_4bit": False,
        "dtype": "bfloat16",
        "chat_template": "native",
    }
    benchmark = "bigcodebench" if name.startswith("bcb") else "llmseceval"
    assert config["dataset_revision"].startswith(benchmark)


@pytest.mark.parametrize("purpose", ["final", "pilot"])
def test_the_two_journals_of_a_session_differ_only_in_what_the_union_allows(purpose: str) -> None:
    pairs = [("bcb-greedy-hauto-f.json", "bcb-greedy-f.json")]
    if purpose == "final":
        pairs.append(("bcb-greedy-hauto.json", "bcb-greedy.json"))
    for first, second in pairs:
        a, b = load(purpose, first), load(purpose, second)
        assert {k for k in a if a[k] != b.get(k)} <= set(MERGE_FREE_KEYS)


def test_pilot_and_final_differ_only_in_purpose_and_review_record() -> None:
    for name in PILOT:
        final, pilot = load("final", name), load("pilot", name)
        assert {k for k in final if final[k] != pilot[k]} == {"purpose", "review_record"}


@pytest.mark.parametrize("name", sorted(PILOT))
def test_the_generate_command_accepts_each_pilot_file_up_to_the_gpu_check(
    tmp_path: Path, name: str
) -> None:
    # On a machine without the L4 every configuration check passes and the run
    # stops at the GPU check, before any model is loaded.
    shutil.copy(CONFIGS / "pilot" / name, tmp_path / "generate.json")
    (tmp_path / "render-record.json").write_text("{}", encoding="utf-8")
    (tmp_path / "variants.jsonl").write_text("", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "generate",
            str(tmp_path / "variants.jsonl"),
            str(tmp_path / "generate.json"),
            str(tmp_path / "run"),
        ],
    )
    assert isinstance(result.exception, ValueError)
    assert "requires the declared GPU" in str(result.exception)
