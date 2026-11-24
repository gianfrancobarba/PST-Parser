"""Protocol freeze and its verification."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pstparser.empirical.cli import app
from pstparser.empirical.freeze import (
    Mismatch,
    build_freeze,
    canonical_bytes,
    load_freeze,
    read_spec,
    require_final_freeze,
    verify_freeze,
    write_freeze,
)
from pstparser.repro.manifest import digest_file

META = {"protocol_version": "downstream-1.0.0", "frozen_by": "studente"}
ITEMS = ["protocol/samples/bcb-final.json", "protocol/configs/bcb-greedy.json", "data/x.jsonl"]
WHEN = "2026-11-20T10:00:00Z"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal repository: source with analysis code, lock file, protocol and data."""
    files = {
        "src/pstparser/__init__.py": "",
        "src/pstparser/empirical/analysis.py": "PRIMARY = 'mid-p'\n",
        "src/pstparser/empirical/inference.py": "WEIGHTS = (0.8, 0.2)\n",
        "uv.lock": "version = 1\n",
        "protocol/samples/bcb-final.json": '{"ids": [1, 2]}\n',
        "protocol/configs/bcb-greedy.json": '{"review_record": "../protocol-freeze.json"}\n',
        "data/x.jsonl": '{"task_id": "a"}\n',
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return tmp_path


def frozen(repo: Path, items: list[str] = ITEMS) -> Path:
    return write_freeze(
        build_freeze(repo, items, META, frozen_at=WHEN), repo / "protocol" / "protocol-freeze.json"
    )


def test_a_fresh_freeze_verifies_and_lists_every_hash(repo: Path) -> None:
    path = frozen(repo)
    document = load_freeze(path)
    assert verify_freeze(path) == []
    assert document["items"] == {name: digest_file(repo / name) for name in sorted(ITEMS)}
    assert document["environment"]["uv_lock_sha256"] == digest_file(repo / "uv.lock")
    assert (document["format"], document["frozen_at"], document["protocol_version"]) == (
        "protocol-freeze-v2",
        WHEN,
        "downstream-1.0.0",
    )


def test_serialisation_is_canonical_and_stable(repo: Path) -> None:
    first = build_freeze(repo, ITEMS, META, frozen_at=WHEN)
    second = build_freeze(repo, list(reversed(ITEMS)), META, frozen_at=WHEN)
    assert canonical_bytes(first) == canonical_bytes(second)
    data = canonical_bytes(first)
    assert data.endswith(b"}\n") and b"\r" not in data and b": " not in data
    assert json.loads(data) == first
    path = frozen(repo)
    assert path.read_bytes() == data
    # A freeze rewritten with another layout is refused, even with the same content.
    path.write_text(json.dumps(first, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonically serialised"):
        load_freeze(path)


def test_one_changed_byte_is_a_mismatch(repo: Path) -> None:
    path = frozen(repo)
    sample = repo / "protocol/samples/bcb-final.json"
    expected = digest_file(sample)
    sample.write_bytes(sample.read_bytes().replace(b"2", b"3"))
    assert verify_freeze(path) == [
        Mismatch("changed", "protocol/samples/bcb-final.json", expected, digest_file(sample))
    ]


def test_a_missing_file_is_reported_and_the_command_fails(repo: Path) -> None:
    path = frozen(repo)
    (repo / "data/x.jsonl").unlink()
    assert [(m.kind, m.path) for m in verify_freeze(path)] == [("missing", "data/x.jsonl")]
    result = CliRunner().invoke(app, ["freeze-verify", str(path)])
    assert result.exit_code == 1
    assert "missing: data/x.jsonl" in result.output


def test_the_freeze_covers_the_analysis_code_and_the_lock_file(repo: Path) -> None:
    path = frozen(repo)
    (repo / "src/pstparser/empirical/analysis.py").write_bytes(b"PRIMARY = 'exact'\n")
    assert [(m.kind, m.path) for m in verify_freeze(path)] == [("changed", "src/pstparser")]
    (repo / "src/pstparser/empirical/analysis.py").write_bytes(b"PRIMARY = 'mid-p'\n")
    (repo / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    assert [(m.kind, m.path) for m in verify_freeze(path)] == [("changed", "uv.lock")]


def test_files_of_protocol_left_out_of_the_freeze_are_reported(repo: Path) -> None:
    path = frozen(repo)
    (repo / "protocol/samples/extra.json").write_text("{}\n", encoding="utf-8")
    # Written with or after the freeze: never listed in it.
    (repo / "protocol/FREEZE-2026-11-20.md").write_text("# Freeze\n", encoding="utf-8")
    (repo / "protocol/deviations.md").write_text("# Deviations\n", encoding="utf-8")
    (repo / "protocol/protocol-freeze-hauto.json").write_text("{}\n", encoding="utf-8")
    (repo / "protocol/acks").mkdir()
    (repo / "protocol/acks/supervisor.eml").write_text("ok\n", encoding="utf-8")
    assert [(m.kind, m.path) for m in verify_freeze(path)] == [
        ("unlisted", "protocol/samples/extra.json")
    ]


def test_a_freeze_is_never_overwritten(repo: Path) -> None:
    frozen(repo)
    with pytest.raises(FileExistsError):
        frozen(repo)


@pytest.mark.parametrize(
    ("items", "meta", "message"),
    [
        (["data/x.jsonl", "data/x.jsonl"], META, "without repetitions"),
        (["data/absent.jsonl"], META, "no such file"),
        (["../outside.txt"], META, "relative paths inside"),
        (["data\\x.jsonl"], META, "relative paths inside"),
        (["data/x.jsonl"], {"protocol_version": "1"}, "frozen_by"),
        (["data/x.jsonl"], {**META, "surprise": 1}, "unknown metadata"),
    ],
)
def test_invalid_freezes_are_refused(repo: Path, items, meta, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_freeze(repo, items, meta)


def test_a_final_run_needs_a_freeze_listing_its_variants_and_configuration(repo: Path) -> None:
    config = repo / "protocol/configs/bcb-greedy.json"
    variants = repo / "data/x.jsonl"
    path = frozen(repo)
    assert require_final_freeze(path, variants, config)["items"]
    other = repo / "data/y.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not list the variant file"):
        require_final_freeze(path, other, config)
    review = repo / "review.md"
    review.write_text("reviewed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="require a protocol freeze"):
        require_final_freeze(review, variants, config)
    (repo / "data/x.jsonl").write_text('{"task_id": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"does not verify: changed data/x\.jsonl"):
        require_final_freeze(path, variants, config)


def test_the_freeze_command_reads_a_spec_and_writes_inside_protocol(repo: Path) -> None:
    spec = repo / "protocol" / "freeze-spec.yaml"
    spec.write_text(
        "format: freeze-spec-v1\nprotocol_version: downstream-1.0.0\nfrozen_by: studente\n"
        "conditions: {bigcodebench: [T, H-gold, E], llmseceval: [T, H-gold]}\nitems:\n"
        + "".join(f"  - {item}\n" for item in [*ITEMS, "protocol/freeze-spec.yaml"]),
        encoding="utf-8",
    )
    out = repo / "protocol" / "protocol-freeze.json"
    outside = CliRunner().invoke(
        app, ["freeze", "--spec", str(spec), "--out", str(repo / "f.json"), "--root", str(repo)]
    )
    assert isinstance(outside.exception, ValueError)
    result = CliRunner().invoke(
        app, ["freeze", "--spec", str(spec), "--out", str(out), "--root", str(repo)]
    )
    assert result.exit_code == 0, result.exception
    document = load_freeze(out)
    assert document["conditions"] == {
        "bigcodebench": ["T", "H-gold", "E"],
        "llmseceval": ["T", "H-gold"],
    }
    assert "protocol/freeze-spec.yaml" in document["items"]
    verified = CliRunner().invoke(app, ["freeze-verify", str(out)])
    assert verified.exit_code == 0 and "verified" in verified.output
    again = CliRunner().invoke(
        app, ["freeze", "--spec", str(spec), "--out", str(out), "--root", str(repo)]
    )
    assert isinstance(again.exception, FileExistsError)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("format: other\nitems: []\n", "expected a freeze-spec-v1"),
        ("format: freeze-spec-v1\n", "holds items"),
        (
            "format: freeze-spec-v1\nitems: [1]\nprotocol_version: x\nfrozen_by: y\n",
            "list of paths",
        ),
        ("format: freeze-spec-v1\nitems: []\nextra: 1\n", "holds items"),
    ],
)
def test_malformed_specs_are_refused(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_spec(path)
