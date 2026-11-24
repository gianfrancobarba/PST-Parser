import hashlib
import importlib.util
import shutil
import subprocess
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Reference values produced by git itself (`git hash-object -w`, `git mktree`):
# the tree holding only a.txt = "hello\n", and the empty tree.
HELLO_TREE = "2e81171448eb9f2ee3821e3d447aa6b2fe3ddba1"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_archive", ROOT / "scripts" / "verify_archive.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_script()


def make_zip(path: Path, members: dict[str, bytes], comment: bytes = b"") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.comment = comment
    return path


def test_crlf_text_member_is_rejected_and_named(tmp_path, capsys) -> None:
    archive = make_zip(tmp_path / "a.zip", {"src/a.py": b"x = 1\r\n", "src/b.py": b"y = 2\n"})
    assert verify.main([str(archive)]) == 1
    assert "CRLF found: src/a.py" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [".py", ".md", ".yaml", ".json", ".toml", ".jsonl", ".ipynb"])
def test_every_declared_text_suffix_is_checked_for_crlf(tmp_path, capsys, suffix) -> None:
    archive = make_zip(tmp_path / "a.zip", {f"d/x{suffix}": b"a\r\nb\n"})
    assert verify.main([str(archive)]) == 1
    assert f"CRLF found: d/x{suffix}" in capsys.readouterr().err


def test_lf_archive_passes_and_prints_key_hashes(tmp_path, capsys) -> None:
    prompt = b"line one\nline two\n"
    archive = make_zip(
        tmp_path / "a.zip",
        {
            "prompts/system_pst.md": prompt,
            "protocol/prompts/system.txt": b"s\n",
            "configs/_base/model/m.yaml": b"k: v\n",
            "src/a.py": b"x\n",
        },
    )
    assert verify.main([str(archive)]) == 0
    out = capsys.readouterr().out
    assert f"{hashlib.sha256(prompt).hexdigest()}  prompts/system_pst.md" in out
    assert f"{hashlib.sha256(b's' + bytes([10])).hexdigest()}  protocol/prompts/system.txt" in out
    assert "  configs/_base/model/m.yaml" in out
    assert "src/a.py" not in out


def test_binary_member_with_crlf_bytes_is_not_rejected(tmp_path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"data/raw/book.xlsx": b"PK\r\n\x00\x01"})
    assert verify.main([str(archive)]) == 0


def test_content_tree_matches_hashes_computed_by_git(tmp_path) -> None:
    with zipfile.ZipFile(make_zip(tmp_path / "one.zip", {"a.txt": b"hello\n"})) as archive:
        assert verify.content_tree(archive) == HELLO_TREE
    with zipfile.ZipFile(make_zip(tmp_path / "none.zip", {})) as archive:
        assert verify.content_tree(archive) == EMPTY_TREE


def test_content_tree_depends_on_bytes_and_not_on_member_order(tmp_path) -> None:
    members = {"src/a.py": b"x = 1\n", "src-b": b"y\n", "src.c": b"z\n"}
    with zipfile.ZipFile(make_zip(tmp_path / "a.zip", members)) as archive:
        original = verify.content_tree(archive)
    changed = members | {"src/a.py": b"x = 2\n"}
    with zipfile.ZipFile(make_zip(tmp_path / "b.zip", changed)) as archive:
        assert verify.content_tree(archive) != original
    reordered = dict(reversed(list(members.items())))
    with zipfile.ZipFile(make_zip(tmp_path / "c.zip", reordered)) as archive:
        assert verify.content_tree(archive) == original


def test_crlf_conversion_changes_the_tree(tmp_path) -> None:
    with zipfile.ZipFile(make_zip(tmp_path / "crlf.zip", {"a.txt": b"hello\r\n"})) as archive:
        assert verify.content_tree(archive) != HELLO_TREE


def test_expected_tree_is_compared_with_the_archive_contents(tmp_path, capsys) -> None:
    archive = make_zip(tmp_path / "a.zip", {"a.txt": b"hello\n"}, comment=b"f" * 40)
    assert verify.main([str(archive), "--expected-tree", "0" * 40]) == 1
    assert "tree mismatch" in capsys.readouterr().err
    assert verify.main([str(archive), "--expected-tree", HELLO_TREE]) == 0
    out = capsys.readouterr().out
    assert f"tree {HELLO_TREE}" in out
    assert f"commit {'f' * 40}" in out


def test_archive_without_a_commit_comment_reports_it(tmp_path, capsys) -> None:
    archive = make_zip(tmp_path / "a.zip", {"a.txt": b"hello\n"})
    assert verify.main([str(archive)]) == 0
    assert "commit not recorded" in capsys.readouterr().out


@pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="needs git and a repository checkout (not an archive extraction)",
)
def test_head_archive_holds_the_lf_blob_of_the_system_prompt(tmp_path) -> None:
    out = tmp_path / "head.zip"
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD"],
        cwd=ROOT,
        check=True,
    )
    with zipfile.ZipFile(out) as archive:
        digest = hashlib.sha256(archive.read("prompts/system_pst.md")).hexdigest()
        assert verify.find_crlf(archive) == []
        tree = verify.content_tree(archive)
        commit = verify.archive_commit(archive)
    blob = subprocess.run(
        ["git", "show", "HEAD:prompts/system_pst.md"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    assert digest == hashlib.sha256(blob).hexdigest()
    assert digest.startswith("787bf89d")
    assert not digest.startswith("e5cbb607")

    def rev_parse(name: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", name], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    assert tree == rev_parse("HEAD^{tree}")
    assert commit == rev_parse("HEAD")
