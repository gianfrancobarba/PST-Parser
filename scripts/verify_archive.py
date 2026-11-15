"""Check that a source archive carries LF text and report the hashes of key files.

Usage: verify_archive.py ARCHIVE [--expected-tree <git tree hash>]

Exit status is 0 only when no text member contains CRLF and, if requested, the
git tree hash recomputed from the archive members equals the expected one. The
tree is derived from the bytes in the archive, not from its comment, so the
check needs neither git nor the repository and holds on any machine.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import stat
import sys
import zipfile
from pathlib import Path

TEXT_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".ipynb")
KEY_PATTERNS = (
    "prompts/*",
    "protocol/prompts/*",
    "protocol/configs/*",
    "configs/*",
    "*variants*.jsonl",
)


def find_crlf(archive: zipfile.ZipFile) -> list[str]:
    """Return the text members that contain a CRLF line ending."""
    return [
        info.filename
        for info in archive.infolist()
        if info.filename.endswith(TEXT_SUFFIXES) and b"\r\n" in archive.read(info)
    ]


def key_hashes(archive: zipfile.ZipFile) -> dict[str, str]:
    """Return the SHA-256 of the archive members that identify a run."""
    return {
        info.filename: hashlib.sha256(archive.read(info)).hexdigest()
        for info in archive.infolist()
        if not info.is_dir() and any(fnmatch.fnmatch(info.filename, p) for p in KEY_PATTERNS)
    }


def archive_commit(archive: zipfile.ZipFile) -> str | None:
    """Return the commit that ``git archive`` recorded in the ZIP comment, if any."""
    comment = archive.comment.decode("ascii", "replace").strip()
    return comment if re.fullmatch(r"[0-9a-f]{40}", comment) else None


def _git_object(kind: bytes, body: bytes) -> bytes:
    return hashlib.sha1(kind + b" %d\x00" % len(body) + body, usedforsecurity=False).digest()


def _entry_mode(info: zipfile.ZipInfo) -> bytes:
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        return b"120000"
    return b"100755" if mode & 0o111 else b"100644"


def content_tree(archive: zipfile.ZipFile) -> str:
    """Return the git tree hash (SHA-1) of the files stored in the archive.

    It equals ``git rev-parse <commit>^{tree}`` exactly when every member holds
    the bytes of its blob, which is what an LF archive of that commit contains.
    """
    root: dict[str, object] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        *parents, name = info.filename.split("/")
        node = root
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"{info.filename}: {part} is both a file and a directory")
            node = child
        if name in node:
            raise ValueError(f"duplicate archive member: {info.filename}")
        node[name] = (_entry_mode(info), _git_object(b"blob", archive.read(info)))

    def tree_hash(node: dict[str, object]) -> bytes:
        entries: list[tuple[bytes, bytes, bytes]] = []
        for name, value in node.items():
            encoded = name.encode("utf-8")
            if isinstance(value, dict):
                entries.append((encoded + b"/", b"40000 " + encoded, tree_hash(value)))
            else:
                mode, digest = value  # type: ignore[misc]
                entries.append((encoded, mode + b" " + encoded, digest))
        body = b"".join(head + b"\x00" + digest for _, head, digest in sorted(entries))
        return _git_object(b"tree", body)

    return tree_hash(root).hex()


def main(argv: list[str] | None = None) -> int:
    """Run the verification and return the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-tree", default=None)
    args = parser.parse_args(argv)

    with zipfile.ZipFile(args.archive) as archive:
        crlf = find_crlf(archive)
        for name, digest in sorted(key_hashes(archive).items()):
            print(f"{digest}  {name}")
        tree = content_tree(archive)
        commit = archive_commit(archive)
    print(f"tree {tree}")
    print(f"commit {commit or 'not recorded'}")

    status = 0
    for name in crlf:
        print(f"CRLF found: {name}", file=sys.stderr)
        status = 1
    if args.expected_tree and tree != args.expected_tree:
        print(f"tree mismatch: expected {args.expected_tree}, archive has {tree}", file=sys.stderr)
        status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
