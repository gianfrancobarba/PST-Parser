"""Protocol freeze: hash every frozen input, the code and the lock file, and verify them.

``protocol/protocol-freeze.json`` lists by SHA-256 every file the
study is frozen on (samples, gold, variants, configurations, oracles, analysis
plan), the digest of the package source (generation and analysis code alike,
07 RD-4) and the lock file. It never holds its own hash: a generation
configuration points at the freeze through ``review_record`` and the freeze
lists the configuration. The file is written once, canonically serialised;
the date it was written is proved by the dated email to the supervisor that
carries its SHA-256, not by this file.

Format v2 also lists every other file of the repository by hash
(``repository``): gold, notebooks, prompts, configurations and
``pyproject.toml`` included. In a git checkout the files are those git tracks,
and a new file that git does not ignore is reported; elsewhere (an archive on
Colab) every file is listed and only the listed ones are checked.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

import yaml

from pstparser.repro.manifest import digest_directory, digest_file, source_snapshot

FREEZE_FORMAT: Final = "protocol-freeze-v2"
SPEC_FORMAT: Final = "freeze-spec-v1"
#: Package source whose digest enters the freeze: generation, oracles and analysis.
CODE_DIRECTORY: Final = "src/pstparser"
CODE_PATTERNS: Final = ("*.py",)
#: Files written with or after the freeze, never listed in it: full paths from the
#: repository root, so the names are allowed only at the root of protocol/.
UNLISTED_ALLOWED: Final = (
    "protocol/protocol-freeze*.json",
    "protocol/FREEZE-*.md",
    "protocol/deviations.md",
)
#: Directories whose new files are expected after the freeze.
UNLISTED_PREFIXES: Final = ("protocol/acks/", "results/")
#: Never part of the repository inventory outside git.
_NOT_SOURCE: Final = frozenset(
    {".git", "__pycache__", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"}
)
_META_KEYS: Final = {
    "supervisor_ack",
    "preregistration",
    "pilot",
    "extraction_policies",
    "conditions",
    "renderer_version",
}


@dataclass(frozen=True)
class Mismatch:
    """One difference between the freeze and the repository."""

    kind: Literal["changed", "missing", "unlisted"]
    path: str
    expected: str | None
    found: str | None


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """Canonical serialisation: sorted keys, no spaces, UTF-8, one final line feed."""
    text = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def _relative(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path or not path:
        raise ValueError(f"{path!r}: items are relative paths inside the repository")
    return pure.as_posix()


def code_digest(root: Path) -> str:
    """Digest of the package source (names and contents of every ``*.py``)."""
    digest = digest_directory(root / CODE_DIRECTORY, CODE_PATTERNS)
    if digest is None:
        raise ValueError(f"{root / CODE_DIRECTORY}: no package source")
    return digest


def _allowed_unlisted(relative: str) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in UNLISTED_ALLOWED) or any(
        relative.startswith(prefix) for prefix in UNLISTED_PREFIXES
    )


def _covered(relative: str) -> bool:
    """Checked on their own: the package source (``code``) and the lock file."""
    pure = PurePosixPath(relative)
    in_code = pure.is_relative_to(CODE_DIRECTORY) and any(
        pure.match(pattern) for pattern in CODE_PATTERNS
    )
    return in_code or relative == "uv.lock"


def _git(root: Path, *arguments: str) -> list[str] | None:
    """Paths printed by ``git ls-files`` in ``root``; ``None`` outside a git checkout."""
    if not (root / ".git").exists() or shutil.which("git") is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", *arguments],
        capture_output=True,
        check=True,
    )
    return sorted(name for name in result.stdout.decode("utf-8").split("\0") if name)


def repository_files(root: Path) -> tuple[list[str], bool]:
    """The files of the repository and whether git listed them.

    In a git checkout: the files git tracks. Elsewhere: every file under
    ``root`` outside caches, virtual environments and compiled files.
    """
    tracked = _git(root, "--cached")
    if tracked is not None:
        return tracked, True
    files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not _NOT_SOURCE & set(path.relative_to(root).parts)
        and path.suffix != ".pyc"
    ]
    return sorted(files), False


def build_freeze(
    root: Path, items: Sequence[str], meta: Mapping[str, Any], *, frozen_at: str | None = None
) -> dict[str, Any]:
    """The freeze document of ``items`` (paths relative to ``root``) and ``meta``.

    ``frozen_at`` (UTC, ``YYYY-MM-DDTHH:MM:SSZ``) defaults to now; with the same
    inputs and time the document, and its SHA-256, are the same.

    ``meta`` holds ``protocol_version``, ``frozen_by`` and the descriptive keys
    of the freeze (supervisor acknowledgement, preregistration, pilot,
    extraction policies, conditions, renderer version).

    Raises:
        ValueError: If an item is repeated, missing, outside the repository, or
            a metadata key is unknown.
    """
    root = Path(root)
    names = [_relative(item) for item in items]
    if len(set(names)) != len(names) or not names:
        raise ValueError("items must be a non-empty list without repetitions")
    unknown = set(meta) - _META_KEYS - {"protocol_version", "frozen_by"}
    if unknown:
        raise ValueError(f"unknown metadata keys: {sorted(unknown)}")
    for key in ("protocol_version", "frozen_by"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"metadata {key} is required")
    hashes: dict[str, str] = {}
    for name in names:
        digest = digest_file(root / name)
        if digest is None:
            raise ValueError(f"{name}: no such file")
        hashes[name] = digest
    lock = digest_file(root / "uv.lock")
    if lock is None:
        raise ValueError("uv.lock is required")
    snapshot = source_snapshot(root)
    files, from_git = repository_files(root)
    inventory = {
        name: digest_file(root / name)
        for name in files
        if name not in hashes and not _allowed_unlisted(name) and not _covered(name)
    }
    return {
        "format": FREEZE_FORMAT,
        "protocol_version": meta["protocol_version"],
        "frozen_at": frozen_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_by": meta["frozen_by"],
        **{key: meta.get(key) for key in sorted(_META_KEYS)},
        "items": dict(sorted(hashes.items())),
        "code": {
            "directory": CODE_DIRECTORY,
            "patterns": list(CODE_PATTERNS),
            "sha256": code_digest(root),
        },
        "source": {"snapshot_digest": snapshot["digest"]},
        "environment": {"uv_lock_sha256": lock},
        "repository": {
            "listed_by": "git" if from_git else "directory",
            "files": {name: digest for name, digest in inventory.items() if digest is not None},
        },
    }


def read_spec(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Items and metadata of a freeze specification (``freeze-spec-v1``, YAML).

    Raises:
        ValueError: If the format or the key set is not the expected one.
    """
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != SPEC_FORMAT:
        raise ValueError(f"expected a {SPEC_FORMAT} document")
    allowed = {"format", "items", "protocol_version", "frozen_by"} | _META_KEYS
    if not set(document) <= allowed or "items" not in document:
        raise ValueError(f"a freeze specification holds items and {sorted(allowed - {'items'})}")
    items = document["items"]
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError("items must be a list of paths")
    meta = {key: value for key, value in document.items() if key not in {"format", "items"}}
    return items, meta


def write_freeze(document: Mapping[str, Any], out: Path) -> Path:
    """Write the freeze once; an existing file is never replaced."""
    out = Path(out)
    with out.open("xb") as handle:
        handle.write(canonical_bytes(document))
    return out


def default_root(freeze: Path) -> Path:
    """The repository of a freeze kept in ``<root>/protocol/``."""
    return Path(freeze).resolve().parent.parent


def load_freeze(path: Path) -> dict[str, Any]:
    """Read a freeze and check its format and canonical serialisation.

    Raises:
        ValueError: If the file is not a canonically serialised freeze.
    """
    data = Path(path).read_bytes()
    document = json.loads(data.decode("utf-8"))
    if not isinstance(document, dict) or document.get("format") != FREEZE_FORMAT:
        raise ValueError(f"{path}: not a {FREEZE_FORMAT} document")
    if canonical_bytes(document) != data:
        raise ValueError(f"{path}: not canonically serialised")
    return document


def verify_freeze(path: Path, root: Path | None = None) -> list[Mismatch]:
    """Every difference between a freeze and the repository; empty when they match.

    Checks each listed file, every other file of the repository recorded in
    the freeze, the package source, the lock file, and that no file is left out
    of the freeze: in ``protocol/`` always, elsewhere when git can tell the new
    files apart (other than the freeze, its addenda, the deviation log, the
    acknowledgements and ``results/``).

    Raises:
        ValueError: If the file is not a freeze (see :func:`load_freeze`).
    """
    document = load_freeze(path)
    root = Path(root) if root is not None else default_root(path)
    found: list[Mismatch] = []
    for name, expected in document["items"].items():
        actual = digest_file(root / name)
        if actual is None:
            found.append(Mismatch("missing", name, expected, None))
        elif actual != expected:
            found.append(Mismatch("changed", name, expected, actual))
    code = digest_directory(root / document["code"]["directory"], document["code"]["patterns"])
    if code != document["code"]["sha256"]:
        found.append(
            Mismatch("changed", document["code"]["directory"], document["code"]["sha256"], code)
        )
    lock = digest_file(root / "uv.lock")
    if lock != document["environment"]["uv_lock_sha256"]:
        found.append(
            Mismatch("changed", "uv.lock", document["environment"]["uv_lock_sha256"], lock)
        )
    inventory = document["repository"]["files"]
    for name, expected in inventory.items():
        actual = digest_file(root / name)
        if actual is None:
            found.append(Mismatch("missing", name, expected, None))
        elif actual != expected:
            found.append(Mismatch("changed", name, expected, actual))
    known = set(document["items"]) | set(inventory)
    present: set[str] = set()
    protocol = root / "protocol"
    if protocol.is_dir():
        present |= {
            PurePosixPath(file.relative_to(root).as_posix()).as_posix()
            for file in protocol.rglob("*")
            if file.is_file()
        }
    in_git = _git(root, "--cached", "--others", "--exclude-standard")
    if in_git is not None:
        present |= {name for name in in_git if (root / name).is_file()}
    for name in sorted(present - known):
        if not _allowed_unlisted(name) and not _covered(name):
            found.append(Mismatch("unlisted", name, None, digest_file(root / name)))
    return found


def require_final_freeze(
    review_record: Path, variants: Path, configuration: Path
) -> dict[str, Any]:
    """Accept a final run only on a valid freeze that lists its variants and configuration.

    Raises:
        ValueError: If the freeze does not verify, or does not list the SHA-256
            of the variant file or of the configuration.
    """
    try:
        document = load_freeze(review_record)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(
            f"final runs require a protocol freeze as review_record: {error}"
        ) from error
    mismatches = verify_freeze(review_record)
    if mismatches:
        shown = ", ".join(f"{m.kind} {m.path}" for m in mismatches[:5])
        raise ValueError(f"the protocol freeze does not verify: {shown}")
    listed = set(document["items"].values())
    for label, path in (("variant file", variants), ("configuration", configuration)):
        if digest_file(path) not in listed:
            raise ValueError(f"the protocol freeze does not list the {label}")
    _require_frozen_gold(document, default_root(review_record), variants)
    return document


def _require_frozen_gold(document: Mapping[str, Any], root: Path, variants: Path) -> None:
    """Every H-gold variant must come from a listed rendering of frozen, listed gold.

    The render records (``*.render.json``) listed in the freeze name the prompt
    hashes of the H-gold variants they wrote, the state of the gold and the
    hash of its annotation file.

    Raises:
        ValueError: If an H-gold variant has no such record, its record is not
            on frozen gold, or the freeze does not list the annotation file.
    """
    lines = Path(variants).read_bytes().decode("utf-8").split("\n")
    wanted = {
        row["prompt_hash"]
        for row in (json.loads(line) for line in lines if line)
        if row.get("condition") == "H-gold"
    }
    if not wanted:
        return
    listed = set(document["items"].values())
    covered: set[str] = set()
    for name in document["items"]:
        if not name.endswith(".render.json"):
            continue
        record = json.loads((root / name).read_text(encoding="utf-8"))
        hashes = set(record.get("h_gold_prompt_hashes", ())) & wanted
        if not hashes:
            continue
        if record.get("gold_status") != "frozen":
            raise ValueError(f"{name}: H-gold of a final run requires frozen gold")
        if record.get("annotations_sha256") not in listed:
            raise ValueError(f"{name}: the protocol freeze does not list the annotation file")
        covered |= hashes
    if wanted - covered:
        raise ValueError(
            "H-gold of a final run requires frozen gold: "
            f"{len(wanted - covered)} variants have no listed render record"
        )
