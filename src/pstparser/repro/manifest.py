"""The record written alongside every run, describing how it was produced.

A manifest answers the question "what exactly produced this artefact": which
commit, which configuration, which corpus, which seeds and which environment.
It is the reason a result can be traced back and reproduced later.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from pstparser import __version__
from pstparser.config import ExperimentConfig
from pstparser.repro.seeding import SeedRecord

MANIFEST_FILE = "run_manifest.json"

#: Packages whose exact version is recorded, when installed.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "unsloth",
    "unsloth_zoo",
    "numpy",
    "pandas",
    "scikit-learn",
)


@dataclass(frozen=True)
class GitState:
    """Position of the working tree in version control.

    Attributes:
        commit: Full hash of the checked out commit, when available.
        branch: Name of the current branch, when available.
        dirty: Whether tracked files differ from the commit.
    """

    commit: str | None
    branch: str | None
    dirty: bool | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the state."""
        return {"commit": self.commit, "branch": self.branch, "dirty": self.dirty}


def build_manifest(
    config: ExperimentConfig,
    seeds: SeedRecord,
    stage: str,
    *,
    inputs: Mapping[str, str | Path] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest for a single run.

    Args:
        config: The resolved experiment configuration.
        seeds: Record of the seeds that were applied.
        stage: Pipeline stage that produced the artefacts, such as ``train``.
        inputs: Files whose content should be fingerprinted, keyed by a label.
            Missing files are recorded with a null digest.
        extra: Additional stage-specific fields merged into the manifest.

    Returns:
        A JSON-serialisable mapping.
    """
    manifest: dict[str, Any] = {
        "stage": stage,
        "experiment": config.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "manifest_version": 2,
        "source": source_snapshot(),
        "git": git_state().as_dict(),
        "config": {
            "resolved": config.model_dump(mode="json"),
            "digest": digest_text(json.dumps(config.model_dump(mode="json"), sort_keys=True)),
        },
        "inputs": {label: digest_file(path) for label, path in (inputs or {}).items()},
        "seeds": seeds.as_dict(),
        "environment": environment(),
    }
    manifest.update(extra or {})
    return manifest


def write_manifest(manifest: Mapping[str, Any], directory: str | Path) -> Path:
    """Write a manifest into a run directory.

    Args:
        manifest: The mapping returned by :func:`build_manifest`.
        directory: Run directory, created if absent.

    Returns:
        The path that was written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / manifest_filename(str(manifest["stage"]))
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return destination


def manifest_filename(stage: str) -> str:
    """Return a safe filename for a stage-specific manifest."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", stage):
        raise ValueError(f"invalid manifest stage: {stage!r}")
    return f"{stage}_manifest.json"


def ensure_stage_outputs(directory: str | Path, stage: str, filenames: Iterable[str] = ()) -> Path:
    """Refuse a rerun before any existing stage artefact can be overwritten."""
    directory = Path(directory)
    occupied = [
        directory / name
        for name in (manifest_filename(stage), *filenames)
        if (directory / name).exists()
    ]
    if occupied:
        raise FileExistsError(f"stage outputs already exist; choose a new directory: {occupied[0]}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def new_stage_directory(parent: str | Path, stage: str) -> Path:
    """Choose a fresh output directory without touching historical artefacts."""
    manifest_filename(stage)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return Path(parent) / f"{stage}-{stamp}-{uuid4().hex[:8]}"


def source_snapshot(root: str | Path | None = None) -> dict[str, Any]:
    """Fingerprint executable package files and dependency inputs, never secrets.

    Package files are resolved from this module when installed; project inputs
    are located in the supplied root or current working directory. Missing lock
    metadata is explicit, rather than replaced with a version-control claim.
    """
    root = Path(root) if root is not None else Path.cwd()
    package = root / "src" / "pstparser"
    if not package.is_dir():
        package = Path(__file__).resolve().parents[1]
    files = {
        f"pstparser/{path.relative_to(package).as_posix()}": digest_file(path)
        for path in sorted(package.rglob("*.py"))
    }
    for name in ("uv.lock", "pyproject.toml"):
        files[name] = digest_file(root / name)
    return {"files": files, "digest": digest_text(json.dumps(files, sort_keys=True))}


def digest_file(path: str | Path) -> str | None:
    """Compute the SHA-256 digest of a file.

    Args:
        path: File to read.

    Returns:
        The hexadecimal digest, or ``None`` if the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        return None

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_text(text: str) -> str:
    """Compute the SHA-256 digest of a string encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_directory(directory: str | Path, patterns: Iterable[str] = ("*",)) -> str | None:
    """Compute a digest covering the files of a directory.

    File names and contents both contribute, so a rename changes the result.

    Args:
        directory: Directory to walk.
        patterns: Glob patterns selecting the files to include.

    Returns:
        The hexadecimal digest, or ``None`` if the directory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    selected = sorted(
        {path for pattern in patterns for path in directory.rglob(pattern) if path.is_file()}
    )
    hasher = hashlib.sha256()
    for path in selected:
        hasher.update(path.relative_to(directory).as_posix().encode("utf-8"))
        hasher.update((digest_file(path) or "").encode("utf-8"))
    return hasher.hexdigest()


def git_state() -> GitState:
    """Read the position of the working tree in version control.

    Returns:
        The state, with null fields when git is unavailable or the directory is
        not a repository.
    """
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return GitState(
        commit=commit,
        branch=branch,
        dirty=None if status is None else bool(status),
    )


def environment() -> dict[str, Any]:
    """Describe the interpreter, the platform and the relevant package versions."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "accelerator": accelerator(),
    }


def package_versions() -> dict[str, str | None]:
    """Report the installed version of each tracked package."""
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def accelerator() -> dict[str, Any]:
    """Describe the compute device, when torch is installed."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch is not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device", "torch": torch.__version__}

    return {
        "available": True,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "capability": list(torch.cuda.get_device_capability(0)),
        "cudnn": torch.backends.cudnn.version(),
        "memory_bytes": [
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ],
        "driver": _gpu_driver(),
    }


def _gpu_driver() -> str | None:
    """Report the NVIDIA driver version, or ``None`` when it cannot be queried."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.split()
    return lines[0] if result.returncode == 0 and lines else None


def _git(*args: str) -> str | None:
    """Run a git command, returning its trimmed output or ``None`` on failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()
