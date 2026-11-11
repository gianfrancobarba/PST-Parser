"""Execute Python benchmark checks in a constrained, disposable Docker container.

Only a fresh job directory is mounted, read-only. Images must already exist
locally and be pinned by digest. This runner never falls back to host execution.
It does not claim protection against a compromised container runtime/kernel.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SandboxConfig:
    """Frozen runtime limits; the image must contain Python and benchmark dependencies."""

    image: str
    timeout_seconds: int = 10
    memory_mb: int = 256
    pids: int = 64

    def __post_init__(self) -> None:
        """Reject mutable image tags and unbounded resource settings."""
        if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", self.image):
            raise ValueError("sandbox image must be pinned as name@sha256:<64 hex digits>")
        if min(self.timeout_seconds, self.memory_mb, self.pids) <= 0:
            raise ValueError("sandbox resource limits must be positive")


@dataclass(frozen=True)
class ExecutionResult:
    """Execution classification; infrastructure failure is not a failed solution."""

    status: Literal["passed", "failed", "timeout", "infrastructure_error"]
    exit_code: int | None
    detail: str


def container_command(config: SandboxConfig, directory: Path, name: str) -> list[str]:
    """Build an isolated run without network, privileges, credentials or repo mount."""
    return [
        "docker",
        "create",
        "--name",
        name,
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(config.pids),
        "--memory",
        f"{config.memory_mb}m",
        "--memory-swap",
        f"{config.memory_mb}m",
        "--cpus",
        "1",
        "--user",
        "65534:65534",
        "--log-driver",
        "none",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=32m",
        "--mount",
        f"type=bind,source={directory.resolve()},target=/work,readonly",
        "--workdir",
        "/work",
        "--entrypoint",
        "python",
        config.image,
        "-I",
        "-B",
        "/work/check.py",
    ]


def run_python_check(
    code: str,
    harness: str,
    config: SandboxConfig,
    *,
    temporary_root: Path | None = None,
) -> ExecutionResult:
    """Run source plus a trusted harness; exit 0 means its checks passed.

    The harness is versioned benchmark infrastructure, never model output.
    stdout/stderr from generated code are discarded to prevent output floods;
    raw generation and oracle identity must be logged by the caller. Assertion,
    syntax, import and runtime errors have a failed exit status. A container is
    forcibly removed on timeout, interruption and completion alike.
    """
    name = f"pst-eval-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="pst-eval-", dir=temporary_root) as job:
        directory = Path(job)
        (directory / "candidate.py").write_text(code, encoding="utf-8", newline="")
        # A supervisor checks a completion marker as well as the exit code.
        # This detects sys.exit(0)/os._exit(0) before tests, not a malicious
        # program intentionally tampering with its in-process test harness.
        worker = (
            "import sys\nsys.path.insert(0, '/work')\n"
            + "exec(compile("
            + repr(harness)
            + ", '<trusted-harness>', 'exec'))\n"
            + "from pathlib import Path\nPath(sys.argv[1]).write_text('completed')\n"
        )
        marker = f"/tmp/{uuid.uuid4().hex}"
        supervisor = (
            "import subprocess, sys\nfrom pathlib import Path\n"
            f"marker = {json.dumps(marker)}\n"
            "result = subprocess.run([sys.executable, '-I', '-B', '/work/worker.py', marker])\n"
            "completed = Path(marker).is_file() and Path(marker).read_text() == 'completed'\n"
            "sys.exit(0 if result.returncode == 0 and completed else 1)\n"
        )
        (directory / "worker.py").write_text(worker, encoding="utf-8", newline="")
        (directory / "check.py").write_text(supervisor, encoding="utf-8", newline="")
        # TemporaryDirectory is 0700 on POSIX; the non-root container user
        # needs read/traverse access only to these deliberately non-secret files.
        directory.chmod(0o755)
        for path in directory.iterdir():
            path.chmod(0o644)
        may_exist = True
        container_created = False
        try:
            created = subprocess.run(
                container_command(config, directory, name),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if created.returncode != 0:
                may_exist = False
                return ExecutionResult("infrastructure_error", None, created.stderr[-2000:])
            container_created = True
            started = subprocess.run(
                ["docker", "start", name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if started.returncode != 0:
                return ExecutionResult("infrastructure_error", None, started.stderr[-2000:])
            try:
                waited = subprocess.run(
                    ["docker", "wait", name],
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult("timeout", None, "wall-clock limit exceeded")
            if waited.returncode != 0 or not waited.stdout.strip().isdigit():
                return ExecutionResult("infrastructure_error", None, waited.stderr[-2000:])
            exit_code = int(waited.stdout.strip())
            return ExecutionResult(
                "passed" if exit_code == 0 else "failed",
                exit_code,
                "container exit status",
            )
        except OSError as exc:
            may_exist = container_created
            return ExecutionResult("infrastructure_error", None, str(exc))
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult("infrastructure_error", None, str(exc))
        finally:
            if may_exist:
                try:
                    removed = subprocess.run(
                        ["docker", "rm", "--force", name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=30,
                        check=False,
                        text=True,
                    )
                    if removed.returncode != 0 and "No such container" not in removed.stderr:
                        raise RuntimeError(f"container cleanup failed; inspect {name}")
                except (OSError, subprocess.TimeoutExpired):
                    raise RuntimeError(f"container cleanup failed; inspect {name}") from None
