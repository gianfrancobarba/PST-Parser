"""Explicit security judgments, not an automatic claim that code is safe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SecurityState = Literal["V", "N", "I", "U"]


@dataclass(frozen=True)
class SecurityJudgment:
    """A reviewed scenario outcome with retained evidence and oracle identity."""

    usable: bool
    checks_complete: bool
    confirmed_findings: tuple[str, ...]
    oracle_version: str
    evidence: str

    def __post_init__(self) -> None:
        """Require a traceable oracle and justification even for negative results."""
        if type(self.usable) is not bool or type(self.checks_complete) is not bool:
            raise ValueError("usable and checks_complete must be booleans")
        if not isinstance(self.confirmed_findings, tuple) or any(
            not isinstance(finding, str) or not finding.strip()
            for finding in self.confirmed_findings
        ):
            raise ValueError("findings must be a tuple of non-empty evidence references")
        if not isinstance(self.oracle_version, str) or not isinstance(self.evidence, str):
            raise ValueError("oracle_version and evidence must be strings")
        if not self.oracle_version.strip() or not self.evidence.strip():
            raise ValueError("oracle_version and evidence are required")

    @property
    def state(self) -> SecurityState:
        """Apply protocol precedence I, then confirmed V, then N, otherwise U."""
        if not self.usable:
            return "I"
        if self.confirmed_findings:
            return "V"
        return "N" if self.checks_complete else "U"

    @property
    def outcome(self) -> int:
        """Operational utility/no-finding endpoint; not absolute security."""
        return int(self.state == "N")
