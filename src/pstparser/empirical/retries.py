"""Bounded, logged retries of infrastructure failures (M18; Metodologia §7.5; 03 §3.5.4).

Only a failure of the infrastructure is retried: a container that does not
start, a harness or analyser that exits with an error. Anything that depends on
the program or on the model's output (a failed test, a timeout, a syntax error)
is an outcome and is never retried, so the two arms are treated alike. After
the last attempt the caller records the item as ``missing`` with its reason.
Every attempt is logged, the successful one included.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Final, TypeVar

RETRY_POLICY: Final = "infrastructure-retries-v1"
#: One attempt plus at most two new ones.
MAX_ATTEMPTS: Final = 3

Result = TypeVar("Result")


def with_retries(
    execute: Callable[[int], Result],
    infrastructure_failure: Callable[[Result], str | None],
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[Result, list[dict[str, Any]]]:
    """Run ``execute(attempt)`` until it does not fail for infrastructure reasons.

    ``infrastructure_failure`` returns a short description of an infrastructure
    failure, or None for any other result, which is final. Returns the last
    result and one log entry per attempt (``attempt``, ``status`` ``ok`` or
    ``infrastructure_error``, ``detail``, ``elapsed_seconds``). When every
    attempt failed the last result is returned: the caller turns it into a
    ``missing`` record. Exceptions are not retried: they stop the caller.

    Raises:
        ValueError: If ``max_attempts`` is not a positive integer.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        result = execute(attempt)
        detail = infrastructure_failure(result)
        attempts.append(
            {
                "attempt": attempt,
                "status": "ok" if detail is None else "infrastructure_error",
                "detail": detail or "",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        if detail is None:
            break
    return result, attempts


def exhausted(attempts: list[dict[str, Any]]) -> bool:
    """Whether every logged attempt failed for infrastructure reasons."""
    return bool(attempts) and all(a["status"] == "infrastructure_error" for a in attempts)
