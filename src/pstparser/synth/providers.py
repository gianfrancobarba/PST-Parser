"""Text completion providers used to expand the corpus.

The interface is deliberately narrow: one prompt in, one completion out. Any
service exposing a chat completions endpoint compatible with the OpenAI wire
format can be used by changing the base URL and the model name.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

DEFAULT_TIMEOUT = 120.0
DEFAULT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5.0


class ProviderError(RuntimeError):
    """Raised when a provider cannot be reached or refuses the request."""


@runtime_checkable
class Provider(Protocol):
    """Anything able to answer a single instruction."""

    def complete(self, system: str, user: str, temperature: float) -> str:
        """Return the model's answer to one instruction.

        Args:
            system: Instruction describing the task.
            user: The request itself.
            temperature: Sampling temperature.

        Returns:
            The answer text.
        """
        ...


class ChatCompletionsProvider:
    """A remote service exposing an OpenAI-compatible chat completions endpoint.

    Attributes:
        base_url: Root of the API, without a trailing slash.
        model: Identifier of the model to call.
        api_key: Credential sent as a bearer token.
        timeout: Seconds to wait for a response.
        attempts: How many times a failing request is retried.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT,
        attempts: int = DEFAULT_ATTEMPTS,
    ) -> None:
        """Configure the provider.

        Args:
            base_url: Root of the API.
            model: Identifier of the model to call.
            api_key: Credential sent as a bearer token.
            timeout: Seconds to wait for a response.
            attempts: How many times a failing request is retried.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.attempts = attempts

    def complete(self, system: str, user: str, temperature: float) -> str:
        """Send one instruction and return the answer.

        Args:
            system: Instruction describing the task.
            user: The request itself.
            temperature: Sampling temperature.

        Returns:
            The answer text, stripped of surrounding whitespace.

        Raises:
            ProviderError: If every attempt fails, or if the response does not
                carry a completion.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        response = self._post("/chat/completions", payload)

        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape from {self.base_url}: {exc}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON request, retrying transient failures.

        Everything but a refusal the service would repeat is retried, and
        nothing leaves as anything other than a :class:`ProviderError`. The
        breadth is deliberate. A request can fail through unrelated exception
        hierarchies, and a connection the peer closes midway arrives as
        ``RemoteDisconnected``, which is a reset socket and a malformed status
        line at once while belonging to neither of the families the obvious
        clauses name. Naming them is a list that stays correct until the next
        one appears, and the cost of it being incomplete is not one lost
        request but a run of many that ends with nothing to show.

        Args:
            path: Endpoint path appended to the base URL.
            payload: Body to send, serialised as JSON.

        Returns:
            The decoded response body.

        Raises:
            ProviderError: If every attempt fails.
        """
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error = ProviderError(f"no attempt was made against {self.base_url}")
        for attempt in range(1, self.attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    body: dict[str, Any] = json.loads(handle.read().decode("utf-8"))
                    return body
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                last_error = ProviderError(f"HTTP {exc.code} from {self.base_url}: {detail}")
                if exc.code < 500 and exc.code != 429:
                    break
            except Exception as exc:
                last_error = ProviderError(
                    f"request to {self.base_url} failed: {type(exc).__name__}: {exc}"
                )

            if attempt < self.attempts:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise last_error


def provider_from_env(
    base_url: str,
    model: str,
    api_key_env: str,
    timeout: float = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
) -> ChatCompletionsProvider:
    """Build a provider, reading its credential from the environment.

    Args:
        base_url: Root of the API.
        model: Identifier of the model to call.
        api_key_env: Name of the variable holding the credential.
        timeout: Seconds to wait for a response.
        attempts: How many times a failing request is retried.

    Returns:
        A configured provider.

    Raises:
        ProviderError: If the variable is unset or empty.
    """
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ProviderError(f"environment variable {api_key_env} is not set")
    return ChatCompletionsProvider(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        attempts=attempts,
    )
