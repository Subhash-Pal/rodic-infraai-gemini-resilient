"""Small, dependency-free exception types shared by the Gemini client and the key rotation."""

from __future__ import annotations


class GeminiError(RuntimeError):
    """Base class for Gemini evidence-source failures."""


class GeminiAuthError(GeminiError):
    """The API key was rejected (invalid key, no permission). Another key may still work."""


class GeminiQuotaError(GeminiError):
    """HTTP 429: rate limit or quota exhausted for this key (and model).

    ``daily`` is True when the message points to a per-day quota, which will not clear
    within a minute.
    """

    def __init__(self, message: str, *, daily: bool = False) -> None:
        super().__init__(message)
        self.daily = daily


class GeminiUnavailable(GeminiError):
    """No key / model could produce usable output. The app continues with CV evidence only.

    ``attempts`` lists what was tried as dicts with the key LABEL (never the key value),
    the model name and the outcome.
    """

    def __init__(self, message: str, attempts: list | None = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])
