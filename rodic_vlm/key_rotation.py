"""Gemini API key rotation with an offline (CV-only) fallback.

Why: free-tier Gemini limits are per Google Cloud project AND per model. One exhausted key
used to silently drop the Gemini evidence. This module spreads calls over several keys
(``GEMINI_API_KEY``, ``GEMINI_API_KEY1`` ... ``GEMINI_API_KEY5``) and, when every key/model
is unavailable, returns a clearly marked "unavailable" evidence record so the app keeps
working on computer-vision evidence alone.

Policy
------
* Primary model first: try it on every available key, then the next fallback model on every
  key (quality of the evidence is better with the primary model).
* HTTP 429 -> that (key, model) pair cools down (60 s; 1 h if the message names a per-day
  quota) and the next key is tried at once (no waiting).
* Auth failure -> that key is parked for 6 h and the next key is tried.
* Any other failure of a model (5xx after the client's bounded retries, bad output) -> move to
  the next model; keys are not cycled for an outage that is not about the key.
* Key VALUES are never logged, stored in results or put in exceptions. Only the secret NAME
  (label, e.g. ``GEMINI_API_KEY2``) appears.
* Keys that hold the same value are used once. Keys from one Google Cloud project share one
  quota, so rotation only helps when the keys belong to different projects.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .errors import (
    GeminiAuthError,
    GeminiQuotaError,
    GeminiUnavailable,
)

DEFAULT_KEY_NAMES = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY{i}" for i in range(1, 6)]

DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "env_names": DEFAULT_KEY_NAMES,
    "rate_limit_cooldown_seconds": 60,
    "daily_quota_cooldown_seconds": 3600,
    "auth_cooldown_seconds": 21600,
    "max_total_seconds": 90,
}

DECISION_BOUNDARY = {
    "status": "not_provided",
    "reason": (
        "Engineering significance and final disposition "
        "require qualified human validation."
    ),
}


def _read_secret(name: str) -> str | None:
    value = None

    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        value = None

    value = value or os.environ.get(name)

    if not value:
        return None

    value = str(value).strip()

    return value or None


def load_keys(names: list[str]) -> list[tuple[str, str]]:
    """Return ``[(label, key), ...]`` for the names that are set, skipping duplicate values."""

    keys: list[tuple[str, str]] = []
    seen: set[str] = set()

    for name in names:
        value = _read_secret(name)

        if value and value not in seen:
            seen.add(value)
            keys.append((name, value))

    return keys


class KeyRing:
    """Thread-safe key pool with per-(key, model) cooldowns and round-robin start."""

    def __init__(
        self,
        keys: list[tuple[str, str]],
        *,
        clock: Callable[[], float] = time.monotonic,
        rate_limit_cooldown: float = 60.0,
        daily_quota_cooldown: float = 3600.0,
        auth_cooldown: float = 21600.0,
    ) -> None:
        self._keys = list(keys)
        self._clock = clock
        self._rate_cd = float(rate_limit_cooldown)
        self._daily_cd = float(daily_quota_cooldown)
        self._auth_cd = float(auth_cooldown)
        self._cool: dict[tuple[str, str | None], float] = {}
        self._start = 0
        self._lock = threading.Lock()

    def labels(self) -> list[str]:
        return [label for label, _ in self._keys]

    def _cooling(self, label: str, model: str | None, now: float) -> bool:
        if self._cool.get((label, None), 0.0) > now:   # auth failure
            return True

        return model is not None and self._cool.get((label, model), 0.0) > now

    def candidates(self, model: str) -> list[tuple[str, str]]:
        """Keys usable for ``model`` now, in round-robin order."""

        with self._lock:
            now = self._clock()
            n = len(self._keys)
            ordered = [self._keys[(self._start + i) % n] for i in range(n)] if n else []

            return [
                (label, key)
                for label, key in ordered
                if not self._cooling(label, model, now)
            ]

    def mark_quota(self, label: str, model: str, *, daily: bool = False) -> None:
        with self._lock:
            seconds = self._daily_cd if daily else self._rate_cd
            self._cool[(label, model)] = self._clock() + seconds

    def mark_auth_failed(self, label: str) -> None:
        with self._lock:
            self._cool[(label, None)] = self._clock() + self._auth_cd

    def mark_success(self, label: str) -> None:
        """Start the next call from the following key to spread load."""

        with self._lock:
            labels = [lab for lab, _ in self._keys]

            if label in labels:
                self._start = (labels.index(label) + 1) % len(labels)

    def status(self, model: str | None = None) -> dict[str, str]:
        """``{label: "ready" | "cooling" | "auth_failed"}`` for display (names only)."""

        with self._lock:
            now = self._clock()
            out: dict[str, str] = {}

            for label, _ in self._keys:
                if self._cool.get((label, None), 0.0) > now:
                    out[label] = "auth_failed"
                elif model is not None and self._cool.get((label, model), 0.0) > now:
                    out[label] = "cooling"
                else:
                    out[label] = "ready"

            return out


_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "app" / "gemini_config.json"


def load_settings(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _DEFAULT_CONFIG

    try:
        with Path(path).open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        config = {}

    settings = dict(DEFAULT_SETTINGS)
    settings.update(config.get("key_rotation", {}) or {})

    return settings


_DEFAULT_RING: KeyRing | None = None
_DEFAULT_RING_LOCK = threading.Lock()


def get_default_keyring(config_path: Path | None = None) -> KeyRing:
    """Process-wide key ring, built lazily so late-set secrets are picked up."""

    global _DEFAULT_RING

    with _DEFAULT_RING_LOCK:
        if _DEFAULT_RING is None or not _DEFAULT_RING.labels():
            settings = load_settings(config_path)

            _DEFAULT_RING = KeyRing(
                load_keys(list(settings["env_names"])),
                rate_limit_cooldown=settings["rate_limit_cooldown_seconds"],
                daily_quota_cooldown=settings["daily_quota_cooldown_seconds"],
                auth_cooldown=settings["auth_cooldown_seconds"],
            )

        return _DEFAULT_RING


def reset_default_keyring() -> None:
    """Forget the cached ring (tests, or after changing secrets)."""

    global _DEFAULT_RING

    with _DEFAULT_RING_LOCK:
        _DEFAULT_RING = None


def analyze_with_rotation(
    image_bytes: bytes,
    *,
    config_path: Path | None = None,
    contract_path: Path | None = None,
    analyze_fn: Callable[..., dict[str, Any]] | None = None,
    keyring: KeyRing | None = None,
    models: list[str] | None = None,
    max_total_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Gemini evidence with key rotation. Raises ``GeminiUnavailable`` if nothing worked."""

    if analyze_fn is None:
        from .gemini_vlm import analyze_image_with_gemini as analyze_fn  # lazy: needs google-genai

    if models is None:
        from .gemini_vlm import get_model_chain

        models = get_model_chain(config_path)

    settings = load_settings(config_path)

    if max_total_seconds is None:
        max_total_seconds = float(settings["max_total_seconds"])

    ring = keyring or get_default_keyring(config_path)

    if not ring.labels():
        raise GeminiUnavailable(
            "No Gemini API key is configured "
            "(set GEMINI_API_KEY, optionally GEMINI_API_KEY1 to GEMINI_API_KEY5).",
            attempts=[],
        )

    attempts: list[dict[str, str]] = []
    started = clock()

    def out_of_time() -> bool:
        return (clock() - started) > max_total_seconds

    for model in models:
        for label, key in ring.candidates(model):

            if out_of_time():
                attempts.append({"key": label, "model": model, "outcome": "time_budget_exhausted"})
                raise GeminiUnavailable(
                    "Gemini time budget exhausted before any key produced usable output.",
                    attempts=attempts,
                )

            try:
                result = analyze_fn(
                    image_bytes,
                    config_path=config_path,
                    contract_path=contract_path,
                    api_key=key,
                    models_override=[model],
                    fail_fast_on_quota=True,
                )

            except GeminiQuotaError as exc:
                ring.mark_quota(label, model, daily=exc.daily)
                attempts.append(
                    {"key": label, "model": model, "outcome": "quota_daily" if exc.daily else "quota"}
                )
                continue

            except GeminiAuthError:
                ring.mark_auth_failed(label)
                attempts.append({"key": label, "model": model, "outcome": "auth_failed"})
                continue

            except Exception as exc:  # model-level failure: not about this key
                attempts.append(
                    {"key": label, "model": model, "outcome": f"error:{type(exc).__name__}"}
                )
                break  # next model

            ring.mark_success(label)
            attempts.append({"key": label, "model": model, "outcome": "ok"})

            result = dict(result)
            result["status"] = "ok"
            result["key_label"] = label
            result["rotation_attempts"] = attempts

            return result

    raise GeminiUnavailable(
        "Gemini is unavailable: every configured key/model was rate-limited, rejected or failed.",
        attempts=attempts,
    )


def unavailable_result(exc: Exception) -> dict[str, Any]:
    """CV-only fallback: same shape as a normal result, with no observations.

    ``queued_for_retry`` is True because failures are not cached: the next run of the app
    (for example the 'Retry Gemini' button) tries again.
    """

    attempts = list(getattr(exc, "attempts", []) or [])

    return {
        "source": "gemini_vlm",
        "status": "unavailable",
        "model": None,
        "queued_for_retry": True,
        "reason": type(exc).__name__,
        "detail": str(exc)[:300],
        "rotation_attempts": attempts,
        "image_assessment": {},
        "observations": [],
        "negative_evidence": [],
        "limitations": [
            "Gemini visual evidence was unavailable for this image; "
            "the result is computer-vision evidence only."
        ],
        "engineering_decision": dict(DECISION_BOUNDARY),
    }


def skipped_result() -> dict[str, Any]:
    """Used when the operator switched Gemini off for the run."""

    result = unavailable_result(RuntimeError("Gemini analysis was switched off for this run."))
    result["status"] = "skipped"
    result["queued_for_retry"] = False
    result["reason"] = "disabled_by_operator"
    result["detail"] = ""

    return result
