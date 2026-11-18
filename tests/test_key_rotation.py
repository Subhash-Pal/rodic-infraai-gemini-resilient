"""Key rotation with fake Gemini clients and a fake clock (no network, no real keys)."""

import pytest

from rodic_vlm.errors import GeminiAuthError, GeminiQuotaError, GeminiUnavailable
from rodic_vlm.key_rotation import (
    KeyRing,
    analyze_with_rotation,
    load_keys,
    skipped_result,
    unavailable_result,
)

SECRET_A = "SECRET-VALUE-AAAA"
SECRET_B = "SECRET-VALUE-BBBB"
SECRET_C = "SECRET-VALUE-CCCC"
MODELS = ["primary", "fallback"]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def ring(clock=None, keys=None):
    return KeyRing(
        keys or [("K0", SECRET_A), ("K1", SECRET_B), ("K2", SECRET_C)],
        clock=clock or Clock(),
    )


def make_fn(behaviour, calls):
    """behaviour: {(key_value, model): exception | dict}."""

    def fn(image_bytes, config_path=None, contract_path=None, api_key=None,
           models_override=None, fail_fast_on_quota=False):
        model = models_override[0]
        calls.append((api_key, model, fail_fast_on_quota))
        outcome = behaviour.get((api_key, model), {"observations": [], "model": model})

        if isinstance(outcome, Exception):
            raise outcome

        return dict(outcome)

    return fn


def run(fn, rk, **kw):
    return analyze_with_rotation(b"img", analyze_fn=fn, keyring=rk, models=MODELS, **kw)


def test_first_key_used_and_labelled_without_leaking_value():
    calls = []
    result = run(make_fn({}, calls), ring())

    assert result["status"] == "ok"
    assert result["key_label"] == "K0"
    assert SECRET_A not in repr(result)
    assert calls[0][2] is True  # fail-fast on quota is requested


def test_quota_on_one_key_rotates_to_next_key_same_model():
    calls = []
    beh = {(SECRET_A, "primary"): GeminiQuotaError("429")}
    result = run(make_fn(beh, calls), ring())

    assert result["key_label"] == "K1"
    assert result["model"] == "primary"
    assert [a["outcome"] for a in result["rotation_attempts"]] == ["quota", "ok"]


def test_round_robin_spreads_load_after_success():
    calls = []
    rk = ring()
    fn = make_fn({}, calls)

    labels = [run(fn, rk)["key_label"] for _ in range(4)]

    assert labels == ["K0", "K1", "K2", "K0"]


def test_cooled_key_is_skipped_then_returns_after_cooldown():
    clock = Clock()
    rk = ring(clock)
    calls = []
    beh = {(SECRET_A, "primary"): GeminiQuotaError("429")}
    fn = make_fn(beh, calls)

    run(fn, rk)                       # K0 quota -> K1 ok, K0 cools down
    calls.clear()
    rk._start = 0                     # force a start at K0 to prove it is skipped
    run(fn, rk)
    assert all(key != SECRET_A for key, _m, _f in calls)

    clock.now += 61                   # per-minute cooldown over
    calls.clear()
    rk._start = 0
    beh.clear()
    assert run(make_fn(beh, calls), rk)["key_label"] == "K0"


def test_daily_quota_cools_down_much_longer():
    clock = Clock()
    rk = ring(clock)
    beh = {(SECRET_A, "primary"): GeminiQuotaError("PerDay", daily=True)}
    run(make_fn(beh, []), rk)

    clock.now += 61
    assert "K0" not in [label for label, _ in rk.candidates("primary")]

    clock.now += 3600
    assert "K0" in [label for label, _ in rk.candidates("primary")]


def test_all_keys_exhausted_on_primary_falls_to_fallback_model():
    beh = {(s, "primary"): GeminiQuotaError("429") for s in (SECRET_A, SECRET_B, SECRET_C)}
    result = run(make_fn(beh, []), ring())

    assert result["model"] == "fallback"
    assert result["status"] == "ok"


def test_auth_failure_parks_key_and_tries_next():
    beh = {(SECRET_A, "primary"): GeminiAuthError("bad key")}
    rk = ring()
    result = run(make_fn(beh, []), rk)

    assert result["key_label"] == "K1"
    assert rk.status("primary")["K0"] == "auth_failed"


def test_everything_fails_raises_unavailable_with_attempts_only_names():
    beh = {(s, m): GeminiQuotaError("429") for s in (SECRET_A, SECRET_B, SECRET_C) for m in MODELS}

    with pytest.raises(GeminiUnavailable) as info:
        run(make_fn(beh, []), ring())

    attempts = info.value.attempts
    assert len(attempts) == 6
    assert {a["key"] for a in attempts} == {"K0", "K1", "K2"}
    assert SECRET_A not in repr(attempts) and SECRET_A not in str(info.value)


def test_non_quota_model_failure_moves_to_next_model_not_next_key():
    calls = []
    beh = {(SECRET_A, "primary"): RuntimeError("503 overloaded")}
    result = run(make_fn(beh, calls), ring())

    primary_calls = [c for c in calls if c[1] == "primary"]
    assert len(primary_calls) == 1          # did not try K1/K2 for the same outage
    assert result["model"] == "fallback"


def test_no_keys_configured_is_unavailable():
    with pytest.raises(GeminiUnavailable):
        run(make_fn({}, []), KeyRing([]))


def test_time_budget_stops_the_loop():
    clock = Clock()
    rk = ring(clock)

    def slow(image_bytes, **kw):
        clock.now += 50
        raise GeminiQuotaError("429")

    with pytest.raises(GeminiUnavailable) as info:
        analyze_with_rotation(b"i", analyze_fn=slow, keyring=rk, models=MODELS,
                              max_total_seconds=60, clock=clock)

    assert any(a["outcome"] == "time_budget_exhausted" for a in info.value.attempts)


def test_load_keys_skips_unset_and_duplicate_values(monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "same")
    monkeypatch.setenv("GEMINI_API_KEY1", "same")
    monkeypatch.setenv("GEMINI_API_KEY3", "other")

    keys = load_keys(["GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3"])

    assert [label for label, _ in keys] == ["GEMINI_API_KEY", "GEMINI_API_KEY3"]


def test_fallback_records_have_arbitration_compatible_shape():
    for rec, status in ((unavailable_result(GeminiUnavailable("x", attempts=[{"key": "K0"}])), "unavailable"),
                        (skipped_result(), "skipped")):
        assert rec["status"] == status
        assert rec["observations"] == [] and rec["negative_evidence"] == []
        assert rec["engineering_decision"]["status"] == "not_provided"

    assert unavailable_result(RuntimeError("x"))["queued_for_retry"] is True
    assert skipped_result()["queued_for_retry"] is False
