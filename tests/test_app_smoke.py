"""Headless start-up check of the Streamlit app (no upload, no network)."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_app_starts_without_error_and_asks_for_images():
    at = testing.AppTest.from_file(str(APP), default_timeout=90).run()

    assert not at.exception
    assert any("Upload one or more" in i.value for i in at.info)
    assert at.sidebar.slider[0].value == 0.5
    assert any("Gemini keys configured" in c.value or "key status unavailable" in c.value
               for c in at.sidebar.caption)
