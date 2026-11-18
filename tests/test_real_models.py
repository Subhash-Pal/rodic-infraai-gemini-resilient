"""Runs the committed checkpoints on CPU (skipped when torch or the weights are missing)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "models" / "best_codebrim.pt").exists():
    pytest.skip("checkpoints not present", allow_module_level=True)

from rodic_inference import pipeline
from rodic_inference.inference import load_codebrim_model, load_sdnet_model
from rodic_inference.report_json import to_json_bytes


@pytest.fixture(scope="module")
def models():
    return load_codebrim_model(), load_sdnet_model()


def texture(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(150, 18, (768, 1024, 3)).clip(0, 255).astype(np.uint8)

    return cv2.GaussianBlur(img, (0, 0), 2)


def test_real_checkpoints_run_and_a_dark_line_raises_the_sdnet_score(models):
    clean = texture()
    cracked = clean.copy()
    cv2.polylines(cracked, [np.array([[100, 200], [480, 330], [950, 540]], np.int32)], False, (25, 25, 25), 3)

    b_clean = pipeline.compute_cv_bundle(models, clean)
    b_crack = pipeline.compute_cv_bundle(models, cracked)

    for bundle in (b_clean, b_crack):
        scores = [r["score"] for r in bundle["codebrim"]]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert len(bundle["tile_scores"]["sdnet"]) == len(bundle["tiles"])

    assert b_crack["sdnet"]["crack_probability"] > b_clean["sdnet"]["crack_probability"]


def test_full_pipeline_with_real_models_produces_valid_json_and_boxes(models):
    img = texture(1)
    cv2.polylines(img, [np.array([[100, 200], [480, 330], [950, 540]], np.int32)], False, (25, 25, 25), 3)
    data = cv2.imencode(".jpg", img)[1].tobytes()

    bundle = pipeline.compute_cv_bundle(models, img)
    out = pipeline.assemble(
        filename="t.jpg",
        image_bytes=data,
        image_rgb=img,
        bundle=bundle,
        gemini_result=pipeline.resolve_gemini(None, data),
        threshold=0.5,
    )

    assert out["regions"], "expected at least one region on a clear synthetic crack"
    h, w = img.shape[:2]
    for r in out["regions"]:
        x1, y1, x2, y2 = r["box"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h

    assert out["annotated_rgb"].shape == img.shape
    assert b"rodic.inspection.v1" in to_json_bytes(out["record"])
