"""Shared builders for tests that need a CV bundle without torch or checkpoints."""

import numpy as np

CLASSES = ["background", "crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"]


def make_bundle(
    *,
    image_size=(448, 448),
    codebrim=None,
    sdnet=0.02,
    tiles=None,
    tile_scores=None,
):
    h, w = image_size
    tiles = tiles or [(0, 0, 224, 224), (224, 0, 448, 224), (0, 224, 224, 448), (224, 224, 448, 448)]
    codebrim = codebrim or {}
    scores = {name: float(codebrim.get(name, 0.0)) for name in CLASSES}
    scores.setdefault("background", 0.9)

    if tile_scores is None:
        tile_scores = {
            "codebrim": {name: [scores[name]] * len(tiles) for name in CLASSES},
            "sdnet": [float(sdnet)] * len(tiles),
        }

    return {
        "tiles": tiles,
        "tile_scores": tile_scores,
        "scale": 1.0,
        "image_size": (h, w),
        "work_size": (h, w),
        "codebrim": [{"defect": n, "score": scores[n]} for n in CLASSES],
        "sdnet": {"crack_probability": float(sdnet)},
    }


def gemini_ok(defect_type="crack"):
    return {
        "source": "gemini_vlm",
        "model": "fake-model",
        "status": "ok",
        "key_label": "GEMINI_API_KEY1",
        "image_assessment": {"image_quality": "good", "visibility": "clear"},
        "observations": [
            {
                "evidence_type": defect_type,
                "observation": "thin linear mark",
                "location_description": "centre",
                "visual_confidence": "medium",
            }
        ],
        "negative_evidence": [],
        "limitations": [],
        "engineering_decision": {"status": "not_provided", "reason": "human validation required"},
    }


def blank_image(h=448, w=448, value=140):
    return np.full((h, w, 3), value, np.uint8)
