"""One-image analysis pipeline, independent of Streamlit (so it can be unit-tested).

Steps: validate -> CV bundle (whole-image scores + tile scores; the expensive, cacheable part)
-> regions at the chosen threshold -> Gemini evidence (rotated keys, CV-only fallback)
-> arbitration -> indicative recommendations -> structured JSON record.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rodic_inference import localization
from rodic_inference.evidence_arbitration import build_inspection_evidence_record
from rodic_inference.irc_recommender import (
    Recommendation,
    normalise_class,
    recommend,
    severity_counts,
    worst_severity,
)
from rodic_inference.report_json import build_record
from rodic_inference.version import APP_VERSION


def compute_cv_bundle(models: tuple[Any, Any], image_rgb: np.ndarray) -> dict[str, Any]:
    """Whole-image CODEBRIM + SDNET2018 scores and tile scores (needs torch + checkpoints)."""
    from rodic_inference.inference import run_codebrim_inference, run_sdnet_inference

    codebrim_model, sdnet_model = models

    bundle = localization.compute_tile_bundle(models, image_rgb)
    bundle["codebrim"] = run_codebrim_inference(codebrim_model, image_rgb)
    bundle["sdnet"] = run_sdnet_inference(sdnet_model, image_rgb)

    return bundle


def detected_codebrim_defects(codebrim: list[dict[str, Any]], threshold: float) -> list[str]:
    return [
        r["defect"]
        for r in codebrim
        if float(r["score"]) >= threshold and r["defect"] != "background"
    ]


def build_recommendations(
    codebrim: list[dict[str, Any]],
    sdnet_crack: float,
    threshold: float,
) -> list[Recommendation]:
    """One indicative recommendation per detected class (crack width is never guessed)."""
    scores = {r["defect"]: float(r["score"]) for r in codebrim}
    recs: list[Recommendation] = []
    seen: set[str] = set()

    for defect in detected_codebrim_defects(codebrim, threshold):
        cls = normalise_class(defect)
        recs.append(recommend(defect, confidence=scores[defect]))
        seen.add(cls)

    if float(sdnet_crack) >= threshold and "crack" not in seen:
        recs.append(recommend("crack", confidence=float(sdnet_crack)))

    if not recs:
        recs.append(recommend("background"))

    return recs


def resolve_gemini(
    gemini_fn: Callable[[bytes], dict[str, Any]] | None,
    image_bytes: bytes,
) -> dict[str, Any]:
    """Gemini evidence, or a clearly marked fallback record. Never raises."""
    from rodic_vlm.key_rotation import skipped_result, unavailable_result

    if gemini_fn is None:
        return skipped_result()

    try:
        result = dict(gemini_fn(image_bytes))
        result.setdefault("status", "ok")
        return result
    except Exception as exc:  # GeminiUnavailable, auth, missing keys, anything else
        return unavailable_result(exc)


def assemble(
    *,
    filename: str,
    image_bytes: bytes,
    image_rgb: np.ndarray,
    bundle: dict[str, Any],
    gemini_result: dict[str, Any],
    threshold: float,
    draw_boxes: bool = True,
) -> dict[str, Any]:
    """Combine cached CV scores, Gemini evidence and the threshold into the final analysis."""
    codebrim = bundle["codebrim"]
    sdnet = bundle["sdnet"]

    detected = detected_codebrim_defects(codebrim, threshold)
    regions = localization.regions_from_bundle(bundle, threshold, image_rgb=image_rgb)
    annotated = localization.draw_regions(image_rgb, regions) if draw_boxes else image_rgb

    gemini_for_arbitration = gemini_result if gemini_result.get("status", "ok") == "ok" else None

    # Computer-vision evidence for arbitration = CODEBRIM detections plus an SDNET2018 crack
    # (the original app ignored SDNET here, which contradicted its own "Computer-Vision
    # Evidence" panel). Scores are still never averaged.
    cv_classes = list(detected)

    if float(sdnet["crack_probability"]) >= threshold and "crack" not in cv_classes:
        cv_classes.append("crack")

    evidence = build_inspection_evidence_record(
        {"predictions": [{"class": d} for d in cv_classes]},
        gemini_for_arbitration,
    )
    arbitration = evidence["arbitration"]

    recs = build_recommendations(codebrim, float(sdnet["crack_probability"]), threshold)

    record = build_record(
        filename=filename,
        image_bytes=image_bytes,
        image_shape=tuple(int(v) for v in image_rgb.shape[:2]),
        threshold=threshold,
        cv_bundle=bundle,
        detected_defects=detected,
        regions=regions,
        gemini_result=gemini_result,
        arbitration=arbitration,
        recommendations=[r.to_dict() for r in recs],
        worst_severity=worst_severity(recs),
        app_version=APP_VERSION,
    )

    return {
        "record": record,
        "annotated_rgb": annotated,
        "regions": regions,
        "detected_defects": detected,
        "arbitration": arbitration,
        "recommendations": recs,
        "severity_counts": severity_counts(recs),
    }
