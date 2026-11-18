"""Structured JSON record for one inspected image (schema ``rodic.inspection.v1``).

The record keeps every evidence source separate (CODEBRIM, SDNET2018, tile regions, Gemini), never
averages scores, never contains an API key or the image itself (only its SHA-256), and always
carries the human-validation boundary and the known limits.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rodic_inference import localization
from rodic_inference.irc_recommender import DISCLAIMER, STANDARDS

SCHEMA_VERSION = "rodic.inspection.v1"

LIMITS = [
    "Screening aid with human validation; not a structural safety judgement or certification.",
    "Scores are source-specific and uncalibrated; they are not probabilities and are never averaged.",
    "CODEBRIM was seen to miss cracks on plaster, stucco and painted walls (domain shift).",
    "Regions are coarse tile-aligned boxes from classifier scores, not a trained detector.",
    "Severity and repair text are indicative lookups; standards are named, not quoted, and must be "
    "checked against the current IRC / MoRTH editions by a qualified engineer.",
    "Crack width is not measured in this application; crack severity is provisional.",
]


def _clean(value: Any) -> Any:
    """Make a value JSON-safe: finite floats, plain lists/dicts, no numpy types."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]

    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    return round(number, 6) if math.isfinite(number) else None


def review_flags(
    *,
    detected_defects: list[str],
    regions: list[dict[str, Any]],
    sdnet_crack: float,
    codebrim_scores: dict[str, float],
    threshold: float,
    gemini_result: dict[str, Any] | None,
    evidence_state: str,
) -> list[str]:
    flags = ["human_validation_required"]

    status = (gemini_result or {}).get("status", "ok" if gemini_result else "unavailable")

    if status == "unavailable":
        flags.append("gemini_unavailable_cv_only")
    elif status == "skipped":
        flags.append("gemini_skipped_by_operator")

    if evidence_state == "MODEL_CONFLICT":
        flags.append("cv_and_gemini_conflict")
    elif evidence_state == "INSUFFICIENT_EVIDENCE":
        flags.append("insufficient_evidence_recheck_image")
    elif evidence_state == "PARTIAL_AGREEMENT":
        flags.append("single_source_evidence")

    cb_crack = float(codebrim_scores.get("crack", 0.0)) >= threshold
    sd_crack = float(sdnet_crack) >= threshold

    if cb_crack != sd_crack:
        flags.append("cv_models_disagree_on_crack")

    if (detected_defects or sd_crack) and not regions:
        flags.append("detection_without_localized_region")

    if any(r.get("image_wide") for r in regions):
        flags.append("detection_image_wide_not_localized")

    if cb_crack or sd_crack:
        flags.append("crack_width_not_measured")

    return flags


def build_record(
    *,
    filename: str,
    image_bytes: bytes,
    image_shape: tuple[int, int],
    threshold: float,
    cv_bundle: dict[str, Any],
    detected_defects: list[str],
    regions: list[dict[str, Any]],
    gemini_result: dict[str, Any] | None,
    arbitration: dict[str, Any],
    recommendations: list[dict[str, Any]],
    worst_severity: str,
    app_version: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    height, width = image_shape
    codebrim = cv_bundle["codebrim"]
    sdnet = cv_bundle["sdnet"]
    codebrim_scores = {r["defect"]: float(r["score"]) for r in codebrim}

    gemini = dict(gemini_result or {"status": "unavailable"})
    gemini.setdefault("status", "ok")

    flags = review_flags(
        detected_defects=detected_defects,
        regions=regions,
        sdnet_crack=float(sdnet["crack_probability"]),
        codebrim_scores=codebrim_scores,
        threshold=threshold,
        gemini_result=gemini,
        evidence_state=arbitration["evidence_state"],
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "app_version": app_version,
        "image": {
            "filename": Path(filename or "image").name,
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "size_bytes": len(image_bytes),
            "width_px": int(width),
            "height_px": int(height),
        },
        "configuration": {"detection_threshold": float(threshold)},
        "cv_evidence": {
            "codebrim": [
                {
                    "defect": r["defect"],
                    "score": float(r["score"]),
                    "detected": float(r["score"]) >= threshold,
                }
                for r in codebrim
            ],
            "sdnet2018": {
                "crack_probability": float(sdnet["crack_probability"]),
                "crack_detected": float(sdnet["crack_probability"]) >= threshold,
            },
            "detected_codebrim_defects": list(detected_defects),
            "score_policy": "source-specific, uncalibrated, not averaged",
        },
        "localization": {
            "method": localization.METHOD,
            "note": localization.NOTE,
            "tile_size_px": localization.TILE,
            "tile_stride_px": localization.STRIDE,
            "regions": [
                {
                    "label": r["label"],
                    "source": r["source"],
                    "tile_score_max": float(r["score"]),
                    "tile_count": int(r["tile_count"]),
                    "image_wide": bool(r.get("image_wide", False)),
                    "box_basis": r.get("box_basis", "tile"),
                    "box_xyxy_px": [int(v) for v in r["box"]],
                }
                for r in regions
            ],
        },
        "gemini_evidence": gemini,
        "arbitration": arbitration,
        "recommendations": {
            "worst_severity": worst_severity,
            "references": list(STANDARDS),
            "items": recommendations,
            "disclaimer": DISCLAIMER,
        },
        "review": {
            "needs_engineer_review": True,
            "flags": flags,
        },
        "limits": list(LIMITS),
    }

    return _clean(record)


def to_json_bytes(record_or_records: Any) -> bytes:
    """Pretty UTF-8 JSON for download (a record or a list of records)."""
    return json.dumps(_clean(record_or_records), indent=2, ensure_ascii=False).encode("utf-8")
