import json

import pytest

from rodic_inference import pipeline
from rodic_inference.report_json import SCHEMA_VERSION, to_json_bytes
from rodic_vlm.errors import GeminiUnavailable
from rodic_vlm.key_rotation import unavailable_result
from tests.helpers import blank_image, gemini_ok, make_bundle

DATA = b"fake-image-bytes"


def run(bundle=None, gemini=None, threshold=0.5, filename="wall.jpg"):
    return pipeline.assemble(
        filename=filename,
        image_bytes=DATA,
        image_rgb=blank_image(),
        bundle=bundle or make_bundle(),
        gemini_result=gemini or gemini_ok(),
        threshold=threshold,
    )


def test_record_has_schema_and_is_json_serialisable():
    out = run(make_bundle(codebrim={"crack": 0.9}, sdnet=0.9))
    rec = out["record"]

    assert rec["schema_version"] == SCHEMA_VERSION
    assert {"image", "cv_evidence", "localization", "gemini_evidence", "arbitration",
            "recommendations", "review", "limits"} <= set(rec)
    assert json.loads(to_json_bytes(rec))["image"]["filename"] == "wall.jpg"
    assert rec["review"]["needs_engineer_review"] is True


def test_image_is_never_embedded_only_its_hash():
    import hashlib

    rec = run()["record"]

    assert rec["image"]["sha256"] == hashlib.sha256(DATA).hexdigest()
    assert "fake-image-bytes" not in json.dumps(rec)


def test_filename_path_is_stripped():
    assert run(filename="../../etc/passwd.jpg")["record"]["image"]["filename"] == "passwd.jpg"


def test_detected_crack_gives_provisional_recommendation_and_regions():
    out = run(make_bundle(codebrim={"crack": 0.9}, sdnet=0.9))
    rec = out["record"]

    assert out["regions"] and rec["localization"]["regions"][0]["box_xyxy_px"]
    classes = [i["defect_class"] for i in rec["recommendations"]["items"]]
    assert "crack" in classes
    assert "crack_width_not_measured" in rec["review"]["flags"]
    assert rec["recommendations"]["disclaimer"] and rec["recommendations"]["references"]


def test_clean_image_gets_a_no_defect_entry_and_no_regions():
    out = run(make_bundle(sdnet=0.02), gemini=gemini_ok("other"))

    assert out["regions"] == []
    assert len(out["record"]["recommendations"]["items"]) == 1


def test_models_disagreeing_on_crack_is_flagged():
    flags = run(make_bundle(codebrim={"crack": 0.05}, sdnet=0.95))["record"]["review"]["flags"]

    assert "cv_models_disagree_on_crack" in flags


def test_unavailable_gemini_continues_cv_only_and_is_flagged():
    gem = unavailable_result(GeminiUnavailable("all keys exhausted", attempts=[{"key": "K0", "model": "m", "outcome": "quota"}]))
    out = run(make_bundle(codebrim={"crack": 0.9}, sdnet=0.9), gemini=gem)
    rec = out["record"]

    assert rec["gemini_evidence"]["status"] == "unavailable"
    assert "gemini_unavailable_cv_only" in rec["review"]["flags"]
    assert out["detected_defects"] == ["crack"]
    assert rec["gemini_evidence"]["rotation_attempts"][0]["outcome"] == "quota"


def test_nan_scores_become_null_not_invalid_json():
    bundle = make_bundle(codebrim={"crack": float("nan")}, sdnet=float("inf"))
    text = to_json_bytes(run(bundle)["record"]).decode()

    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_resolve_gemini_never_raises_and_marks_status():
    def boom(_b):
        raise GeminiUnavailable("no keys", attempts=[])

    assert pipeline.resolve_gemini(boom, DATA)["status"] == "unavailable"
    assert pipeline.resolve_gemini(None, DATA)["status"] == "skipped"
    assert pipeline.resolve_gemini(lambda b: {"observations": []}, DATA)["status"] == "ok"


def test_batch_json_is_a_list_of_records():
    recs = [run(filename=f"f{i}.jpg")["record"] for i in range(3)]
    parsed = json.loads(to_json_bytes(recs))

    assert [r["image"]["filename"] for r in parsed] == ["f0.jpg", "f1.jpg", "f2.jpg"]


def test_sdnet_only_crack_counts_as_cv_evidence_for_arbitration():
    out = run(make_bundle(codebrim={"crack": 0.05}, sdnet=0.95), gemini=gemini_ok("crack"))

    assert out["detected_defects"] == []                       # CODEBRIM list unchanged
    assert "crack" in out["arbitration"]["cv_evidence"]["recognized_defects"]
    assert out["arbitration"]["evidence_state"] != "INSUFFICIENT_EVIDENCE"
