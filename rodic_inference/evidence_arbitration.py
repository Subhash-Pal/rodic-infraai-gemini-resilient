
from typing import Any, Dict, List, Optional


EVIDENCE_STATES = [
    "HIGH_AGREEMENT",
    "PARTIAL_AGREEMENT",
    "MODEL_CONFLICT",
    "INSUFFICIENT_EVIDENCE",
]


DEFECT_CLASSES = {
    "crack",
    "spallation",
    "efflorescence",
    "exposed_bars",
    "corrosion_stain",
}


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _extract_cv_labels(cv_result: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(cv_result, dict):
        return []

    labels = []

    # Expected deployment inference format.
    predictions = cv_result.get("predictions", [])

    if isinstance(predictions, list):
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue

            label = prediction.get("class")
            if label is None:
                label = prediction.get("label")

            if label is not None:
                normalized = _normalize_label(label)
                if normalized in DEFECT_CLASSES:
                    labels.append(normalized)

    # Alternative single-label format.
    label = cv_result.get("predicted_class")
    if label is not None:
        normalized = _normalize_label(label)
        if normalized in DEFECT_CLASSES:
            labels.append(normalized)

    return sorted(set(labels))


def _extract_gemini_labels(
    gemini_result: Optional[Dict[str, Any]]
) -> List[str]:
    if not isinstance(gemini_result, dict):
        return []

    labels = []

    observations = gemini_result.get("observations", [])

    if not isinstance(observations, list):
        return []

    for observation in observations:
        if not isinstance(observation, dict):
            continue

        evidence_type = observation.get("evidence_type")

        if evidence_type is None:
            continue

        normalized = _normalize_label(evidence_type)

        if normalized in DEFECT_CLASSES:
            labels.append(normalized)

    return sorted(set(labels))


def arbitrate_evidence(
    cv_result: Optional[Dict[str, Any]],
    gemini_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare source-specific CV and Gemini visual evidence.

    Important:
    - No confidence averaging.
    - No calibrated probability is created.
    - Gemini does not make engineering decisions.
    - Human validation remains mandatory for consequential decisions.
    """

    cv_labels = _extract_cv_labels(cv_result)
    gemini_labels = _extract_gemini_labels(gemini_result)

    cv_available = bool(cv_labels)
    gemini_available = bool(gemini_labels)

    agreement = sorted(set(cv_labels).intersection(gemini_labels))
    cv_only = sorted(set(cv_labels).difference(gemini_labels))
    gemini_only = sorted(set(gemini_labels).difference(cv_labels))

    if not cv_available and not gemini_available:
        state = "INSUFFICIENT_EVIDENCE"
        rationale = (
            "Neither source provided a recognized infrastructure-defect "
            "observation."
        )

    elif cv_available and gemini_available and agreement:
        if not cv_only and not gemini_only:
            state = "HIGH_AGREEMENT"
            rationale = (
                "The computer-vision and multimodal evidence sources "
                "identify the same recognized defect class or classes."
            )
        else:
            state = "PARTIAL_AGREEMENT"
            rationale = (
                "The sources share at least one recognized defect class, "
                "but one or both sources contain additional observations."
            )

    elif cv_available and gemini_available:
        state = "MODEL_CONFLICT"
        rationale = (
            "Both sources provide recognized defect observations, but "
            "there is no overlapping defect class."
        )

    else:
        state = "PARTIAL_AGREEMENT"
        rationale = (
            "Only one evidence source provided a recognized defect "
            "observation; independent corroboration is unavailable."
        )

    return {
        "evidence_state": state,
        "cv_evidence": {
            "recognized_defects": cv_labels,
            "source_available": cv_available,
        },
        "gemini_evidence": {
            "recognized_defects": gemini_labels,
            "source_available": gemini_available,
        },
        "agreement": agreement,
        "cv_only": cv_only,
        "gemini_only": gemini_only,
        "rationale": rationale,
        "confidence_policy": {
            "confidence_averaged": False,
            "calibrated_probability_created": False,
            "source_specific_evidence_preserved": True,
        },
        "human_validation": {
            "required": True,
            "engineering_decision_provided": False,
        },
    }


def build_inspection_evidence_record(
    cv_result: Optional[Dict[str, Any]],
    gemini_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the evidence-grounded inspection record consumed by the UI.
    """

    arbitration = arbitrate_evidence(cv_result, gemini_result)

    return {
        "record_type": "evidence_grounded_infrastructure_inspection",
        "arbitration": arbitration,
        "decision_boundary": (
            "AI organizes and reconciles visual evidence. "
            "Qualified human inspection remains responsible for "
            "engineering significance and final disposition."
        ),
    }
