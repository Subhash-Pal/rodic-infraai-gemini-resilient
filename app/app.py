import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = APP_DIR.parent

if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))
from rodic_vlm.gemini_vlm import analyze_image_with_gemini
from rodic_inference.evidence_arbitration import build_inspection_evidence_record



# ------------------------------------------------------------
# Deployment paths
# ------------------------------------------------------------

from rodic_inference.inference import (
    ingest_image_bytes,
    load_codebrim_model,
    load_sdnet_model,
    run_codebrim_inference,
    run_sdnet_inference,
)


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------


@st.cache_data(show_spinner=False)
def run_gemini_vlm(image_bytes):
    """
    Run Gemini visual evidence extraction.

    Gemini provides visual evidence only.
    It does not provide engineering severity,
    structural safety, or final disposition.
    """
    return analyze_image_with_gemini(image_bytes)


st.set_page_config(
    page_title="RODIC InfraAI Inspection",
    page_icon="🏗️",
    layout="wide",
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .evidence-box {
        padding: 1rem;
        border-radius: 0.6rem;
        border: 1px solid #ddd;
        margin-top: 1rem;
    }

    .human-note {
        padding: 1rem;
        border-left: 5px solid #d97706;
        background-color: #fff7ed;
        border-radius: 0.4rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    '<div class="main-title">'
    'RODIC InfraAI — AI-Assisted Infrastructure Inspection'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Domain-trained computer vision for infrastructure defect screening'
    '</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:

    st.header("Inspection Configuration")

    confidence_threshold = st.slider(
        "Detection threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.divider()

    st.caption(
        "Current POC models"
    )

    st.caption(
        "CODEBRIM — multi-label structural defect screening"
    )

    st.caption(
        "SDNET2018 — crack / no-crack screening"
    )

    st.divider()

    st.caption(
        "Inference runtime: CPU-compatible deployment path"
    )


# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an infrastructure image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Maximum file size: 10 MB.",
)


# ------------------------------------------------------------
# Model cache
# ------------------------------------------------------------
@st.cache_resource
def get_models():

    codebrim_model = load_codebrim_model()
    sdnet_model = load_sdnet_model()

    return codebrim_model, sdnet_model


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    try:

        image_rgb = ingest_image_bytes(
            image_bytes
        )

        st.success(
            "Image validated successfully."
        )

    except Exception as exc:

        st.error(
            f"Image validation failed: {exc}"
        )

        st.stop()


    # --------------------------------------------------------
    # Image preview
    # --------------------------------------------------------
    st.subheader("Inspection Image")

    st.image(
        image_rgb,
        width="stretch",
    )


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------
    with st.spinner(
        "Loading inspection models..."
    ):

        try:

            codebrim_model, sdnet_model = get_models()

        except Exception as exc:

            st.error(
                f"Model loading failed: {exc}"
            )

            st.stop()


    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------
    with st.spinner(
        "Running computer-vision inspection..."
    ):

        try:

            codebrim_results = (
                run_codebrim_inference(
                    codebrim_model,
                    image_rgb,
                )
            )

            sdnet_result = (
                run_sdnet_inference(
                    sdnet_model,
                    image_rgb,
                )
            )

        except Exception as exc:

            st.error(
                f"Inference failed: {exc}"
            )

            st.stop()


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    st.subheader("Computer-Vision Evidence")


    left_column, right_column = st.columns(2)


    # --------------------------------------------------------
    # CODEBRIM results
    # --------------------------------------------------------
    with left_column:

        st.markdown(
            "### CODEBRIM"
        )

        for result in codebrim_results:

            score = float(
                result["score"]
            )

            detected = (
                score >= confidence_threshold
            )

            label = (
                "Detected"
                if detected
                else "Not detected"
            )

            st.write(
                f"**{result['defect']}** — "
                f"{score:.1%} — {label}"
            )

            st.progress(
                min(max(score, 0.0), 1.0)
            )


    # --------------------------------------------------------
    # SDNET results
    # --------------------------------------------------------
    with right_column:

        st.markdown(
            "### SDNET2018"
        )

        crack_probability = float(
            sdnet_result[
                "crack_probability"
            ]
        )

        crack_detected = (
            crack_probability
            >= confidence_threshold
        )

        st.metric(
            "Crack probability",
            f"{crack_probability:.1%}",
        )

        if crack_detected:

            st.warning(
                "Crack evidence detected."
            )

        else:

            st.success(
                "No crack evidence above threshold."
            )


    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------
    st.markdown(
        '<div class="evidence-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Evidence Summary"
    )

    detected_defects = [
        result["defect"]
        for result in codebrim_results
        if float(result["score"])
        >= confidence_threshold
        and result["defect"] != "background"
    ]

    if detected_defects:

        st.write(
            "Potential defect classes identified:"
        )

        for defect in detected_defects:
            st.write(
                f"- {defect}"
            )

    else:

        st.write(
            "No CODEBRIM defect class exceeded "
            "the configured threshold."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------


    # --------------------------------------------------------
    # Gemini VLM — visual evidence extraction
    # --------------------------------------------------------

    st.subheader("Multimodal Visual Evidence")

    with st.spinner("Analyzing image with Gemini VLM..."):
        gemini_result = None
        try:
            gemini_result = run_gemini_vlm(image_bytes)

            st.success(
                f"Gemini VLM completed using "
                f"{gemini_result.get('model', 'configured model')}"
            )

            # Image quality
            image_assessment = gemini_result.get(
                "image_assessment", {}
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Image quality",
                    image_assessment.get(
                        "image_quality",
                        "not reported"
                    )
                )

            with col2:
                st.metric(
                    "Visibility",
                    image_assessment.get(
                        "visibility",
                        "not reported"
                    )
                )

            # Observations
            observations = gemini_result.get(
                "observations",
                []
            )

            st.markdown("### Visual observations")

            if observations:
                for idx, observation in enumerate(
                    observations,
                    start=1
                ):
                    evidence_type = observation.get(
                        "evidence_type",
                        "other"
                    )

                    visual_confidence = observation.get(
                        "visual_confidence",
                        "not reported"
                    )

                    location = observation.get(
                        "location_description",
                        "not specified"
                    )

                    text = observation.get(
                        "observation",
                        "No observation text provided."
                    )

                    st.markdown(
                        f"**{idx}. {evidence_type}** — "
                        f"{text}"
                    )

                    st.caption(
                        f"Location: {location} | "
                        f"Visual confidence: {visual_confidence}"
                    )
            else:
                st.info(
                    "Gemini reported no specific visual observations."
                )

            # Negative evidence
            negative_evidence = gemini_result.get(
                "negative_evidence",
                []
            )

            if negative_evidence:
                st.markdown("### Negative evidence")

                for item in negative_evidence:
                    observation = item.get(
                        "observation",
                        "Not specified"
                    )

                    confidence = item.get(
                        "confidence",
                        "not reported"
                    )

                    st.markdown(
                        f"- {observation} "
                        f"_(confidence: {confidence})_"
                    )

            # Limitations
            limitations = gemini_result.get(
                "limitations",
                []
            )

            if limitations:
                st.markdown("### VLM limitations")

                for limitation in limitations:
                    st.markdown(f"- {limitation}")

            # Explicit decision boundary
            engineering_decision = gemini_result.get(
                "engineering_decision",
                {}
            )

            st.warning(
                "Engineering decision: "
                + str(
                    engineering_decision.get(
                        "status",
                        "not_provided"
                    )
                )
            )

            st.caption(
                engineering_decision.get(
                    "reason",
                    "Engineering significance and final "
                    "disposition require qualified human validation."
                )
            )

        except Exception as exc:
            st.error(
                "Gemini VLM was unavailable for this image. "
                "The domain CV results remain available."
            )

            st.caption(
                f"VLM runtime detail: {type(exc).__name__}"
            )

    # --------------------------------------------------------
    # Evidence arbitration
    # --------------------------------------------------------
    try:
        cv_evidence_for_arbitration = {
            "predictions": [
                {"class": defect}
                for defect in detected_defects
            ]
        }

        inspection_evidence = build_inspection_evidence_record(
            cv_evidence_for_arbitration,
            gemini_result,
        )

        arbitration = inspection_evidence["arbitration"]

        st.subheader("Evidence Arbitration")

        evidence_state = arbitration["evidence_state"]

        state_labels = {
            "HIGH_AGREEMENT": "High agreement",
            "PARTIAL_AGREEMENT": "Partial agreement",
            "MODEL_CONFLICT": "Model conflict",
            "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
        }

        st.markdown(
            f"**Evidence state:** "
            f"{state_labels.get(evidence_state, evidence_state)}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Computer-vision evidence**")

            cv_defects = arbitration["cv_evidence"][
                "recognized_defects"
            ]

            if cv_defects:
                for defect in cv_defects:
                    st.write(f"- {defect}")
            else:
                st.write("No recognized defect evidence.")

        with col2:
            st.markdown("**Gemini visual evidence**")

            gemini_defects = arbitration["gemini_evidence"][
                "recognized_defects"
            ]

            if gemini_defects:
                for defect in gemini_defects:
                    st.write(f"- {defect}")
            else:
                st.write("No recognized defect evidence.")

        if arbitration["agreement"]:
            st.markdown(
                "**Agreement:** "
                + ", ".join(arbitration["agreement"])
            )

        if arbitration["cv_only"]:
            st.markdown(
                "**CV-only observations:** "
                + ", ".join(arbitration["cv_only"])
            )

        if arbitration["gemini_only"]:
            st.markdown(
                "**Gemini-only observations:** "
                + ", ".join(arbitration["gemini_only"])
            )

        st.caption(arbitration["rationale"])

        st.info(
            "Evidence arbitration preserves source-specific evidence. "
            "It does not average confidence values or create calibrated "
            "probabilities."
        )

    except Exception as arbitration_exc:
        st.error(
            "Evidence arbitration was unavailable for this image."
        )

        st.caption(
            f"Arbitration runtime detail: "
            f"{type(arbitration_exc).__name__}"
        )

    # Human validation boundary
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="human-note">
        <strong>Human validation required</strong><br>
        These results are AI-generated inspection evidence.
        They are intended to support inspection workflows and
        prioritization, not to replace qualified engineering
        judgement or formal structural assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )


else:

    st.info(
        "Upload an infrastructure image to begin inspection."
    )


st.markdown(
    "### Evidence boundary"
)

st.info(
    "Gemini is used as a multimodal visual-evidence source. "
    "Its observations are not treated as calibrated probabilities "
    "and are not converted directly into structural safety, "
    "engineering severity, or final disposition."
)
