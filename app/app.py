import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = APP_DIR.parent

if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from rodic_inference.inference import (
    ingest_image_bytes,
    load_codebrim_model,
    load_sdnet_model,
)
from rodic_inference.irc_recommender import DISCLAIMER, STANDARDS
from rodic_inference.pipeline import assemble, compute_cv_bundle, resolve_gemini
from rodic_inference.report_json import to_json_bytes
from rodic_inference.version import APP_VERSION
from rodic_vlm.key_rotation import analyze_with_rotation, get_default_keyring

MAX_FILES = 10


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="RODIC InfraAI Inspection",
    page_icon="🏗️",
    layout="wide",
)


# ------------------------------------------------------------
# Cached work
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_models():
    return load_codebrim_model(), load_sdnet_model()


@st.cache_data(show_spinner=False, max_entries=32)
def cv_bundle_cached(image_bytes):
    """Whole-image + tile scores for both models. Threshold-independent, so moving the
    slider re-uses this and only recomputes boxes."""
    image_rgb = ingest_image_bytes(image_bytes)
    return compute_cv_bundle(get_models(), image_rgb)


@st.cache_data(show_spinner=False, max_entries=64)
def run_gemini_vlm(image_bytes):
    """Gemini visual evidence with key rotation.

    Gemini provides visual evidence only: no engineering severity, structural safety or
    final disposition. A failure raises, and Streamlit does not cache exceptions, so the
    'Retry Gemini' button (or any rerun) tries again for images that had no Gemini evidence.
    """
    return analyze_with_rotation(image_bytes)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    .evidence-box {
        padding: 1rem; border-radius: 0.6rem; border: 1px solid #ddd; margin-top: 1rem;
    }
    .human-note {
        padding: 1rem; border-left: 5px solid #d97706; background-color: #fff7ed;
        border-radius: 0.4rem; margin-top: 1rem; color: #3b2a10;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">RODIC InfraAI — AI-Assisted Infrastructure Inspection</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Domain-trained computer vision for infrastructure defect screening</div>',
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

    draw_boxes = st.checkbox("Draw defect regions on the image", value=True)
    use_gemini = st.checkbox(
        "Use Gemini visual evidence",
        value=True,
        help="Switch off to save API quota; the result is then computer-vision evidence only.",
    )

    st.divider()
    st.caption("Current POC models")
    st.caption("CODEBRIM — multi-label structural defect screening")
    st.caption("SDNET2018 — crack / no-crack screening")
    st.caption("Region legend: CB = CODEBRIM tile score, SD = SDNET2018 tile score")

    st.divider()

    try:
        from rodic_vlm.gemini_vlm import get_model_chain

        ring = get_default_keyring()
        states = ring.status(get_model_chain()[0])
        ready = sum(1 for s in states.values() if s == "ready")
        st.caption(f"Gemini keys configured: {len(states)} (ready for primary model: {ready})")
    except Exception:
        st.caption("Gemini key status unavailable.")

    st.caption("Inference runtime: CPU-compatible deployment path")
    st.caption(f"App version {APP_VERSION}")


# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload infrastructure images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help=f"Up to {MAX_FILES} images per run, maximum 10 MB each.",
)


# ------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------
def render_cv(codebrim, sdnet, threshold):
    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown("### CODEBRIM")

        for result in codebrim:
            score = float(result["score"])
            label = "Detected" if score >= threshold else "Not detected"

            st.write(f"**{result['defect']}** — {score:.1%} — {label}")
            st.progress(min(max(score, 0.0), 1.0))

    with right_column:
        st.markdown("### SDNET2018")

        crack_probability = float(sdnet["crack_probability"])

        st.metric("Crack probability", f"{crack_probability:.1%}")

        if crack_probability >= threshold:
            st.warning("Crack evidence detected.")
        else:
            st.success("No crack evidence above threshold.")


def render_regions(regions):
    if not regions:
        st.caption("No region reached the detection threshold.")
        return

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        note = " — image-wide (not localized)" if region.get("image_wide") else ""

        st.write(
            f"- **{region['label']}** [{region['source']}] "
            f"tile score {region['score']:.2f}, box ({x1}, {y1}) to ({x2}, {y2}) px{note}"
        )

    st.caption(
        "Regions come from classifier scores on 224 px tiles; crack boxes are tightened with a "
        "classical dark-line search. This is not a trained object detector and is not validated."
    )


def render_gemini(gemini_result):
    status = gemini_result.get("status", "ok")

    if status == "skipped":
        st.info("Gemini was switched off for this run. Evidence below is computer vision only.")
        return

    if status == "unavailable":
        st.warning(
            "Gemini VLM was unavailable for this image, so this result is computer-vision "
            "evidence only. Use **Retry Gemini** above to try again."
        )

        attempts = gemini_result.get("rotation_attempts", [])

        st.caption(
            f"Reason: {gemini_result.get('reason', 'unknown')}. "
            "Rate-limited keys recover in about a minute (a daily quota in about an hour)."
        )

        if attempts:
            st.caption(
                "Attempts: "
                + "; ".join(f"{a['key']} / {a['model']}: {a['outcome']}" for a in attempts[:12])
            )

        return

    attempts = gemini_result.get("rotation_attempts", [])
    skipped_keys = sum(1 for a in attempts if a["outcome"] not in ("ok",))

    st.success(
        f"Gemini VLM completed using {gemini_result.get('model', 'configured model')}"
        + (f" after skipping {skipped_keys} unavailable key/model attempt(s)." if skipped_keys else "")
    )

    image_assessment = gemini_result.get("image_assessment", {})

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Image quality", image_assessment.get("image_quality", "not reported"))

    with col2:
        st.metric("Visibility", image_assessment.get("visibility", "not reported"))

    st.markdown("### Visual observations")

    observations = gemini_result.get("observations", [])

    if observations:
        for idx, observation in enumerate(observations, start=1):
            st.markdown(
                f"**{idx}. {observation.get('evidence_type', 'other')}** — "
                f"{observation.get('observation', 'No observation text provided.')}"
            )
            st.caption(
                f"Location: {observation.get('location_description', 'not specified')} | "
                f"Visual confidence: {observation.get('visual_confidence', 'not reported')}"
            )
    else:
        st.info("Gemini reported no specific visual observations.")

    negative_evidence = gemini_result.get("negative_evidence", [])

    if negative_evidence:
        st.markdown("### Negative evidence")

        for item in negative_evidence:
            st.markdown(
                f"- {item.get('observation', 'Not specified')} "
                f"_(confidence: {item.get('confidence', 'not reported')})_"
            )

    limitations = gemini_result.get("limitations", [])

    if limitations:
        st.markdown("### VLM limitations")

        for limitation in limitations:
            st.markdown(f"- {limitation}")

    decision = gemini_result.get("engineering_decision", {})

    st.warning("Engineering decision: " + str(decision.get("status", "not_provided")))
    st.caption(
        decision.get(
            "reason",
            "Engineering significance and final disposition require qualified human validation.",
        )
    )


def render_arbitration(arbitration):
    st.subheader("Evidence Arbitration")

    state_labels = {
        "HIGH_AGREEMENT": "High agreement",
        "PARTIAL_AGREEMENT": "Partial agreement",
        "MODEL_CONFLICT": "Model conflict",
        "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
    }

    state = arbitration["evidence_state"]

    st.markdown(f"**Evidence state:** {state_labels.get(state, state)}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Computer-vision evidence**")
        defects = arbitration["cv_evidence"]["recognized_defects"]

        if defects:
            for defect in defects:
                st.write(f"- {defect}")
        else:
            st.write("No recognized defect evidence.")

    with col2:
        st.markdown("**Gemini visual evidence**")
        defects = arbitration["gemini_evidence"]["recognized_defects"]

        if defects:
            for defect in defects:
                st.write(f"- {defect}")
        else:
            st.write("No recognized defect evidence.")

    if arbitration["agreement"]:
        st.markdown("**Agreement:** " + ", ".join(arbitration["agreement"]))

    if arbitration["cv_only"]:
        st.markdown("**CV-only observations:** " + ", ".join(arbitration["cv_only"]))

    if arbitration["gemini_only"]:
        st.markdown("**Gemini-only observations:** " + ", ".join(arbitration["gemini_only"]))

    st.caption(arbitration["rationale"])
    st.info(
        "Evidence arbitration preserves source-specific evidence. "
        "It does not average confidence values or create calibrated probabilities."
    )


def render_recommendations(recommendations, worst):
    st.subheader("Indicative repair guidance (IRC / MoRTH-referenced)")

    st.warning(DISCLAIMER)
    st.markdown(f"**Highest indicative severity in this image:** {worst}")

    for rec in recommendations:
        st.markdown(
            f"**{rec.defect_class}** — indicative severity: **{rec.severity}** "
            f"({rec.severity_basis})"
        )

        for action in rec.actions:
            st.write(f"- {action}")

        for note in rec.notes:
            st.caption(note)

    with st.expander("Standards the engineer should consult"):
        for reference in STANDARDS:
            st.write(f"- {reference}")

        st.caption(
            "Standards are named, not quoted. Clause-level guidance must be confirmed against "
            "the current IRC / MoRTH editions by a qualified engineer."
        )


def render_human_note():
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


def render_image_result(index, name, data, result, threshold):
    record = result["record"]

    tab_boxes, tab_original = st.tabs(["Annotated (defect regions)", "Original"])

    with tab_boxes:
        st.image(result["annotated_rgb"], width="stretch")
        render_regions(result["regions"])

    with tab_original:
        st.image(ingest_image_bytes(data), width="stretch")

    st.subheader("Computer-Vision Evidence")
    render_cv(
        record["cv_evidence"]["codebrim"],
        record["cv_evidence"]["sdnet2018"],
        threshold,
    )

    st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
    st.markdown("### Evidence Summary")

    detected = result["detected_defects"]

    if detected:
        st.write("Potential defect classes identified:")

        for defect in detected:
            st.write(f"- {defect}")
    else:
        st.write("No CODEBRIM defect class exceeded the configured threshold.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Multimodal Visual Evidence")
    render_gemini(record["gemini_evidence"])

    render_arbitration(result["arbitration"])

    render_recommendations(
        result["recommendations"],
        record["recommendations"]["worst_severity"],
    )

    flags = record["review"]["flags"]

    st.markdown("**Review flags:** " + ", ".join(f"`{f}`" for f in flags))

    st.download_button(
        "Download this image's JSON",
        data=to_json_bytes(record),
        file_name=f"{Path(name).stem}_inspection.json",
        mime="application/json",
        key=f"json_{index}",
    )

    render_human_note()


# ------------------------------------------------------------
# Main flow
# ------------------------------------------------------------
if uploaded_files:
    files = list(uploaded_files)

    if len(files) > MAX_FILES:
        st.warning(f"Only the first {MAX_FILES} images are analysed in one run.")
        files = files[:MAX_FILES]

    with st.spinner("Loading inspection models..."):
        try:
            get_models()
        except Exception as exc:
            st.error(f"Model loading failed: {exc}")
            st.stop()

    results = []
    progress = st.progress(0.0, text="Analysing images...")

    for position, uploaded in enumerate(files):
        data = uploaded.getvalue()
        name = uploaded.name

        try:
            image_rgb = ingest_image_bytes(data)
            bundle = cv_bundle_cached(data)

            if use_gemini:
                gemini_result = resolve_gemini(run_gemini_vlm, data)
            else:
                gemini_result = resolve_gemini(None, data)

            analysis = assemble(
                filename=name,
                image_bytes=data,
                image_rgb=image_rgb,
                bundle=bundle,
                gemini_result=gemini_result,
                threshold=confidence_threshold,
                draw_boxes=draw_boxes,
            )

            results.append((name, data, analysis, None))

        except Exception as exc:
            results.append((name, data, None, f"{type(exc).__name__}: {exc}"))

        progress.progress((position + 1) / len(files), text=f"Analysed {position + 1} of {len(files)}")

    progress.empty()

    # ---------------- batch summary ----------------
    st.subheader(f"Batch summary — {len(results)} image(s)")

    rows = []
    failed_gemini = 0

    for name, _data, analysis, error in results:
        if analysis is None:
            rows.append({"File": name, "Status": f"Failed: {error}"})
            continue

        record = analysis["record"]
        gemini_status = record["gemini_evidence"].get("status", "ok")

        if gemini_status == "unavailable":
            failed_gemini += 1

        rows.append(
            {
                "File": name,
                "CODEBRIM defects": ", ".join(analysis["detected_defects"]) or "none above threshold",
                "SDNET crack prob.": f"{record['cv_evidence']['sdnet2018']['crack_probability']:.1%}",
                "Regions": len(analysis["regions"]),
                "Gemini": gemini_status,
                "Evidence state": analysis["arbitration"]["evidence_state"],
                "Indicative severity": record["recommendations"]["worst_severity"],
                "Review flags": len(record["review"]["flags"]),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)

    good_records = [a["record"] for _n, _d, a, _e in results if a is not None]

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if good_records:
            st.download_button(
                f"Download all JSON ({len(good_records)} images)",
                data=to_json_bytes(good_records),
                file_name="rodic_inspection_batch.json",
                mime="application/json",
                key="json_all",
            )

    with action_col2:
        if failed_gemini:
            st.button(
                f"Retry Gemini for {failed_gemini} image(s)",
                help="Images with Gemini evidence are cached and not re-sent.",
            )

    # ---------------- per-image detail ----------------
    for index, (name, data, analysis, error) in enumerate(results):
        if analysis is None:
            st.error(f"{name}: {error}")
            continue

        with st.expander(name, expanded=(len(results) == 1)):
            render_image_result(index, name, data, analysis, confidence_threshold)

else:
    st.info("Upload one or more infrastructure images to begin inspection.")


st.markdown("### Evidence boundary")

st.info(
    "Gemini is used as a multimodal visual-evidence source. "
    "Its observations are not treated as calibrated probabilities "
    "and are not converted directly into structural safety, "
    "engineering severity, or final disposition."
)
